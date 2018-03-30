#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))
source("lib/R/gcrmapred.R")
source("lib/R/rmapred.R")
source("lib/R/config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
num_tr_subset <- as.integer(cmd_args[1])
cdfname <- cmd_args[2]
dataset_tr_name_combos <- combn(dataset_names, num_tr_subset)
for (col in 1:ncol(dataset_tr_name_combos)) {
    eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col]), collapse="_")
    print(paste("Loading:", eset_tr_name))
    load(paste0("data/", eset_tr_name, ".Rda"))
}
for (dataset_te_name in dataset_names) {
    eset_te_name <- paste0("eset_", dataset_te_name)
    print(paste("Loading:", eset_te_name))
    load(paste0("data/", eset_te_name, ".Rda"))
}
if ("gcrma" %in% cmd_args[3:length(cmd_args)]) {
    affinities <- compute.affinities(cdfname, verbose=TRUE)
}
for (col in 1:ncol(dataset_tr_name_combos)) {
    cel_files_tr <- c()
    for (dataset_tr_name in dataset_tr_name_combos[,col]) {
        cel_path_tr <- paste0("data/raw/", dataset_tr_name)
        if (dir.exists(cel_path_tr)) {
            cel_files_tr <- append(cel_files_tr, list.files(path=cel_path_tr, full.names=TRUE, pattern="\\.CEL$"))
        }
        else {
            cel_files_tr <- NULL
            break
        }
    }
    if (is.null(cel_files_tr)) next
    print(paste0(c("Creating AffyBatch:", dataset_tr_name_combos[,col]), collapse=" "))
    affybatch_tr <- ReadAffy(filenames=cel_files_tr, cdfname=cdfname, verbose=TRUE)
    eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col]), collapse="_")
    for (norm_type in cmd_args[3:length(cmd_args)]) {
        eset_tr_norm_name <- paste0(c(eset_tr_name, norm_type, "tr"), collapse="_")
        print(paste("Creating:", eset_tr_norm_name))
        if (norm_type == "gcrma") {
            norm_obj <- gcrmatrain(affybatch_tr, affinities)
        }
        else if (norm_type == "rma") {
            norm_obj <- rmatrain(affybatch_tr)
        }
        rownames(norm_obj$xnorm) <- sub("\\.CEL$", "", rownames(norm_obj$xnorm))
        if (cdfname == "hgu133plus2hsentrezg") {
            eset_tr_norm <- ExpressionSet(
                assayData=t(norm_obj$xnorm),
                phenoData=phenoData(get(eset_tr_name)),
                featureData=AnnotatedDataFrame(data.frame(
                    Symbol=getSYMBOL(featureNames(eset_tr_norm), paste0(cdfname, ".db"))
                )),
                annotation=cdfname
            )
        }
        else {
            eset_tr_norm <- get(eset_tr_name)
            exprs(eset_tr_norm) <- t(norm_obj$xnorm)
        }
        assign(eset_tr_norm_name, eset_tr_norm)
        save(list=eset_tr_norm_name, file=paste0("data/", eset_tr_norm_name, ".Rda"))
        eset_tr_norm_obj_name <- paste0(eset_tr_norm_name, "_obj")
        assign(eset_tr_norm_obj_name, norm_obj)
        save(list=eset_tr_norm_obj_name, file=paste0("data/", eset_tr_norm_obj_name, ".Rda"))
        for (dataset_te_name in setdiff(dataset_names, dataset_tr_name_combos[,col])) {
            cel_path_te <- paste0("data/raw/", dataset_te_name)
            if (!dir.exists(cel_path_te)) next
            cel_files_te <- list.files(path=cel_path_te, full.names=TRUE, pattern="\\.CEL$")
            eset_te_name <- paste0("eset_", dataset_te_name)
            eset_te_norm_name <- paste0(eset_tr_norm_name, "_", dataset_te_name, "_te")
            print(paste("Creating AffyBatch:", dataset_te_name))
            affybatch_te <- ReadAffy(filenames=cel_files_te, cdfname=cdfname, verbose=TRUE)
            print(paste("Creating:", eset_te_norm_name))
            if (norm_type == "gcrma") {
                xnorm_te <- gcrmaaddon(norm_obj, affybatch_te, affinities)
            }
            else if (norm_type == "rma") {
                xnorm_te <- rmaaddon(norm_obj, affybatch_te)
            }
            rownames(xnorm_te) <- sub("\\.CEL$", "", rownames(xnorm_te))
            if (cdfname == "hgu133plus2hsentrezg") {
                eset_te_norm <- ExpressionSet(
                    assayData=t(xnorm_te),
                    phenoData=phenoData(get(eset_te_name)),
                    featureData=AnnotatedDataFrame(data.frame(
                        Symbol=getSYMBOL(featureNames(eset_te_norm), paste0(cdfname, ".db"))
                    )),
                    annotation=cdfname
                )
            }
            else {
                eset_te_norm <- get(eset_te_name)
                exprs(eset_te_norm) <- t(xnorm_te)
            }
            assign(eset_te_norm_name, eset_te_norm)
            save(list=eset_te_norm_name, file=paste0("data/", eset_te_norm_name, ".Rda"))
            remove(list=c(eset_te_norm_name))
        }
        remove(list=c(eset_tr_norm_obj_name, eset_tr_norm_name))
    }
}
