#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))
source("lib/R/gcrmapred.R")
source("lib/R/rmapred.R")
source("config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
num_tr_subset <- as.integer(cmd_args[1])
id_type <- cmd_args[2]
if (id_type == "gene") {
    cdfname <- "hgu133plus2hsentrezg"
} else {
    cdfname <- "hgu133plus2"
}
dataset_tr_name_combos <- combn(dataset_names, num_tr_subset)
for (col in 1:ncol(dataset_tr_name_combos)) {
    no_raw_data <- FALSE
    for (dataset_tr_name in dataset_tr_name_combos[,col]) {
        if (!dir.exists(paste0("data/raw/", dataset_tr_name))) {
            no_raw_data <- TRUE
            break
        }
    }
    if (no_raw_data) next
    # load a reference eset set
    for (norm_type in norm_types) {
        suffixes <- c(norm_type)
        if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
        if (num_tr_subset > 1) {
            eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes, "merged", "tr"), collapse="_")
        }
        else {
            eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes), collapse="_")
        }
        eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
        if (file.exists(eset_tr_file) & !exists(eset_tr_name)) {
            print(paste("Loading:", eset_tr_name), quote=FALSE)
            load(eset_tr_file)
            for (dataset_te_name in setdiff(dataset_names, dataset_tr_name_combos[,col])) {
                if (!dir.exists(paste0("data/raw/", dataset_te_name))) next
                eset_te_name <- paste0(c("eset", dataset_te_name, suffixes), collapse="_")
                if (!exists(eset_te_name)) {
                    print(paste("Loading:", eset_te_name), quote=FALSE)
                    load(paste0("data/", eset_te_name, ".Rda"))
                }
            }
            break
        }
    }
}
if ("gcrma" %in% cmd_args[3:length(cmd_args)]) {
    print(paste("Affinities:", cdfname), quote=FALSE)
    affinities <- compute.affinities(cdfname, verbose=TRUE)
}
for (col in 1:ncol(dataset_tr_name_combos)) {
    cel_tr_files <- c()
    for (dataset_tr_name in dataset_tr_name_combos[,col]) {
        cel_tr_dir <- paste0("data/raw/", dataset_tr_name)
        if (dir.exists(cel_tr_dir)) {
            cel_tr_files <- append(cel_tr_files, list.files(path=cel_tr_dir, full.names=TRUE, pattern="\\.CEL$"))
        }
        else {
            cel_tr_files <- NULL
            break
        }
    }
    if (is.null(cel_tr_files)) next
    for (norm_type in norm_types) {
        suffixes <- c(norm_type)
        if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
        if (num_tr_subset > 1) {
            eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes, "merged", "tr"), collapse="_")
        }
        else {
            eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes), collapse="_")
        }
        if (exists(eset_tr_name)) {
            ref_suffixes <- suffixes
            break
        }
    }
    print(paste0(c("Creating AffyBatch:", dataset_tr_name_combos[,col]), collapse=" "), quote=FALSE)
    affybatch_tr <- ReadAffy(filenames=cel_tr_files, cdfname=cdfname, verbose=TRUE)
    for (norm_type in cmd_args[3:length(cmd_args)]) {
        suffixes <- c(norm_type)
        if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
        eset_tr_norm_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes, "tr"), collapse="_")
        print(paste("Creating:", eset_tr_norm_name), quote=FALSE)
        if (norm_type == "gcrma") {
            norm_obj <- gcrmatrain(affybatch_tr, affinities)
        }
        else if (norm_type == "rma") {
            norm_obj <- rmatrain(affybatch_tr)
        }
        rownames(norm_obj$xnorm) <- sub("\\.CEL$", "", rownames(norm_obj$xnorm))
        if (id_type == "gene") {
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
            cel_te_dir <- paste0("data/raw/", dataset_te_name)
            if (!dir.exists(cel_te_dir)) next
            cel_te_files <- list.files(path=cel_te_dir, full.names=TRUE, pattern="\\.CEL$")
            print(paste("Creating AffyBatch:", dataset_te_name), quote=FALSE)
            affybatch_te <- ReadAffy(filenames=cel_te_files, cdfname=cdfname, verbose=TRUE)
            eset_te_name <- paste0(c("eset", dataset_te_name, ref_suffixes), collapse="_")
            eset_te_norm_name <- paste0(eset_tr_norm_name, "_", dataset_te_name, "_te")
            print(paste("Creating:", eset_te_norm_name), quote=FALSE)
            if (norm_type == "gcrma") {
                xnorm_te <- gcrmaaddon(norm_obj, affybatch_te, affinities)
            }
            else if (norm_type == "rma") {
                xnorm_te <- rmaaddon(norm_obj, affybatch_te)
            }
            rownames(xnorm_te) <- sub("\\.CEL$", "", rownames(xnorm_te))
            if (id_type == "gene") {
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
