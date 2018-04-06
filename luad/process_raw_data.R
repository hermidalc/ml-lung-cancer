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
            cat("Loading:", eset_tr_name, "\n")
            load(eset_tr_file)
            for (dataset_te_name in setdiff(dataset_names, dataset_tr_name_combos[,col])) {
                if (!dir.exists(paste0("data/raw/", dataset_te_name))) next
                eset_te_name <- paste0(c("eset", dataset_te_name, suffixes), collapse="_")
                if (!exists(eset_te_name)) {
                    cat("Loading:", eset_te_name, "\n")
                    load(paste0("data/", eset_te_name, ".Rda"))
                }
            }
            break
        }
    }
}
if ("gcrma" %in% cmd_args[3:length(cmd_args)]) {
    cat("Affinities:", cdfname, "\n")
    affinities <- compute.affinities(cdfname, verbose=TRUE)
}
norm_obj_cache <- list()
affybatch_cache <- list()
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
    dataset_tr_name_str <- paste0(dataset_tr_name_combos[,col], collapse="_")
    if (length(dataset_tr_name_combos[,col]) > 1 || !exists(dataset_tr_name_str, where=norm_obj_cache)) {
        cat(paste0(c("Creating AffyBatch:", dataset_tr_name_combos[,col]), collapse=" "), "\n")
        affybatch_tr <- ReadAffy(filenames=cel_tr_files, cdfname=cdfname, verbose=TRUE)
    }
    for (norm_type in cmd_args[3:length(cmd_args)]) {
        suffixes <- c(norm_type)
        if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
        eset_tr_norm_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes, "tr"), collapse="_")
        cat("Creating:", eset_tr_norm_name, "\n")
        if (
            length(dataset_tr_name_combos[,col]) == 1 &&
            exists(dataset_tr_name_str, where=norm_obj_cache) &&
            exists(norm_type, where=norm_obj_cache[[dataset_tr_name_str]])
        ) {
            cat("Loading cached norm object:", paste0(dataset_tr_name_str, "_", norm_type), "\n")
            norm_obj <- norm_obj_cache[[dataset_tr_name_str]][[norm_type]]
        }
        else if (norm_type == "gcrma") {
            norm_obj <- gcrmatrain(affybatch_tr, affinities)
        }
        else if (norm_type == "rma") {
            norm_obj <- rmatrain(affybatch_tr)
        }
        if (
            length(dataset_tr_name_combos[,col]) == 1 &&
            !exists(dataset_tr_name_str, where=norm_obj_cache) &&
            !exists(norm_type, where=norm_obj_cache[[dataset_tr_name_str]])
        ) norm_obj_cache[[dataset_tr_name_str]][[norm_type]] <- norm_obj
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
            if (!exists(dataset_te_name, where=affybatch_cache)) {
                cel_te_dir <- paste0("data/raw/", dataset_te_name)
                if (!dir.exists(cel_te_dir)) next
                cat("Creating AffyBatch:", dataset_te_name, "\n")
                cel_te_files <- list.files(path=cel_te_dir, full.names=TRUE, pattern="\\.CEL$")
                affybatch_te <- ReadAffy(filenames=cel_te_files, cdfname=cdfname, verbose=TRUE)
                bg_correct_te <- TRUE
            }
            else {
                cat("Loading cached background corrected AffyBatch:", dataset_te_name, "\n")
                affybatch_te <- affybatch_cache[[dataset_te_name]]
                bg_correct_te <- FALSE
            }
            eset_te_name <- paste0(c("eset", dataset_te_name, ref_suffixes), collapse="_")
            eset_te_norm_name <- paste0(eset_tr_norm_name, "_", dataset_te_name, "_te")
            cat("Creating:", eset_te_norm_name, "\n")
            if (norm_type == "gcrma") {
                normaddon_obj <- gcrmaaddon(norm_obj, affybatch_te, affinities, bg.correct=bg_correct_te)
            }
            else if (norm_type == "rma") {
                normaddon_obj <- rmaaddon(norm_obj, affybatch_te, bg.correct=bg_correct_te)
            }
            if (!exists(dataset_te_name, where=affybatch_cache))
                affybatch_cache[[dataset_te_name]] <- normaddon_obj$abg
            rownames(normaddon_obj$xnorm) <- sub("\\.CEL$", "", rownames(normaddon_obj$xnorm))
            if (id_type == "gene") {
                eset_te_norm <- ExpressionSet(
                    assayData=t(normaddon_obj$xnorm),
                    phenoData=phenoData(get(eset_te_name)),
                    featureData=AnnotatedDataFrame(data.frame(
                        Symbol=getSYMBOL(featureNames(eset_te_norm), paste0(cdfname, ".db"))
                    )),
                    annotation=cdfname
                )
            }
            else {
                eset_te_norm <- get(eset_te_name)
                exprs(eset_te_norm) <- t(normaddon_obj$xnorm)
            }
            assign(eset_te_norm_name, eset_te_norm)
            save(list=eset_te_norm_name, file=paste0("data/", eset_te_norm_name, ".Rda"))
            remove(list=c(eset_te_norm_name))
        }
        remove(list=c(eset_tr_norm_obj_name, eset_tr_norm_name))
    }
}
