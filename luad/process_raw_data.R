#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))
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
affybatch_cache <- list()
norm_obj_cache <- list()
for (col in 1:ncol(dataset_tr_name_combos)) {
    skip_processing <- FALSE
    for (dataset_tr_name in dataset_tr_name_combos[,col]) {
        if (!dir.exists(paste0("data/raw/", dataset_tr_name)) {
            skip_processing <- TRUE
            break
        }
    }
    if (skip_processing) next
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
    for (norm_type in cmd_args[3:length(cmd_args)]) {
        suffixes <- c(norm_type)
        if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
        affybatch_tr <- NULL
        for (dataset_tr_name in dataset_tr_name_combos[,col]) {
            if (exists(dataset_tr_name, where=affybatch_cache)) {
                cat("Loading cached AffyBatch:", dataset_tr_name, "\n")
                affybatch <- affybatch_cache[[dataset_tr_name]]
            }
            else {
                cat("Creating AffyBatch:", dataset_tr_name, "\n")
                affybatch <- ReadAffy(
                    celfile.path=paste0("data/raw/", dataset_tr_name), cdfname=cdfname, verbose=TRUE
                )
                affybatch_cache[[dataset_tr_name]] <- affybatch
            }
            dataset_tr_bg_name <- paste0(dataset_tr_name, suffixes, collapse="_")
            if (exists(dataset_tr_bg_name, where=affybatch_cache)) {
                cat("Loading cached AffyBatch:", dataset_tr_bg_name, "\n")
                affybatch <- affybatch_cache[[dataset_tr_bg_name]]
            }
            else {
                cat("Creating AffyBatch:", dataset_tr_bg_name, "\n")
                if (norm_type == "gcrma") {
                    affybatch <- bg.adjust.gcrma(
                        affybatch, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE
                    )
                }
                else if (norm_type == "rma") {
                    cat("Performing background correction\n")
                    affybatch <- bg.correct.rma(affybatch)
                }
                affybatch_cache[[dataset_tr_bg_name]] <- affybatch
            }
            if (is.null(affybatch_tr)) {
                affybatch_tr <- affybatch
            }
            else {
                affybatch_tr <- merge.AffyBatch(affybatch_tr, affybatch, notes="")
            }
        }
        dataset_tr_norm_name <- paste0(c(dataset_tr_name_combos[,col], suffixes), collapse="_")
        eset_tr_norm_name <- paste0("eset", dataset_tr_norm_name, "tr", collapse="_")
        cat("Creating:", eset_tr_norm_name, "\n")
        if (length(dataset_tr_name_combos[,col]) == 1) {
            if (exists(dataset_tr_norm_name, where=norm_obj_cache)) {
                cat("Loading cached norm object:", dataset_tr_norm_name, "\n")
                norm_obj <- norm_obj_cache[[dataset_tr_norm_name]]
            }
            else {
                norm_obj <- rmatrain(affybatch_tr)
                norm_obj_cache[[dataset_tr_norm_name]] <- norm_obj
            }
        }
        else {
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
            if (exists(dataset_te_name, where=affybatch_cache)) {
                cat("Loading cached AffyBatch:", dataset_te_name, "\n")
                affybatch_te <- affybatch_cache[[dataset_te_name]]
            }
            else {
                cat("Creating AffyBatch:", dataset_te_name, "\n")
                affybatch_te <- ReadAffy(celfile.path=cel_te_dir, cdfname=cdfname, verbose=TRUE)
                affybatch_cache[[dataset_te_name]] <- affybatch_te
            }
            dataset_te_bg_name <- paste0(dataset_te_name, suffixes, collapse="_")
            if (exists(dataset_te_bg_name, where=affybatch_cache)) {
                cat("Loading cached AffyBatch:", dataset_te_bg_name, "\n")
                affybatch_te <- affybatch_cache[[dataset_te_bg_name]]
            }
            else {
                cat("Creating AffyBatch:", dataset_te_bg_name, "\n")
                if (norm_type == "gcrma") {
                    affybatch_te <- bg.adjust.gcrma(
                        affybatch_te, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE
                    )
                }
                else if (norm_type == "rma") {
                    affybatch_te <- bg.correct.rma(affybatch_te)
                }
                affybatch_cache[[dataset_te_bg_name]] <- affybatch_te
            }
            eset_te_name <- paste0(c("eset", dataset_te_name, ref_suffixes), collapse="_")
            eset_te_norm_name <- paste0(eset_tr_norm_name, "_", dataset_te_name, "_te")
            cat("Creating:", eset_te_norm_name, "\n")
            xnorm_te <- rmaaddon(norm_obj, affybatch_te)
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
