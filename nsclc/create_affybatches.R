#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--num-tr-combo", type="integer", help="num datasets to combine")
parser$add_argument("--norm-meth", type="character", nargs="+", help="preprocessing/normalization method")
parser$add_argument("--id-type", type="character", nargs="+", help="dataset id type")
parser$add_argument("--load-only", action="store_true", default=FALSE, help="show search and dataset load only")
args <- parser$parse_args()

num_tr_combo <- as.integer(args$num_tr_combo)
if (!is.null(args$norm_meth)) {
    norm_methods <- norm_methods[norm_methods %in% args$norm_meth]
}
if (!is.null(args$id_type)) {
    id_types <- id_types[id_types %in% args$id_type]
}
if ("gene" %in% id_types) {
    cdfname <- "hgu133plus2hsentrezg"
} else {
    cdfname <- "hgu133plus2"
}
if ("gcrma" %in% norm_methods) {
    cat("Loading CDF:", cdfname, "\n")
    affinities <- compute.affinities(cdfname, verbose=TRUE)
}
for (norm_meth in norm_methods) {
    if (norm_meth == "gcrma") {
        dataset_tr_name_combos <- combn(dataset_names, num_tr_combo)
        for (col in 1:ncol(dataset_tr_name_combos)) {
            cel_files <- c()
            for (dataset_name in dataset_tr_name_combos[,col]) {
                cel_path <- paste0("data/raw/", dataset_name)
                if (dir.exists(cel_path)) {
                    cel_files <- append(cel_files, list.files(path=cel_path, full.names=TRUE, pattern="\\.CEL$"))
                }
                else {
                    cel_files <- NULL
                    break
                }
            }
            if (is.null(cel_files)) next
            for (id_type in id_types) {
                suffixes <- c(norm_meth)
                if (id_type != "none") suffixes <- c(suffixes, id_type)
                affybatch_name <- paste0(c("affybatch", dataset_tr_name_combos[,col], suffixes), collapse="_")
                cat("Creating AffyBatch:", affybatch_name, "\n")
                if (args$load_only) next
                affybatch <- ReadAffy(filenames=cel_files, cdfname=cdfname, verbose=TRUE)
                affybatch <- bg.adjust.gcrma(
                    affybatch, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE
                )
                assign(affybatch_name, affybatch)
                save(list=affybatch_name, file=paste0("data/", affybatch_name, ".Rda"))
                remove(list=c(affybatch_name))
            }
        }
    }
    else if (norm_meth == "rma") {
        for (dataset_name in dataset_names) {
            if (!dir.exists(paste0("data/raw/", dataset_name))) next
            for (id_type in id_types) {
                suffixes <- c(norm_meth)
                if (id_type != "none") suffixes <- c(suffixes, id_type)
                affybatch_name <- paste0(c("affybatch", dataset_name, suffixes), collapse="_")
                cat("Creating AffyBatch:", affybatch_name, "\n")
                if (args$load_only) next
                affybatch <- ReadAffy(celfile.path=paste0("data/raw/", dataset_name), cdfname=cdfname, verbose=TRUE)
                cat("Performing background correction\n")
                affybatch <- bg.correct.rma(affybatch)
                assign(affybatch_name, affybatch)
                save(list=affybatch_name, file=paste0("data/", affybatch_name, ".Rda"))
                remove(list=c(affybatch_name))
            }
        }
    }
}
