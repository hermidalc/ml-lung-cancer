#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("affy"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))
source("lib/R/rmapred.R")
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--num-tr-combo", type="integer", help="num datasets to combine")
parser$add_argument("--norm-meth", type="character", nargs="+", help="preprocessing/normalization method")
parser$add_argument("--id-type", type="character", nargs="+", help="dataset id type")
parser$add_argument("--num-cores", type="integer", default=detectCores(), help="num parallel cores")
parser$add_argument("--load-only", action="store_true", default=FALSE, help="show search and eset load only")
parser$add_argument("--save-obj", action="store_true", default=FALSE, help="save add-on param obj")
args <- parser$parse_args()

num_tr_combo <- as.integer(args$num_tr_combo)
if (!is.null(args$norm_meth)) {
    arg_norm_methods <- norm_methods[norm_methods %in% args$norm_meth]
} else {
    arg_norm_methods <- norm_methods
}
if (!is.null(args$id_type)) {
    id_types <- id_types[id_types %in% args$id_type]
} else {
    id_types <- id_types
}
if ("gene" %in% id_types) {
    cdfname <- "hgu133plus2hsentrezg"
} else {
    cdfname <- "hgu133plus2"
}
dataset_tr_name_combos <- combn(dataset_names, num_tr_combo)
for (col in 1:ncol(dataset_tr_name_combos)) {
    skip_processing <- FALSE
    for (dataset_tr_name in dataset_tr_name_combos[,col]) {
        if (!dir.exists(paste0("data/raw/", dataset_tr_name))) {
            skip_processing <- TRUE
            break
        }
    }
    if (skip_processing) next
    for (norm_meth in arg_norm_methods) {
        for (id_type in id_types) {
            # load a reference eset set
            for (ref_norm_meth in norm_methods) {
                ref_suffixes <- c(ref_norm_meth)
                if (!(id_type %in% c("none", "gene"))) ref_suffixes <- c(ref_suffixes, id_type)
                if (num_tr_combo > 1) {
                    eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], ref_suffixes, "merged", "tr"), collapse="_")
                }
                else {
                    eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], ref_suffixes), collapse="_")
                }
                eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
                if (!exists(eset_tr_name)) {
                    if (file.exists(eset_tr_file)) {
                        cat("Loading:", eset_tr_name, "\n")
                        load(eset_tr_file)
                        for (dataset_te_name in setdiff(dataset_names, dataset_tr_name_combos[,col])) {
                            if (!dir.exists(paste0("data/raw/", dataset_te_name))) next
                            eset_te_name <- paste0(c("eset", dataset_te_name, ref_suffixes), collapse="_")
                            if (!exists(eset_te_name)) {
                                cat("Loading:", eset_te_name, "\n")
                                load(paste0("data/", eset_te_name, ".Rda"))
                            }
                        }
                        break
                    }
                }
                else {
                    break
                }
            }
            # load affybatches
            suffixes <- c(norm_meth)
            if (id_type != "none") suffixes <- c(suffixes, id_type)
            if (norm_meth == "gcrma") {
                affybatch_tr_name <- paste0(c("affybatch", dataset_tr_name_combos[,col], suffixes), collapse="_")
                if (!exists(affybatch_tr_name)) {
                    cat("Loading:", affybatch_tr_name, "\n")
                    load(paste0("data/", affybatch_tr_name, ".Rda"))
                }
                affybatch_tr <- get(affybatch_tr_name)
            }
            else if (norm_meth == "rma") {
                if (length(dataset_tr_name_combos[,col]) > 1) {
                    cat("Loading/Merging: ")
                }
                else {
                    cat("Loading: ")
                }
                affybatch_tr <- NULL
                for (dataset_tr_name in dataset_tr_name_combos[,col]) {
                    affybatch_tr_name <- paste0(c("affybatch", dataset_tr_name, suffixes), collapse="_")
                    cat(affybatch_tr_name, "")
                    if (!exists(affybatch_tr_name)) {
                        load(paste0("data/", affybatch_tr_name, ".Rda"))
                    }
                    if (is.null(affybatch_tr)) {
                        affybatch_tr <- get(affybatch_tr_name)
                    }
                    else {
                        affybatch_tr <- merge.AffyBatch(affybatch_tr, get(affybatch_tr_name), notes="")
                    }
                }
                cat("\n")
            }
            if (args$load_only) {
                remove(list=c(affybatch_tr_name, eset_tr_name))
                next
            }
            # process data
            dataset_tr_norm_name <- paste0(c(dataset_tr_name_combos[,col], suffixes), collapse="_")
            eset_tr_norm_name <- paste("eset", dataset_tr_norm_name, "tr", sep="_")
            cat("Creating:", eset_tr_norm_name, "\n")
            norm_obj <- rmatrain(affybatch_tr)
            rownames(norm_obj$xnorm) <- sub("\\.CEL$", "", rownames(norm_obj$xnorm))
            if (id_type == "gene") {
                eset_tr_norm <- ExpressionSet(
                    assayData=t(norm_obj$xnorm),
                    phenoData=phenoData(get(eset_tr_name)),
                    featureData=AnnotatedDataFrame(data.frame(
                        Symbol=getSYMBOL(colnames(norm_obj$xnorm), paste0(cdfname, ".db"))
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
            if (args$save_obj) {
                save(list=eset_tr_norm_obj_name, file=paste0("data/", eset_tr_norm_obj_name, ".Rda"))
            }
            for (dataset_te_name in setdiff(dataset_names, dataset_tr_name_combos[,col])) {
                cel_te_dir <- paste0("data/raw/", dataset_te_name)
                if (!dir.exists(cel_te_dir)) next
                affybatch_te_name <- paste0(c("affybatch", dataset_te_name, suffixes), collapse="_")
                if (!exists(affybatch_te_name)) {
                    cat("Loading:", affybatch_te_name, "\n")
                    load(paste0("data/", affybatch_te_name, ".Rda"))
                }
                eset_te_name <- paste0(c("eset", dataset_te_name, ref_suffixes), collapse="_")
                eset_te_norm_name <- paste(eset_tr_norm_name, dataset_te_name, "te", sep="_")
                cat("Creating:", eset_te_norm_name, "\n")
                xnorm_te <- rmaaddon(norm_obj, get(affybatch_te_name), num.cores=args$num_cores)
                rownames(xnorm_te) <- sub("\\.CEL$", "", rownames(xnorm_te))
                if (id_type == "gene") {
                    eset_te_norm <- ExpressionSet(
                        assayData=t(xnorm_te),
                        phenoData=phenoData(get(eset_te_name)),
                        featureData=AnnotatedDataFrame(data.frame(
                            Symbol=getSYMBOL(colnames(norm_obj$xnorm), paste0(cdfname, ".db"))
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
            # unload multi-dataset gcrma affybatch tr (to save memory)
            if (norm_meth == "gcrma" & num_tr_combo > 1) remove(list=c(affybatch_tr_name))
        }
    }
}
