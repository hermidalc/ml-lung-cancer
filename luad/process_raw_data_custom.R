#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("preprocessCore"))
suppressPackageStartupMessages(library("affy"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))
source("config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
id_type <- cmd_args[1]
if (id_type == "gene") {
    cdfname <- "hgu133plus2hsentrezg"
} else {
    cdfname <- "hgu133plus2"
}
if (id_type == "gene" && "mas5" %in% cmd_args[2:length(cmd_args)]) {
    eset_gse67639_name <- "eset_gse67639_mas5_gene"
    cat("Loading:", eset_gse67639_name, "\n")
    load(paste0("data/", eset_gse67639_name, ".Rda"))
}
for (dataset_name in dataset_names) {
    if (!dir.exists(paste0("data/raw/", dataset_name))) next
    for (norm_type in cmd_args[2:length(cmd_args)]) {
        # load a reference eset
        for (ref_norm_type in norm_types) {
            ref_suffixes <- c(ref_norm_type)
            if (!is.na(id_type) & id_type != "gene") ref_suffixes <- c(ref_suffixes, id_type)
            eset_ref_name <- paste0(c("eset", dataset_name, ref_suffixes), collapse="_")
            eset_ref_file <- paste0("data/", eset_ref_name, ".Rda")
            if (file.exists(eset_ref_file) & !exists(eset_ref_name)) {
                cat("Loading:", eset_ref_name, "\n")
                load(eset_ref_file)
                break
            }
        }
        suffixes <- c(norm_type)
        if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
        eset_norm_name <- paste0(c("eset", dataset_name, suffixes), collapse="_")
        cat("Creating:", eset_norm_name, "\n")
        affybatch <- ReadAffy(
            celfile.path=paste0("data/raw/", dataset_name), cdfname=cdfname, verbose=TRUE
        )
        if (norm_type == "rma") {
            exprs <- exprs(rma(affybatch))
        }
        else if (norm_type == "mas5") {
            exprs <- normalize.quantiles(log2(exprs(mas5(affybatch))))
        }
        colnames(exprs) <- sub("\\.CEL$", "", colnames(exprs))
        if (id_type == "gene") {
            if (exists(eset_gse67639_name)) {
                
            }
            eset_norm <- ExpressionSet(
                assayData=exprs,
                phenoData=phenoData(get(eset_ref_name)),
                featureData=AnnotatedDataFrame(data.frame(
                    Symbol=getSYMBOL(rownames(exprs), paste0(cdfname, ".db"))
                )),
                annotation=cdfname
            )
        }
        else {
            eset_norm <- get(eset_ref_name)
            exprs(eset_norm) <- exprs
        }
        assign(eset_norm_name, eset_norm)
        save(list=eset_norm_name, file=paste0("data/", eset_norm_name, ".Rda"))
    }
}
