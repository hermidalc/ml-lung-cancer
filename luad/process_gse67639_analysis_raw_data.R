#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("gcrma"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))

cdfname <- "hgu133plus2hsentrezg"
affinities <- compute.affinities(cdfname, verbose=TRUE)
for (dataset_te_name in c("gse30219", "gse37745", "gse50081")) {
    cel_path_te <- paste0("data/raw/", dataset_te_name)
    cel_files_te <- list.files(path=cel_path_te, full.names=TRUE, pattern="\\.CEL$")
    print(paste("Creating AffyBatch:", dataset_te_name))
    affybatch_te <- ReadAffy(filenames=cel_files_te, cdfname=cdfname, verbose=TRUE)
    eset_te_name <- paste0("eset_", dataset_te_name)
    print(paste("Loading:", eset_te_name))
    load(paste0("data/", eset_te_name, ".Rda"))
    for (norm_type in c("gcrma", "rma")) {
        eset_te_norm_name <- paste0(eset_te_name, "_gene")
        print(paste("Creating:", eset_te_norm_name))
        if (norm_type == "gcrma") {
            eset_te_norm <- gcrma(affybatch_te, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE)
        }
        else if (norm_type == "rma") {
            eset_te_norm <- rma(affybatch_te, verbose=TRUE)
        }
        colnames(eset_te_norm) <- sub("\\.CEL$", "", colnames(eset_te_norm))
        phenoData(eset_te_norm) <- phenoData(get(eset_te_name))
        featureData(eset_te_norm) <- AnnotatedDataFrame(data.frame(
            Symbol=getSYMBOL(featureNames(eset_te_norm), paste0(cdfname, ".db"))
        ))
        annotation(eset_te_norm) <- cdfname
        assign(eset_te_norm_name, eset_te_norm)
        save(list=eset_te_norm_name, file=paste0("data/", eset_te_norm_name, ".Rda"))
        remove(list=c(eset_te_norm_name))
    }
}
