#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("preprocessCore"))
suppressPackageStartupMessages(library("affy"))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2.db")))
suppressPackageStartupMessages(suppressWarnings(library("hgu133plus2hsentrezg.db")))
suppressPackageStartupMessages(library("annotate"))
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--norm-meth", type="character", nargs="+", help="preprocessing/normalization method")
parser$add_argument("--id-type", type="character", nargs="+", help="dataset id type")
args <- parser$parse_args()
num_tr_combo <- as.integer(args$num_tr_combo)
if (!is.null(args$norm_meth)) {
    arg_norm_methods <- norm_methods[norm_methods %in% args$norm_meth]
}
if (!is.null(args$id_type)) {
    arg_id_types <- id_types[id_types %in% args$id_type]
}
if ("gene" %in% id_types) {
    cdfname <- "hgu133plus2hsentrezg"
} else {
    cdfname <- "hgu133plus2"
}
if (id_type == "gene" & "mas5" %in% arg_norm_methods) {
    eset_gse67639_name <- "eset_gse67639_mas5_gene"
    cat("Loading:", eset_gse67639_name, "\n")
    load(paste0("data/", eset_gse67639_name, ".Rda"))
}
for (dataset_name in dataset_names) {
    if (!dir.exists(paste0("data/raw/", dataset_name))) next
    for (norm_meth in arg_norm_methods) {
        # load a reference eset
        for (ref_norm_meth in norm_methods) {
            for (ref_id_type in id_types) {
                ref_suffixes <- c(ref_norm_meth)
                if (!(ref_id_type %in% c("none", "gene"))) ref_suffixes <- c(ref_suffixes, ref_id_type)
                eset_ref_name <- paste0(c("eset", dataset_name, ref_suffixes), collapse="_")
                eset_ref_file <- paste0("data/", eset_ref_name, ".Rda")
                if (file.exists(eset_ref_file) & !exists(eset_ref_name)) {
                    cat("Loading:", eset_ref_name, "\n")
                    load(eset_ref_file)
                    break
                }
            }
        }
        for (id_type in arg_id_types) {
            suffixes <- c(norm_meth)
            if (id_type != "none") suffixes <- c(suffixes, id_type)
            eset_norm_name <- paste0(c("eset", dataset_name, suffixes), collapse="_")
            cat("Creating:", eset_norm_name, "\n")
            affybatch <- ReadAffy(
                celfile.path=paste0("data/raw/", dataset_name), cdfname=cdfname, verbose=TRUE
            )
            if (norm_meth == "rma") {
                exprs <- exprs(rma(affybatch))
            }
            else if (norm_meth == "mas5") {
                exprs <- normalize.quantiles(log2(exprs(mas5(affybatch))))
            }
            colnames(exprs) <- sub("\\.CEL$", "", colnames(exprs))
            if (id_type == "gene") {
                eset_norm <- ExpressionSet(
                    assayData=exprs,
                    phenoData=phenoData(get(eset_ref_name)),
                    featureData=AnnotatedDataFrame(data.frame(
                        Symbol=getSYMBOL(rownames(exprs), paste0(cdfname, ".db"))
                    )),
                    annotation=cdfname
                )
                if (exists(eset_gse67639_name)) {
                    if (!exists(common_feature_names) {
                        common_feature_names <- intersect(featureNames(get(eset_gse67639_name)), featureNames(eset_norm))
                    }
                    eset_norm <- eset_norm[common_feature_names,]
                }
            }
            else {
                eset_norm <- get(eset_ref_name)
                exprs(eset_norm) <- exprs
            }
            assign(eset_norm_name, eset_norm)
            save(list=eset_norm_name, file=paste0("data/", eset_norm_name, ".Rda"))
            remove(list=c(eset_norm_name))
        }
    }
}
if (exists(eset_gse67639_name) & exists(common_feature_names) {
    eset_gse67639_gene_filtered <- get(eset_gse67639_name)
    eset_gse67639_gene_filtered <- eset_gse67639_gene_filtered[common_feature_names,]
    assign(eset_gse67639_name, eset_gse67639_gene_filtered)
    save(list=eset_gse67639_name, file=paste0("data/", eset_gse67639_name, ".Rda"))
}
