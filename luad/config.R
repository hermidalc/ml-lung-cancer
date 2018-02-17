#!/usr/bin/env/R

dataset_names <- c(
    "gse31210",
    "gse8894",
    "gse30219",
    "gse37745"
)
eset_merged_tr_names <- c(
    "eset_gse31210_gse8894_gse30219",
    "eset_gse31210_gse8894_gse37745",
    "eset_gse31210_gse30219_gse37745",
    "eset_gse8894_gse30219_gse37745"
)
eset_merged_te_names <- c(
    "eset_gse37745",
    "eset_gse30219",
    "eset_gse8894",
    "eset_gse31210"
)
stica_alphas <- c( 0, 0.25, 0.5, 0.75, 1 )
