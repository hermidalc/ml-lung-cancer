#!/usr/bin/env/R

eset_tr_strs = c(
    "eset_gex_gse31210_gse8894_gse30219_gse37745",
    "eset_gex_gse31210_gse8894_gse30219_gse50081",
    "eset_gex_gse31210_gse8894_gse37745_gse50081",
    "eset_gex_gse31210_gse30219_gse37745_gse50081",
    "eset_gex_gse8894_gse30219_gse37745_gse50081"
)
eset_te_strs = c(
    "eset_gex_gse50081",
    "eset_gex_gse37745",
    "eset_gex_gse30219",
    "eset_gex_gse8894",
    "eset_gex_gse31210"
)
stica_alphas = c( 0, 0.25, 0.5, 0.75, 1 )
