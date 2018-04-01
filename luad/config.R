# config
dataset_names <- c(
    "gse8894",
    "gse30219",
    "gse31210",
    "gse37745",
    "gse50081"
)
common_pheno_colnames <- c("Age", "Gender", "Histology", "Relapse", "Batch")
stica_alphas <- c( 0, 0.25, 0.5, 0.75, 1 )
matfact_k <- 20
