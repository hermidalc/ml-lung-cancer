# config
dataset_names <- c(
    "gse8894",
    "gse30219",
    "gse31210",
    "gse37745",
    "gse50081"
)
norm_types <- c(
    "gcrma",
    "rma",
    "mas5"
)
bc_types <- c(
    'none',
    'ctr',
    'std',
    'rta',
    'rtg',
    'qnorm',
    'cbt',
    #'fab',
    #'sva',
    'stica0',
    'stica025',
    'stica05',
    'stica1',
    'svd'
)
common_pheno_colnames <- c("Age", "Gender", "Histology", "Class", "Batch")
matfact_k <- 20
