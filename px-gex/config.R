# config
dataset_names <- c(
    "gse8894",
    "gse30219",
    "gse31210",
    "gse37745",
    "gse50081",
    "gse67639"
)
multi_dataset_names <- c(
    "gse67639"
)
norm_methods <- c(
    "gcrma",
    "rma",
    "mas5"
)
id_types <- c(
    "none",
    "gene"
)
filter_types <- c(
    "none",
    "filtered"
)
merge_types <- c(
    "none",
    "merged"
)
bc_methods <- c(
    "none",
    "ctr",
    "std",
    "rta",
    "rtg",
    "qnorm",
    "cbt",
    # "fab",
    # "sva",
    "stica0",
    "stica025",
    "stica05",
    "stica1",
    "svd"
)
common_pheno_colnames <- c("Age", "Gender", "Histology", "Class", "Batch")
matfact_k <- 20
