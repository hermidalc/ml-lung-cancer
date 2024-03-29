#!/usr/bin/env Rscript

options(warn=1)
suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
source("lib/R/svapred.R")
source("lib/R/stICA.R")
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--datasets-tr", type="character", nargs="+", help="datasets tr")
parser$add_argument("--datasets-te", type="character", nargs="+", help="datasets te")
parser$add_argument("--num-tr-combo", type="integer", help="num datasets to combine")
parser$add_argument("--norm-meth", type="character", nargs="+", help="preprocessing/normalization method")
parser$add_argument("--id-type", type="character", nargs="+", help="dataset id type")
parser$add_argument("--filter-type", type="character", nargs="+", help="dataset filter type")
parser$add_argument("--merge-type", type="character", nargs="+", help="dataset merge type")
parser$add_argument("--bc-meth", type="character", nargs="+", help="dataset batch correct method")
parser$add_argument("--load-only", action="store_true", default=FALSE, help="show search and eset load only")
parser$add_argument("--save-obj", action="store_true", default=FALSE, help="save add-on param obj")
args <- parser$parse_args()

if (!is.null(args$datasets_tr) && !is.null(args$num_tr_combo)) {
    dataset_tr_name_combos <- combn(intersect(dataset_names, args$datasets_tr), as.integer(args$num_tr_combo))
} else if (!is.null(args$datasets_tr)) {
    dataset_tr_name_combos <- combn(intersect(dataset_names, args$datasets_tr), length(args$datasets_tr))
} else {
    dataset_tr_name_combos <- combn(dataset_names, as.integer(args$num_tr_combo))
}
if (!is.null(args$datasets_te)) {
    dataset_te_names <- intersect(dataset_names, args$datasets_te)
} else {
    dataset_te_names <- dataset_names
}
if (!is.null(args$norm_meth)) {
    norm_methods <- norm_methods[norm_methods %in% args$norm_meth]
}
if (!is.null(args$id_type)) {
    id_types <- id_types[id_types %in% args$id_type]
}
if (!is.null(args$filter_type)) {
    filter_types <- filter_types[filter_types %in% args$filter_type]
}
if (!is.null(args$merge_type)) {
    merge_types <- merge_types[merge_types %in% args$merge_type]
}
if (!is.null(args$bc_meth)) {
    bc_methods <- bc_methods[bc_methods %in% args$bc_meth]
}
for (col in 1:ncol(dataset_tr_name_combos)) {
    for (norm_meth in norm_methods) {
        for (id_type in id_types) {
            suffixes <- c(norm_meth)
            if (id_type != "none") suffixes <- c(suffixes, id_type)
            for (filter_type in filter_types) {
                if (filter_type != "none") suffixes <- c(suffixes, filter_type)
                for (merge_type in merge_types) {
                    if (merge_type == "none") {
                        eset_tr_name <- paste0(
                            c("eset", dataset_tr_name_combos[,col], suffixes, "tr"), collapse="_"
                        )
                    }
                    else {
                        eset_tr_name <- paste0(
                            c("eset", dataset_tr_name_combos[,col], suffixes, merge_type, "tr"), collapse="_"
                        )
                    }
                    eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
                    if (!exists(eset_tr_name)) {
                        if (file.exists(eset_tr_file)) {
                            cat("Loading:", eset_tr_name, "\n")
                            load(eset_tr_file)
                        }
                        else {
                            next
                        }
                    }
                    for (dataset_te_name in setdiff(dataset_te_names, dataset_tr_name_combos[,col])) {
                        if (merge_type != "none" || (
                            length(dataset_tr_name_combos[,col]) == 1 &&
                            dataset_tr_name_combos[,col] %in% multi_dataset_names
                        )) {
                            eset_te_name <- paste0(c("eset", dataset_te_name, suffixes), collapse="_")
                        }
                        else {
                            eset_te_name <- paste0(c(eset_tr_name, dataset_te_name, "te"), collapse="_")
                        }
                        eset_te_file <- paste0("data/", eset_te_name, ".Rda")
                        if (!exists(eset_te_name) & file.exists(eset_te_file)) {
                            cat("Loading:", eset_te_name, "\n")
                            load(eset_te_file)
                        }
                    }
                    if (args$load_only) next
                    for (bc_meth in bc_methods) {
                        if (grepl("^(stica\\d+|svd)$", bc_meth)) {
                            Xtr <- exprs(get(eset_tr_name))
                            ptr <- pData(get(eset_tr_name))
                            eset_tr_bc_name <- paste0(sub("_tr$", "", eset_tr_name), "_", bc_meth, "_tr")
                            cat("Creating:", eset_tr_bc_name, "\n")
                            if (substr(bc_meth, 1, 5) == "stica") {
                                bc_obj <- normFact(
                                    "stICA", Xtr, ptr$Batch, "categorical",
                                    ref2=ptr$Class, refType2="categorical", k=matfact_k,
                                    alpha=as.numeric(sub("^0", "0.", regmatches(bc_meth, regexpr("\\d+$", bc_meth))))
                                )
                            }
                            else if (bc_meth == "svd") {
                                bc_obj <- normFact(
                                    "SVD", Xtr, ptr$Batch, "categorical",
                                    ref2=ptr$Class, refType2="categorical", k=matfact_k
                                )
                            }
                            eset_tr_bc <- get(eset_tr_name)
                            exprs(eset_tr_bc) <- bc_obj$Xn
                            assign(eset_tr_bc_name, eset_tr_bc)
                            save(list=eset_tr_bc_name, file=paste0("data/", eset_tr_bc_name, ".Rda"))
                            eset_tr_bc_obj_name <- paste0(eset_tr_bc_name, "_obj")
                            assign(eset_tr_bc_obj_name, bc_obj)
                            if (args$save_obj) {
                                save(list=eset_tr_bc_obj_name, file=paste0("data/", eset_tr_bc_obj_name, ".Rda"))
                            }
                            for (dataset_te_name in setdiff(dataset_te_names, dataset_tr_name_combos[,col])) {
                                if (merge_type != "none" || (
                                    length(dataset_tr_name_combos[,col]) == 1 &&
                                    dataset_tr_name_combos[,col] %in% multi_dataset_names
                                )) {
                                    eset_te_name <- paste0(c("eset", dataset_te_name, suffixes), collapse="_")
                                }
                                else {
                                    eset_te_name <- paste0(c(eset_tr_name, dataset_te_name, "te"), collapse="_")
                                }
                                if (!exists(eset_te_name)) next
                                Xte <- exprs(get(eset_te_name))
                                eset_te_bc_name <- paste0(c(eset_tr_bc_name, dataset_te_name, "te"), collapse="_")
                                cat("Creating:", eset_te_bc_name, "\n")
                                eset_te_bc <- get(eset_te_name)
                                # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
                                # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
                                # Xte_n = dot(U,Vte.T)
                                exprs(eset_te_bc) <- bc_obj$U %*% t((t(Xte) %*% bc_obj$U) %*% solve(t(bc_obj$U) %*% bc_obj$U))
                                assign(eset_te_bc_name, eset_te_bc)
                                save(list=eset_te_bc_name, file=paste0("data/", eset_te_bc_name, ".Rda"))
                                remove(list=c(eset_te_bc_name))
                            }
                            remove(list=c(eset_tr_bc_obj_name, eset_tr_bc_name))
                        }
                        else if (bc_meth %in% c("cbt", "ctr", "fab", "qnorm", "rta", "rtg", "std", "sva")) {
                            Xtr <- t(exprs(get(eset_tr_name)))
                            ptr <- pData(get(eset_tr_name))
                            ytr <- as.factor(ptr$Class + 1)
                            btr <- ptr$Batch
                            butr <- sort(unique(btr))
                            for (j in 1:length(butr)) {
                                if (j != butr[j]) {
                                    btr <- replace(btr, btr == butr[j], j)
                                }
                            }
                            btr <- as.factor(btr)
                            eset_tr_bc_name <- paste0(sub("_tr$", "", eset_tr_name), "_", bc_meth, "_tr")
                            cat("Creating:", eset_tr_bc_name, "\n")
                            eset_tr_bc <- get(eset_tr_name)
                            if (bc_meth == "cbt") {
                                bc_obj <- combatba(Xtr, btr)
                                exprs(eset_tr_bc) <- t(bc_obj$xadj)
                            }
                            else if (bc_meth == "ctr") {
                                bc_obj <- meancenter(Xtr, btr)
                                exprs(eset_tr_bc) <- t(bc_obj$xadj)
                            }
                            else if (bc_meth == "fab") {
                                bc_obj <- fabatch(Xtr, ytr, btr)
                                exprs(eset_tr_bc) <- t(bc_obj$xadj)
                            }
                            else if (bc_meth == "qnorm") {
                                bc_obj <- qunormtrain(Xtr)
                                exprs(eset_tr_bc) <- t(bc_obj$xnorm)
                            }
                            else if (bc_meth == "rta") {
                                bc_obj <- ratioa(Xtr, btr)
                                exprs(eset_tr_bc) <- t(bc_obj$xadj)
                            }
                            else if (bc_meth == "rtg") {
                                bc_obj <- ratiog(Xtr, btr)
                                exprs(eset_tr_bc) <- t(bc_obj$xadj)
                            }
                            else if (bc_meth == "std") {
                                bc_obj <- standardize(Xtr, btr)
                                exprs(eset_tr_bc) <- t(bc_obj$xadj)
                            }
                            else if (bc_meth == "sva") {
                                mod <- model.matrix(~as.factor(Class), data=ptr)
                                mod0 <- model.matrix(~1, data=ptr)
                                # ctrls <- as.numeric(grepl("^AFFX", rownames(t(Xtr))))
                                bc_obj <- svaba(Xtr, btr, mod, mod0, algorithm="fast")
                                exprs(eset_tr_bc) <- t(bc_obj$xadj)
                            }
                            assign(eset_tr_bc_name, eset_tr_bc)
                            save(list=eset_tr_bc_name, file=paste0("data/", eset_tr_bc_name, ".Rda"))
                            eset_tr_bc_obj_name <- paste0(eset_tr_bc_name, "_obj")
                            assign(eset_tr_bc_obj_name, bc_obj)
                            if (args$save_obj) {
                                save(list=eset_tr_bc_obj_name, file=paste0("data/", eset_tr_bc_obj_name, ".Rda"))
                            }
                            for (dataset_te_name in setdiff(dataset_te_names, dataset_tr_name_combos[,col])) {
                                if (merge_type != "none" || (
                                    length(dataset_tr_name_combos[,col]) == 1 &&
                                    dataset_tr_name_combos[,col] %in% multi_dataset_names
                                )) {
                                    eset_te_name <- paste0(c("eset", dataset_te_name, suffixes), collapse="_")
                                }
                                else {
                                    eset_te_name <- paste0(c(eset_tr_name, dataset_te_name, "te"), collapse="_")
                                }
                                if (!exists(eset_te_name)) next
                                Xte <- t(exprs(get(eset_te_name)))
                                pte <- pData(get(eset_te_name))
                                bte <- pte$Batch
                                bute <- sort(unique(bte))
                                for (j in 1:length(bute)) {
                                    if (j != bute[j]) {
                                        bte <- replace(bte, bte == bute[j], j)
                                    }
                                }
                                bte <- as.factor(bte)
                                eset_te_bc_name <- paste0(c(eset_tr_bc_name, dataset_te_name, "te"), collapse="_")
                                cat("Creating:", eset_te_bc_name, "\n")
                                eset_te_bc <- get(eset_te_name)
                                if (bc_meth == "cbt") {
                                    exprs(eset_te_bc) <- t(combatbaaddon(bc_obj, Xte, bte))
                                }
                                else if (bc_meth == "ctr") {
                                    exprs(eset_te_bc) <- t(meancenteraddon(bc_obj, Xte, bte))
                                }
                                else if (bc_meth == "fab") {
                                    exprs(eset_te_bc) <- t(fabatchaddon(bc_obj, Xte, bte))
                                }
                                else if (bc_meth == "qnorm") {
                                    exprs(eset_te_bc) <- t(qunormaddon(bc_obj, Xte))
                                }
                                else if (bc_meth == "rta") {
                                    exprs(eset_te_bc) <- t(ratioaaddon(bc_obj, Xte, bte))
                                }
                                else if (bc_meth == "rtg") {
                                    exprs(eset_te_bc) <- t(ratiogaddon(bc_obj, Xte, bte))
                                }
                                else if (bc_meth == "std") {
                                    exprs(eset_te_bc) <- t(standardizeaddon(bc_obj, Xte, bte))
                                }
                                else if (bc_meth == "sva") {
                                    exprs(eset_te_bc) <- t(svabaaddon(bc_obj, Xte))
                                }
                                assign(eset_te_bc_name, eset_te_bc)
                                save(list=eset_te_bc_name, file=paste0("data/", eset_te_bc_name, ".Rda"))
                                remove(list=c(eset_te_bc_name))
                            }
                            remove(list=c(eset_tr_bc_obj_name, eset_tr_bc_name))
                        }
                    }
                }
            }
        }
    }
}
