#!/usr/bin/env Rscript

options(warn=1)
suppressPackageStartupMessages(library("argparse"))
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
source("lib/R/svapred.R")
source("lib/R/stICA.R")
source("config.R")

parser <- ArgumentParser()
parser$add_argument("--dataset-tr", type="character", nargs="+", help="dataset tr")
parser$add_argument("--dataset-te", type="character", nargs="+", help="dataset te")
parser$add_argument("--num-combo-tr", type="integer", help="num tr datasets to combine")
parser$add_argument("--data-type", type="character", nargs="+", help="data type")
parser$add_argument("--norm-meth", type="character", nargs="+", help="normalization method")
parser$add_argument("--feat-type", type="character", nargs="+", help="feature type")
parser$add_argument("--prep-meth", type="character", nargs="+", help="preprocess method")
parser$add_argument("--bc-meth", type="character", nargs="+", help="batch correction method")
parser$add_argument("--load-only", action="store_true", default=FALSE, help="show search and eset load only")
parser$add_argument("--save-obj", action="store_true", default=FALSE, help="save add-on param obj")
args <- parser$parse_args()
if (!is.null(args$dataset_tr) && !is.null(args$num_combo_tr)) {
    dataset_tr_name_combos <- combn(intersect(dataset_names, args$dataset_tr), as.integer(args$num_combo_tr))
} else if (!is.null(args$dataset_tr)) {
    dataset_tr_name_combos <- combn(intersect(dataset_names, args$dataset_tr), length(args$dataset_tr))
} else {
    if (is.null(args$num_combo_tr)) stop("--num-combo-tr option required")
    dataset_tr_name_combos <- combn(dataset_names, as.integer(args$num_combo_tr))
}
if (!is.null(args$dataset_te)) {
    dataset_te_names <- intersect(dataset_names, args$dataset_te)
} else {
    dataset_te_names <- dataset_names
}
if (!is.null(args$data_type)) {
    data_types <- intersect(data_types, args$data_type)
}
if (!is.null(args$norm_meth)) {
    norm_methods <- intersect(norm_methods, args$norm_meth)
}
if (!is.null(args$feat_type)) {
    feat_types <- intersect(feat_types, args$feat_type)
}
if (!is.null(args$prep_meth)) {
    prep_methods <- intersect(prep_methods, args$prep_meth)
}
if (!is.null(args$bc_meth)) {
    bc_methods <- intersect(bc_methods, args$bc_meth)
}
for (col in 1:ncol(dataset_tr_name_combos)) {
    for (data_type in data_types) {
        for (norm_meth in norm_methods) {
            for (feat_type in feat_types) {
                suffixes <- c(data_type)
                for (suffix in c(norm_meth, feat_type)) {
                    if (!(suffix %in% c("none", "None"))) suffixes <- c(suffixes, suffix)
                }
                for (prep_meth in prep_methods) {
                    suffixes_tr <- suffixes
                    suffixes_te <- suffixes
                    if (!(prep_meth %in% c("none", "None"))) {
                        suffixes_tr <- c(suffixes_tr, prep_meth)
                        if (prep_meth != "mrg") suffixes_te <- suffixes_tr
                    }
                    if (length(dataset_tr_name_combos[,col]) > 1) {
                        eset_tr_name <- paste0(
                            c("eset", dataset_tr_name_combos[,col], suffixes_tr, "tr"), collapse="_"
                        )
                    }
                    else {
                        eset_tr_name <- paste0(
                            c("eset", dataset_tr_name_combos[,col], suffixes_tr), collapse="_"
                        )
                    }
                    eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
                    if (!exists(eset_tr_name)) {
                        if (file.exists(eset_tr_file)) {
                            cat(" Loading:", eset_tr_name, "\n")
                            load(eset_tr_file)
                        }
                        else {
                            next
                        }
                    }
                    for (dataset_te_name in setdiff(dataset_te_names, dataset_tr_name_combos[,col])) {
                        if (length(dataset_tr_name_combos[,col]) == 1 || prep_meth == "mrg") {
                            eset_te_name <- paste0(c("eset", dataset_te_name, suffixes_te), collapse="_")
                        }
                        else {
                            eset_te_name <- paste0(c(eset_tr_name, dataset_te_name, "te"), collapse="_")
                        }
                        eset_te_file <- paste0("data/", eset_te_name, ".Rda")
                        if (!exists(eset_te_name) & file.exists(eset_te_file)) {
                            cat(" Loading:", eset_te_name, "\n")
                            load(eset_te_file)
                        }
                    }
                    if (args$load_only) next
                    for (bc_meth in bc_methods) {
                        if (length(dataset_tr_name_combos[,col]) > 1 || !(bc_meth %in% c("none", "None"))) {
                            eset_tr_bc_name <- paste0(
                                c("eset", dataset_tr_name_combos[,col], suffixes_tr, bc_meth, "tr"),
                                collapse="_"
                            )
                        }
                        else {
                            eset_tr_bc_name <- paste0(c(eset_tr_name, bc_meth), collapse="_")
                        }
                        if (grepl("^(stica\\d+|svd)$", bc_meth)) {
                            Xtr <- exprs(get(eset_tr_name))
                            ptr <- pData(get(eset_tr_name))
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
                                if (length(dataset_tr_name_combos[,col]) == 1 || prep_meth == "mrg") {
                                    eset_te_name <- paste0(c("eset", dataset_te_name, suffixes_te), collapse="_")
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
                                if (length(dataset_tr_name_combos[,col]) == 1 || prep_meth == "mrg") {
                                    eset_te_name <- paste0(c("eset", dataset_te_name, suffixes_te), collapse="_")
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
