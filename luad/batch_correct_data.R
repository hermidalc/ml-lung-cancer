#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
source("lib/R/svapred.R")
source("lib/R/stICA.R")
source("config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
num_tr_subset <- as.integer(cmd_args[1])
norm_type <- cmd_args[2]
id_type <- cmd_args[3]
merged_type <- cmd_args[4]
suffixes <- c(norm_type)
if (!is.na(id_type) & id_type != "none") suffixes <- c(suffixes, id_type)
dataset_tr_name_combos <- combn(dataset_names, num_tr_subset)
for (col in 1:ncol(dataset_tr_name_combos)) {
    if (is.na(merged_type) | merged_type == "none") {
        eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes, "tr"), collapse="_")
    }
    else {
        eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes, "merged", "tr"), collapse="_")
    }
    eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
    if (file.exists(eset_tr_file)) {
        cat("Loading:", eset_tr_name, "\n")
        load(eset_tr_file)
    }
    for (dataset_te_name in setdiff(dataset_names, dataset_tr_name_combos[,col])) {
        if (is.na(merged_type) | merged_type == "none") {
            eset_te_name <- paste0(c(eset_tr_name, dataset_te_name, "te"), collapse="_")
        }
        else {
            eset_te_name <- paste0(c("eset", dataset_te_name, suffixes), collapse="_")
        }
        eset_te_file <- paste0("data/", eset_te_name, ".Rda")
        if (!exists(eset_te_name) & file.exists(eset_te_file)) {
            cat("Loading:", eset_te_name, "\n")
            load(eset_te_file)
        }
    }
}
if (length(cmd_args) > 4) bc_types <- cmd_args[5:length(cmd_args)]
for (bc_type in bc_types) {
    for (col in 1:ncol(dataset_tr_name_combos)) {
        if (is.na(merged_type) | merged_type == "none") {
            eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes, "tr"), collapse="_")
        }
        else {
            eset_tr_name <- paste0(c("eset", dataset_tr_name_combos[,col], suffixes, "merged", "tr"), collapse="_")
        }
        if (!exists(eset_tr_name)) next
        if (grepl("^(stica\\d+|svd)$", bc_type)) {
            Xtr <- exprs(get(eset_tr_name))
            ptr <- pData(get(eset_tr_name))
            eset_tr_bc_name <- paste0(sub("_tr$", "", eset_tr_name), "_", bc_type, "_tr")
            cat("Creating:", eset_tr_bc_name, "\n")
            if (substr(bc_type, 1, 5) == "stica") {
                bc_obj <- normFact(
                    "stICA", Xtr, ptr$Batch, "categorical",
                    ref2=ptr$Class, refType2="categorical", k=matfact_k,
                    alpha=as.numeric(sub("^0", "0.", regmatches(bc_type, regexpr("\\d+$", bc_type))))
                )
            }
            else if (bc_type == "svd") {
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
            save(list=eset_tr_bc_obj_name, file=paste0("data/", eset_tr_bc_obj_name, ".Rda"))
            for (dataset_te_name in setdiff(dataset_names, dataset_tr_name_combos[,col])) {
                if (is.na(merged_type) | merged_type == "none") {
                    eset_te_name <- paste0(c(eset_tr_name, dataset_te_name, "te"), collapse="_")
                }
                else {
                    eset_te_name <- paste0(c("eset", dataset_te_name, suffixes), collapse="_")
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
        else if (bc_type %in% c("cbt", "ctr", "fab", "qnorm", "rta", "rtg", "std", "sva")) {
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
            eset_tr_bc_name <- paste0(sub("_tr$", "", eset_tr_name), "_", bc_type, "_tr")
            cat("Creating:", eset_tr_bc_name, "\n")
            eset_tr_bc <- get(eset_tr_name)
            if (bc_type == "cbt") {
                bc_obj <- combatba(Xtr, btr)
                exprs(eset_tr_bc) <- t(bc_obj$xadj)
            }
            else if (bc_type == "ctr") {
                bc_obj <- meancenter(Xtr, btr)
                exprs(eset_tr_bc) <- t(bc_obj$xadj)
            }
            else if (bc_type == "fab") {
                bc_obj <- fabatch(Xtr, ytr, btr)
                exprs(eset_tr_bc) <- t(bc_obj$xadj)
            }
            else if (bc_type == "qnorm") {
                bc_obj <- qunormtrain(Xtr)
                exprs(eset_tr_bc) <- t(bc_obj$xnorm)
            }
            else if (bc_type == "rta") {
                bc_obj <- ratioa(Xtr, btr)
                exprs(eset_tr_bc) <- t(bc_obj$xadj)
            }
            else if (bc_type == "rtg") {
                bc_obj <- ratiog(Xtr, btr)
                exprs(eset_tr_bc) <- t(bc_obj$xadj)
            }
            else if (bc_type == "std") {
                bc_obj <- standardize(Xtr, btr)
                exprs(eset_tr_bc) <- t(bc_obj$xadj)
            }
            else if (bc_type == "sva") {
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
            save(list=eset_tr_bc_obj_name, file=paste0("data/", eset_tr_bc_obj_name, ".Rda"))
            for (dataset_te_name in setdiff(dataset_names, dataset_tr_name_combos[,col])) {
                if (is.na(merged_type) | merged_type == "none") {
                    eset_te_name <- paste0(c(eset_tr_name, dataset_te_name, "te"), collapse="_")
                }
                else {
                    eset_te_name <- paste0(c("eset", dataset_te_name, suffixes), collapse="_")
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
                if (bc_type == "cbt") {
                    exprs(eset_te_bc) <- t(combatbaaddon(bc_obj, Xte, bte))
                }
                else if (bc_type == "ctr") {
                    exprs(eset_te_bc) <- t(meancenteraddon(bc_obj, Xte, bte))
                }
                else if (bc_type == "fab") {
                    exprs(eset_te_bc) <- t(fabatchaddon(bc_obj, Xte, bte))
                }
                else if (bc_type == "qnorm") {
                    exprs(eset_te_bc) <- t(qunormaddon(bc_obj, Xte))
                }
                else if (bc_type == "rta") {
                    exprs(eset_te_bc) <- t(ratioaaddon(bc_obj, Xte, bte))
                }
                else if (bc_type == "rtg") {
                    exprs(eset_te_bc) <- t(ratiogaddon(bc_obj, Xte, bte))
                }
                else if (bc_type == "std") {
                    exprs(eset_te_bc) <- t(standardizeaddon(bc_obj, Xte, bte))
                }
                else if (bc_type == "sva") {
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
