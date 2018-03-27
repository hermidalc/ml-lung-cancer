#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
# suppressPackageStartupMessages(library("sva"))
source("lib/R/svaba.R")
source("lib/R/normFact.R")
source("lib/R/config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
num_subset <- cmd_args[1]
norm_meth <- cmd_args[2]
dataset_names <- dataset_names[1:num_subset]
dataset_name_combos <- combn(dataset_names, length(dataset_names) - 1)
for (bc_type in cmd_args[3:length(cmd_args)]) {
    if (bc_type %in% c("cbt", "fab", "qnorm", "std", "sva", "stica", "svd")) {
        for (col in 1:ncol(dataset_name_combos)) {
            eset_tr_name <- paste0(c("eset", dataset_name_combos[,col]), collapse="_")
            load(paste0("data/", eset_tr_name, ".Rda"))
            if (bc_type %in% c("stica", "svd")) {
                Xtr <- exprs(get(eset_tr_name))
                ptr <- pData(get(eset_tr_name))
                if (bc_type == "stica") {
                    for (alpha in stica_alphas) {
                        eset_tr_bc_name <- paste0(eset_tr_name, bc_type, gsub("[^0-9]", "", alpha), "_tr")
                        print(paste("Creating: ", eset_tr_bc_name))
                        bc_obj <- normFact(
                            "stICA", Xtr, ptr$Batch, "categorical",
                            ref2=ptr$Relapse, refType2="categorical", k=matfact_k, alpha=alpha
                        )
                        eset_tr_bc <- get(eset_tr_name)
                        exprs(eset_tr_bc) <- bc_obj$Xn
                        assign(eset_tr_bc_name, eset_tr_bc)
                        save(list=eset_tr_bc_name, file=paste0("data/", eset_tr_bc_name, ".Rda"))
                        eset_tr_bc_obj_name <- paste0(eset_tr_bc_name, "_obj")
                        assign(eset_tr_bc_obj_name, bc_obj)
                        save(list=eset_tr_bc_obj_name, file=paste0("data/", eset_tr_bc_obj_name, ".Rda"))
                        for (dataset_te_name in setdiff(dataset_names, dataset_name_combos[,col])) {
                            eset_te_name <- paste0("eset_", dataset_te_name)
                            load(paste0("data/", eset_te_name, ".Rda"))
                            Xte <- exprs(get(eset_te_name))
                            eset_te_bc_name <- paste0(eset_tr_bc_name, "_", dataset_te_name, "_te")
                            print(paste("Creating: ", eset_te_bc_name))
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
                }
                else if (bc_type == "svd") {
                    eset_tr_bc_name <- paste0(eset_tr_name, bc_type, "_tr")
                    print(paste("Creating: ", eset_tr_bc_name))
                    bc_obj <- normFact(
                        "SVD", Xtr, ptr$Batch, "categorical",
                        ref2=ptr$Relapse, refType2="categorical", k=matfact_k
                    )
                    eset_tr_bc <- get(eset_tr_name)
                    exprs(eset_tr_bc) <- bc_obj$Xn
                    assign(eset_tr_bc_name, eset_tr_bc)
                    save(list=eset_tr_bc_name, file=paste0("data/", eset_tr_bc_name, ".Rda"))
                    eset_tr_bc_obj_name <- paste0(eset_tr_bc_name, "_obj")
                    assign(eset_tr_bc_obj_name, bc_obj)
                    save(list=eset_tr_bc_obj_name, file=paste0("data/", eset_tr_bc_obj_name, ".Rda"))
                    for (dataset_te_name in setdiff(dataset_names, dataset_name_combos[,col])) {
                        eset_te_name <- paste0("eset_", dataset_te_name)
                        load(paste0("data/", eset_te_name, ".Rda"))
                        Xte <- exprs(get(eset_te_name))
                        eset_te_bc_name <- paste0(eset_tr_bc_name, "_", dataset_te_name, "_te")
                        print(paste("Creating: ", eset_te_bc_name))
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
            }
            else if (bc_type %in% c("cbt", "fab", "qnorm", "std", "sva")) {
                eset_tr_bc_name <- paste0(eset_tr_name, bc_type, "_tr")
                Xtr <- t(exprs(get(eset_tr_name)))
                ptr <- pData(get(eset_tr_name))
                ytr <- as.factor(ptr$Relapse + 1)
                btr <- ptr$Batch
                butr <- sort(unique(btr))
                for (j in 1:length(butr)) {
                    if (j != butr[j]) {
                        btr <- replace(btr, btr == butr[j], j)
                    }
                }
                btr <- as.factor(btr)
                eset_tr_bc <- get(eset_tr_name)
                if (bc_type == "cbt") {
                    bc_obj <- combatba(Xtr, btr)
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
                else if (bc_type == "std") {
                    bc_obj <- standardize(Xtr, btr)
                    exprs(eset_tr_bc) <- t(bc_obj$xadj)
                }
                else if (bc_type == "sva") {
                    mod <- model.matrix(~as.factor(Relapse), data=ptr)
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
                for (dataset_te_name in setdiff(dataset_names, dataset_name_combos[,col])) {
                    eset_te_bc_name <- paste0(eset_tr_bc_name, "_", dataset_te_name, "_te")
                    print(paste("Creating: ", eset_te_bc_name))
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
                    eset_te_bc <- get(eset_te_name)
                    if (bc_type == "cbt") {
                        exprs(eset_te_bc) <- t(combatbaaddon(bc_obj, Xte, bte))
                    }
                    else if (bc_type == "fab") {
                        exprs(eset_te_bc) <- t(fabatchaddon(bc_obj, Xte, bte))
                    }
                    else if (bc_type == "qnorm") {
                        exprs(eset_te_bc) <- t(qunormaddon(bc_obj, Xte))
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
}
