#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
# suppressPackageStartupMessages(library("sva"))
source("svaba.R")
source("normFact.R")
source("config.R")

cmd_args <- commandArgs(trailingOnly=TRUE)
dataset_name_combos <- combn(dataset_names, length(dataset_names) - 1)
for (bc_type in cmd_args) {
    if (bc_type %in% c("cbt", "fab", "std", "sva", "stica", "svd")) {
        for (col in 1:ncol(dataset_name_combos)) {
            eset_tr_name <- paste0(c("eset", dataset_name_combos[,col]), collapse="_")
            eset_te_name <- paste0(c("eset", setdiff(dataset_names, dataset_name_combos[,col])), collapse="_")
            load(paste0("data/", eset_tr_name, ".Rda"))
            load(paste0("data/", eset_te_name, ".Rda"))
            if (bc_type %in% c("stica", "svd")) {
                ptr <- pData(get(eset_tr_name))
                Xtr <- exprs(get(eset_tr_name))
                Xte <- exprs(get(eset_te_name))
                if (bc_type == "stica") {
                    for (alpha in stica_alphas) {
                        alpha_name <- gsub("[^0-9]", "", alpha)
                        eset_tr_bc_name <- paste0(eset_tr_name, "_tr_", bc_type, alpha_name)
                        eset_te_bc_name <- paste0(eset_te_name, "_te_", bc_type, alpha_name)
                        print(paste(eset_tr_bc_name, "->", eset_te_bc_name))
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
                        # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
                        # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
                        # Xte_n = dot(U,Vte.T)
                        eset_te_bc <- get(eset_te_name)
                        exprs(eset_te_bc) <- bc_obj$U %*% t((t(Xte) %*% bc_obj$U) %*% solve(t(bc_obj$U) %*% bc_obj$U))
                        assign(eset_te_bc_name, eset_te_bc)
                        save(list=eset_te_bc_name, file=paste0("data/", eset_te_bc_name, ".Rda"))
                    }
                }
                else if (bc_type == "svd") {
                    eset_tr_bc_name <- paste0(eset_tr_name, "_tr_", bc_type)
                    eset_te_bc_name <- paste0(eset_te_name, "_te_", bc_type)
                    print(paste(eset_tr_bc_name, "->", eset_te_bc_name))
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
                    # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
                    # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
                    # Xte_n = dot(U,Vte.T)
                    eset_te_bc <- get(eset_te_name)
                    exprs(eset_te_bc) <- bc_obj$U %*% t((t(Xte) %*% bc_obj$U) %*% solve(t(bc_obj$U) %*% bc_obj$U))
                    assign(eset_te_bc_name, eset_te_bc)
                    save(list=eset_te_bc_name, file=paste0("data/", eset_te_bc_name, ".Rda"))
                }
            }
            else if (bc_type %in% c("cbt", "fab", "std", "sva")) {
                eset_tr_bc_name <- paste0(eset_tr_name, "_tr_", bc_type)
                eset_te_bc_name <- paste0(eset_te_name, "_te_", bc_type)
                print(paste(eset_tr_bc_name, "->", eset_te_bc_name))
                ptr <- pData(get(eset_tr_name))
                Xtr <- t(exprs(get(eset_tr_name)))
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
                }
                else if (bc_type == "fab") {
                    bc_obj <- fabatch(Xtr, ytr, btr)
                }
                else if (bc_type == "std") {
                    bc_obj <- standardize(Xtr, btr)
                }
                else if (bc_type == "sva") {
                    mod <- model.matrix(~as.factor(Relapse), data=ptr)
                    mod0 <- model.matrix(~1, data=ptr)
                    # ctrls <- as.numeric(grepl("^AFFX", rownames(t(Xtr))))
                    bc_obj <- svaba(Xtr, btr, mod, mod0, algorithm="fast")
                }
                exprs(eset_tr_bc) <- t(bc_obj$xadj)
                assign(eset_tr_bc_name, eset_tr_bc)
                save(list=eset_tr_bc_name, file=paste0("data/", eset_tr_bc_name, ".Rda"))
                eset_tr_bc_obj_name <- paste0(eset_tr_bc_name, "_obj")
                assign(eset_tr_bc_obj_name, bc_obj)
                save(list=eset_tr_bc_obj_name, file=paste0("data/", eset_tr_bc_obj_name, ".Rda"))
                pte <- pData(get(eset_te_name))
                Xte <- t(exprs(get(eset_te_name)))
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
                else if (bc_type == "std") {
                    exprs(eset_te_bc) <- t(standardizeaddon(bc_obj, Xte, bte))
                }
                else if (bc_type == "sva") {
                    exprs(eset_te_bc) <- t(svabaaddon(bc_obj, Xte))
                }
                assign(eset_te_bc_name, eset_te_bc)
                save(list=eset_te_bc_name, file=paste0("data/", eset_te_bc_name, ".Rda"))
            }
        }
    }
    else if (bc_type %in% c("qnorm")) {
        # currently take first dataset as train
        eset_tr_name <- paste0(c("eset", dataset_names[1]), collapse="_")
        eset_tr_norm_name <- paste0(eset_tr_name, "_tr_", bc_type)
        print(eset_tr_norm_name)
        load(paste0("data/", eset_tr_name, ".Rda"))
        Xtr <- t(exprs(get(eset_tr_name)))
        if (bc_type == "qnorm") {
            norm_obj <- qunormtrain(Xtr)
        }
        eset_tr_norm <- get(eset_tr_name)
        exprs(eset_tr_norm) <- t(norm_obj$xnorm)
        assign(eset_tr_norm_name, eset_tr_norm)
        save(list=eset_tr_norm_name, file=paste0("data/", eset_tr_norm_name, ".Rda"))
        eset_tr_norm_obj_name <- paste0(eset_tr_norm_name, "_obj")
        assign(eset_tr_norm_obj_name, norm_obj)
        save(list=eset_tr_norm_obj_name, file=paste0("data/", eset_tr_norm_obj_name, ".Rda"))
        for (j in 2:length(dataset_names)) {
            eset_te_name <- paste0(c("eset", dataset_names[j]), collapse="_")
            eset_te_norm_name <- paste0(eset_te_name, "_te_norm")
            print(paste(eset_tr_norm_name, "->", eset_te_norm_name))
            load(paste0("data/", eset_te_name, ".Rda"))
            Xte <- t(exprs(get(eset_te_name)))
            eset_te_norm <- get(eset_te_name)
            if (bc_type == "qnorm") {
                exprs(eset_te_norm) <- t(qunormaddon(norm_obj, Xte))
            }
            assign(eset_te_norm_name, eset_te_norm)
            save(list=eset_te_norm_name, file=paste0("data/", eset_te_norm_name, ".Rda"))
        }
    }
}
