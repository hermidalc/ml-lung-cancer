#!/usr/bin/env Rscript

suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
# suppressPackageStartupMessages(library("sva"))
source("svaba.R")
source("config.R")

cmd_args <- commandArgs(trailingOnly = TRUE)
for (bc_type in cmd_args) {
    for (i in 1:length(eset_merged_tr_names)) {
        if (bc_type in c("stica", "svd")) {
            load(paste0("data/", eset_merged_tr_names[i], ".Rda"))
            load(paste0("data/", eset_merged_te_names[i], ".Rda"))
            ptr <- pData(get(eset_merged_tr_names[i]))
            Xtr <- exprs(get(eset_merged_tr_names[i]))
            Xte <- exprs(get(eset_merged_te_names[i]))
            if (bc_type == "stica") {
                for (alpha in stica_alphas) {
                    alpha_name <- gsub("[^0-9]", "", alpha)
                    eset_tr_bc_name <- paste0(eset_merged_tr_names[i], "_tr_", bc_type, alpha_name)
                    eset_te_bc_name <- paste0(eset_merged_te_names[i], "_te_", bc_type, alpha_name)
                    print(paste(eset_tr_bc_name, "->", eset_te_bc_name))
                    bc_obj <- normFact("stICA", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical", alpha=alpha)
                    eset_tr_bc <- get(eset_merged_tr_names[i])
                    exprs(eset_tr_bc) <- bc_obj$Xn
                    assign(eset_tr_bc_name, eset_tr_bc)
                    save(list=eset_tr_bc_name, file=paste0("data/", eset_tr_bc_name, ".Rda"))
                    eset_tr_bc_obj_name <- paste0(eset_tr_bc_name, "_obj")
                    assign(eset_tr_bc_obj_name, bc_obj)
                    save(list=eset_tr_bc_obj_name, file=paste0("data/", eset_tr_bc_obj_name, ".Rda"))
                    eset_te_bc <- get(eset_merged_te_names[i])
                    # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
                    # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
                    # Xte_n = dot(U,Vte.T)
                    exprs(eset_te_bc) <- bc_obj$U %*% t((t(Xte) %*% bc_obj$U) %*% solve(t(bc_obj$U) %*% bc_obj$U))
                    assign(eset_te_bc_name, eset_te_bc)
                    save(list=eset_te_bc_name, file=paste0("data/", eset_te_bc_name, ".Rda"))
                }
            }
            else if (bc_type == "svd") {
                eset_tr_bc_name <- paste0(eset_merged_tr_names[i], "_tr_", bc_type)
                eset_te_bc_name <- paste0(eset_merged_te_names[i], "_te_", bc_type)
                print(paste(eset_tr_bc_name, "->", eset_te_bc_name))
                load(paste0("data/", eset_merged_tr_names[i], ".Rda"))
                bc_obj <- normFact("SVD", Xtr, ptr$Batch, "categorical", ref2=ptr$Relapse, refType2="categorical")
                eset_tr_bc <- get(eset_merged_tr_names[i])
                exprs(eset_tr_bc) <- bc_obj$Xn
                assign(eset_tr_bc_name, eset_tr_bc)
                save(list=eset_tr_bc_name, file=paste0("data/", eset_tr_bc_name, ".Rda"))
                eset_tr_bc_obj_name <- paste0(eset_tr_bc_name, "_obj")
                assign(eset_tr_bc_obj_name, bc_obj)
                save(list=eset_tr_bc_obj_name, file=paste0("data/", eset_tr_bc_obj_name, ".Rda"))
                load(paste0("data/", eset_merged_te_names[i], ".Rda"))
                eset_te_bc <- get(eset_merged_te_names[i])
                # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
                # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
                # Xte_n = dot(U,Vte.T)
                exprs(eset_te_bc) <- bc_obj$U %*% t((t(Xte) %*% bc_obj$U) %*% solve(t(bc_obj$U) %*% bc_obj$U))
                assign(eset_te_bc_name, eset_te_bc)
                save(list=eset_te_bc_name, file=paste0("data/", eset_te_bc_name, ".Rda"))
            }
        }
        else if (bc_type in c("cbt", "fab", "std", "sva")) {
            eset_tr_bc_name <- paste0(eset_merged_tr_names[i], "_tr_", bc_type)
            eset_te_bc_name <- paste0(eset_merged_te_names[i], "_te_", bc_type)
            print(paste(eset_tr_bc_name, "->", eset_te_bc_name))
            load(paste0("data/", eset_merged_tr_names[i], ".Rda"))
            ptr <- pData(get(eset_merged_tr_names[i]))
            Xtr <- t(exprs(get(eset_merged_tr_names[i])))
            ytr <- as.factor(ptr$Relapse + 1)
            btr <- ptr$Batch
            butr <- sort(unique(btr))
            for (j in 1:length(butr)) {
                if (j != butr[j]) {
                    btr <- replace(btr, btr == butr[j], j)
                }
            }
            btr <- as.factor(btr)
            eset_tr_bc <- get(eset_merged_tr_names[i])
            if (bc_type == "cbt") {
                bc_obj <- combatba(Xtr, btr)
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
            load(paste0("data/", eset_merged_te_names[i], ".Rda"))
            pte <- pData(get(eset_merged_te_names[i]))
            Xte <- t(exprs(get(eset_merged_te_names[i])))
            bte <- pte$Batch
            bute <- sort(unique(bte))
            for (j in 1:length(bute)) {
                if (j != bute[j]) {
                    bte <- replace(bte, bte == bute[j], j)
                }
            }
            bte <- as.factor(bte)
            eset_te_bc <- get(eset_merged_te_names[i])
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
