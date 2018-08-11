#!/usr/bin/env Rscript

options(warn=1)
suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))
source("config.R")
eset_tr_name <- "eset_gse8894_gse30219_gse31210_gse37745_gse50081_gcrma_merged_tr"
eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
load(eset_tr_file)
new_batch_nums <- phenoData(eset_gse8894_gse30219_gse31210_gse37745_gse50081_gcrma_merged_tr)$Batch
new_batch_nums[new_batch_nums == 1] <- 0
new_batch_nums[new_batch_nums == 2] <- 1
new_batch_nums[new_batch_nums == 3] <- 2
new_batch_nums[new_batch_nums == 0] <- 3
phenoData(eset_gse8894_gse30219_gse31210_gse37745_gse50081_gcrma_merged_tr)$Batch <- new_batch_nums
Xtr <- t(exprs(get(eset_tr_name)))
ptr <- pData(get(eset_tr_name))
ytr <- as.factor(ptr$Class + 1)
btr <- ptr$Batch
butr <- sort(unique(btr))
for (j in 1:length(butr)) { if (j != butr[j]) { btr <- replace(btr, btr == butr[j], j) } }
btr <- as.factor(btr)
svg(file="pcplot_5_datasets_gcrma_merged_v1.svg")
pcplot(Xtr, btr, col=as.numeric(unique(btr)), main="PCA 5 datasets GC-RMA Merged")
legend("bottomleft", legend=c("gse8894","gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)), pch=20, bty="n")
dev.off()
pc <- prcomp(Xtr, scale.=FALSE)
svg(file="pcplot_5_datasets_gcrma_merged_v2.svg")
plot(pc$x[,1], pc$x[,2], col=as.numeric(btr), main="PCA 5 datasets GC-RMA Merged", xlab="PC1", ylab="PC2")
legend("bottomleft", legend=c("gse8894","gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)), pch=20, bty="n")
dev.off()
eset_tr_name <- "eset_gse8894_gse30219_gse31210_gse37745_gse50081_gcrma_merged_cbt_tr"
eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
load(eset_tr_file)
new_batch_nums <- phenoData(eset_gse8894_gse30219_gse31210_gse37745_gse50081_gcrma_merged_cbt_tr)$Batch
new_batch_nums[new_batch_nums == 1] <- 0
new_batch_nums[new_batch_nums == 2] <- 1
new_batch_nums[new_batch_nums == 3] <- 2
new_batch_nums[new_batch_nums == 0] <- 3
phenoData(eset_gse8894_gse30219_gse31210_gse37745_gse50081_gcrma_merged_cbt_tr)$Batch <- new_batch_nums
Xtr <- t(exprs(get(eset_tr_name)))
ptr <- pData(get(eset_tr_name))
ytr <- as.factor(ptr$Class + 1)
btr <- ptr$Batch
butr <- sort(unique(btr))
for (j in 1:length(butr)) { if (j != butr[j]) { btr <- replace(btr, btr == butr[j], j) } }
btr <- as.factor(btr)
svg(file="pcplot_5_datasets_gcrma_merged_cbt_v1.svg")
pcplot(Xtr, btr, col=as.numeric(unique(btr)), main="PCA 5 datasets GC-RMA Merged ComBat")
legend("bottomleft", legend=c("gse8894","gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)), pch=20, bty="n")
dev.off()
pc <- prcomp(Xtr, scale.=FALSE)
svg(file="pcplot_5_datasets_gcrma_merged_cbt_v2.svg")
plot(pc$x[,1], pc$x[,2], col=as.numeric(btr), main="PCA 5 datasets GC-RMA Merged ComBat", xlab="PC1", ylab="PC2")
legend("bottomleft", legend=c("gse8894","gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)), pch=20, bty="n")
dev.off()
eset_tr_name <- "eset_gse30219_gse31210_gse37745_gse50081_gcrma_tr"
eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
load(eset_tr_file)
new_batch_nums <- phenoData(eset_gse30219_gse31210_gse37745_gse50081_gcrma_tr)$Batch
new_batch_nums[new_batch_nums == 1] <- 2
new_batch_nums[new_batch_nums == 3] <- 1
new_batch_nums[new_batch_nums == 4] <- 3
new_batch_nums[new_batch_nums == 5] <- 4
phenoData(eset_gse30219_gse31210_gse37745_gse50081_gcrma_tr)$Batch <- new_batch_nums
Xtr <- t(exprs(get(eset_tr_name)))
ptr <- pData(get(eset_tr_name))
ytr <- as.factor(ptr$Class + 1)
btr <- ptr$Batch
butr <- sort(unique(btr))
for (j in 1:length(butr)) { if (j != butr[j]) { btr <- replace(btr, btr == butr[j], j) } }
btr <- as.factor(btr)
svg(file="pcplot_4_datasets_gcrma_v1.svg")
pcplot(Xtr, btr, col=as.numeric(unique(btr)) + 1, main="PCA 4 datasets GC-RMA")
legend("topleft", legend=c("gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)) + 1, pch=20, bty="n")
dev.off()
pc <- prcomp(Xtr, scale.=FALSE)
svg(file="pcplot_4_datasets_gcrma_v2.svg")
plot(pc$x[,1], pc$x[,2], col=as.numeric(btr) + 1, main="PCA 4 datasets GC-RMA", xlab="PC1", ylab="PC2")
legend("topleft", legend=c("gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)) + 1, pch=20, bty="n")
dev.off()
eset_tr_name <- "eset_gse30219_gse31210_gse37745_gse50081_rma_merged_tr"
eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
load(eset_tr_file)
new_batch_nums <- phenoData(eset_gse30219_gse31210_gse37745_gse50081_rma_merged_tr)$Batch
new_batch_nums[new_batch_nums == 1] <- 2
new_batch_nums[new_batch_nums == 3] <- 1
new_batch_nums[new_batch_nums == 4] <- 3
new_batch_nums[new_batch_nums == 5] <- 4
phenoData(eset_gse30219_gse31210_gse37745_gse50081_rma_merged_tr)$Batch <- new_batch_nums
Xtr <- t(exprs(get(eset_tr_name)))
ptr <- pData(get(eset_tr_name))
ytr <- as.factor(ptr$Class + 1)
btr <- ptr$Batch
butr <- sort(unique(btr))
for (j in 1:length(butr)) { if (j != butr[j]) { btr <- replace(btr, btr == butr[j], j) } }
btr <- as.factor(btr)
svg(file="pcplot_4_datasets_rma_merged_v1.svg")
pcplot(Xtr, btr, col=as.numeric(unique(btr)) + 1, main="PCA 4 datasets RMA Merged")
legend("bottomleft", legend=c("gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)) + 1, pch=20, bty="n")
dev.off()
pc <- prcomp(Xtr, scale.=FALSE)
svg(file="pcplot_4_datasets_rma_merged_v2.svg")
plot(pc$x[,1], pc$x[,2], col=as.numeric(btr) + 1, main="PCA 4 datasets RMA Merged", xlab="PC1", ylab="PC2")
legend("bottomleft", legend=c("gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)) + 1, pch=20, bty="n")
dev.off()
eset_tr_name <- "eset_gse30219_gse31210_gse37745_gse50081_rma_tr"
eset_tr_file <- paste0("data/", eset_tr_name, ".Rda")
load(eset_tr_file)
new_batch_nums <- phenoData(eset_gse30219_gse31210_gse37745_gse50081_rma_tr)$Batch
new_batch_nums[new_batch_nums == 1] <- 2
new_batch_nums[new_batch_nums == 3] <- 1
new_batch_nums[new_batch_nums == 4] <- 3
new_batch_nums[new_batch_nums == 5] <- 4
phenoData(eset_gse30219_gse31210_gse37745_gse50081_rma_tr)$Batch <- new_batch_nums
Xtr <- t(exprs(get(eset_tr_name)))
ptr <- pData(get(eset_tr_name))
ytr <- as.factor(ptr$Class + 1)
btr <- ptr$Batch
butr <- sort(unique(btr))
for (j in 1:length(butr)) { if (j != butr[j]) { btr <- replace(btr, btr == butr[j], j) } }
btr <- as.factor(btr)
svg(file="pcplot_4_datasets_rma_v1.svg")
pcplot(Xtr, btr, col=as.numeric(unique(btr)) + 1, main="PCA 4 datasets RMA")
legend("topleft", legend=c("gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)) + 1, pch=20, bty="n")
dev.off()
pc <- prcomp(Xtr, scale.=FALSE)
svg(file="pcplot_4_datasets_rma_v2.svg")
plot(pc$x[,1], pc$x[,2], col=as.numeric(btr) + 1, main="PCA 4 datasets RMA", xlab="PC1", ylab="PC2")
legend("topleft", legend=c("gse30219","gse31210","gse37745","gse50081"), col=as.numeric(unique(btr)) + 1, pch=20, bty="n")
dev.off()
