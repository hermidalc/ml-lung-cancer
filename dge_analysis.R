#!/usr/bin/env R

library('Biobase')
library('limma')
load("eset.Rda")
design <- model.matrix( ~0 + factor(pData(eset.filtered)$Relapse))
colnames(design) <- c("NoRelapse", "Relapse")
fit <- lmFit(eset.filtered, design)
contrast.matrix <- makeContrasts(RelapseVsNoRelapse=Relapse-NoRelapse, levels=design)
fit.contrasts <- contrasts.fit(fit, contrast.matrix)
fit.b <- eBayes(fit.contrasts)
table <- topTable(fit.b)
results <- decideTests(fit.b)
summary(results)
