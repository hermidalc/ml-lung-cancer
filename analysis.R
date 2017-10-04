#!/usr/bin/env R

library('tibble')
library('readxl')
library('Biobase')
library('hgu133plus2.db')
library('annotate')
library('genefilter')
library('limma')
# read Excel data
datafile <- '/home/hermidalc/data/nci-lhc-nsclc/japan_luad/AffyU133+2array_NCC_226ADC_16Normal_MAS5normalized_test.xlsx'
exprs <- as.matrix(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 3,
    col_names = TRUE,
    trim_ws = TRUE
)), var="Probeset ID"))
pData <- AnnotatedDataFrame(column_to_rownames(as.data.frame(read_excel(
    datafile,
    sheet = 2,
    col_names = TRUE,
    trim_ws = TRUE
)), var="Biology ID"))
# build eset
eset <- ExpressionSet(
    assayData = exprs,
    phenoData = pData,
    annotation="hgu133plus2"
)
# annotate eset
probesetIds <- featureNames(eset)
geneSymbols <- getSYMBOL(probesetIds,"hgu133plus2.db")
fData(eset) <- data.frame(Symbol=geneSymbols)
# filter control probesets
eset.filtered <- featureFilter(eset,
    require.entrez=FALSE,
    require.GOBP=FALSE, require.GOCC=FALSE,
    require.GOMF=FALSE, require.CytoBand=FALSE,
    remove.dupEntrez=FALSE, feature.exclude="^AFFX"
)
# limma analysis
design <- model.matrix(~0+factor(pData(eset.filtered)$'T/N'))
colnames(design) <- c("Normal", "Tumor")
fit <- lmFit(eset.filtered, design)
contrast.matrix <- makeContrasts(TumorvsNormal=Tumor-Normal, levels=design)
fit.contrasts <- contrasts.fit(fit, contrast.matrix)
fit.b <- eBayes(fit.contrasts)
results = decideTests(fit.b)
write.fit(fit.b, results, "data.txt")
