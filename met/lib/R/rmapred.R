suppressPackageStartupMessages(library("Biobase"))
suppressPackageStartupMessages(library("bapred"))

rmatrain <- function(affybatch) {
    cat("Performing normalization/summarization\n")
    affybatch <- normalizeAffyBatchqntval(affybatch, 'pmonly')
    # store parameters for add-on quantile normalization
    rmadoc <- experimentData(affybatch)@preprocessing[['val']]
    summ.rma <- summarizeval2(affybatch)
    sumdoc.rma <- experimentData(summ.rma)@preprocessing$val$probe.effects
    exprs.train.rma <- exprs(summ.rma)
    rma_obj <- list(
        xnorm=t(exprs.train.rma), rmadoc=rmadoc, sumdoc.rma=sumdoc.rma, nfeature=nrow(exprs.train.rma)
    )
    class(rma_obj) <- "rmatrain"
    return(rma_obj)
}

rmaaddon <- function(rma_obj, affybatch, num.cores=detectCores()) {
    if (class(rma_obj) != "rmatrain")
        stop("Input parameter 'rma_obj' has to be of class 'rmatrain'.")
    cat("Performing add-on normalization/summarization")
    if (num.cores > 1) {
        suppressPackageStartupMessages(require("doParallel"))
        registerDoParallel(cores=num.cores)
        exprs.test.rma <- foreach (cel=1:length(affybatch), .combine="cbind") %dopar% {
            ab.add <- extractAffybatch(cel, affybatch)
            abo.nrm.rma  <- normalizeqntadd(ab.add, rma_obj$rmadoc$mqnts)
            eset <- summarizeadd2(abo.nrm.rma, rma_obj$sumdoc.rma)
            cat(".")
            exprs(eset)
        }
    }
    else {
        exprs.test.rma <- matrix(0, nrow=rma_obj$nfeature, ncol=length(affybatch))
        for (cel in 1:length(affybatch)) {
            ab.add <- extractAffybatch(cel, affybatch)
            abo.nrm.rma  <- normalizeqntadd(ab.add, rma_obj$rmadoc$mqnts)
            eset <- summarizeadd2(abo.nrm.rma, rma_obj$sumdoc.rma)
            exprs.test.rma[,cel] <- exprs(eset)
            cat(".")
        }
    }
    cat("Done.\n")
    return(t(exprs.test.rma))
}
