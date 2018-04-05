rmatrain <- function(affybatchtrain) {
    # perform RMA
    cat("Performing background correction")
    abg <- affy::bg.correct.rma(affybatchtrain)
    cat("Performing normalization/summarization\n")
    a.nrm.rma <- bapred::normalizeAffyBatchqntval(abg, 'pmonly')
    # store parameters for add-on quantile normalization
    rmadoc <- Biobase::experimentData(a.nrm.rma)@preprocessing[['val']]
    summ.rma <- bapred::summarizeval2(a.nrm.rma)
    sumdoc.rma <- Biobase::experimentData(summ.rma)@preprocessing$val$probe.effects
    # extract gene expressions
    exprs.train.rma <- exprs(summ.rma)
    rma_obj <- list(xnorm=t(exprs.train.rma), rmadoc=rmadoc, sumdoc.rma=sumdoc.rma, nfeature=nrow(exprs.train.rma))
    class(rma_obj) <- "rmatrain"
    return(rma_obj)
}

rmaaddon <- function(rma_obj, affybatchtest, parallel=TRUE) {
    if(class(rma_obj) != "rmatrain")
        stop("Input parameter 'rma_obj' has to be of class 'rmatrain'.")
    # perform RMA with add-on quantile normalization
    cat("Performing background correction\n")
    abg <- affy::bg.correct.rma(affybatchtest)
    cat("Performing add-on normalization/summarization")
    if (parallel) {
        suppressPackageStartupMessages(require("doParallel"))
        registerDoParallel(cores=detectCores())
        exprs.test.rma <- foreach (cel=1:length(abg), .combine="cbind") %dopar% {
            ab.add <- bapred::extractAffybatch(cel, abg)
            abo.nrm.rma  <- bapred::normalizeqntadd(ab.add, rma_obj$rmadoc$mqnts)
            eset <- bapred::summarizeadd2(abo.nrm.rma, rma_obj$sumdoc.rma)
            cat(".")
            exprs(eset)
        }
    }
    else {
        exprs.test.rma <- matrix(0, nrow=rma_obj$nfeature, ncol=length(abg))
        for (cel in 1:length(abg)) {
            ab.add <- bapred::extractAffybatch(cel, abg)
            abo.nrm.rma  <- bapred::normalizeqntadd(ab.add, rma_obj$rmadoc$mqnts)
            eset <- bapred::summarizeadd2(abo.nrm.rma, rma_obj$sumdoc.rma)
            exprs.test.rma[,cel] <- exprs(eset)
            cat(".")
        }
    }
    cat("Done.\n")
    return(t(exprs.test.rma))
}
