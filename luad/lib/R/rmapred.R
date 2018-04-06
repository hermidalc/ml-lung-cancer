rmatrain <- function(affybatchtrain) {
    # perform RMA
    cat("Performing background correction\n")
    affybatchtrain <- affy::bg.correct.rma(affybatchtrain)
    cat("Performing normalization/summarization\n")
    affybatchtrain <- bapred::normalizeAffyBatchqntval(affybatchtrain, 'pmonly')
    # store parameters for add-on quantile normalization
    rmadoc <- Biobase::experimentData(affybatchtrain)@preprocessing[['val']]
    summ.rma <- bapred::summarizeval2(affybatchtrain)
    sumdoc.rma <- Biobase::experimentData(summ.rma)@preprocessing$val$probe.effects
    # extract gene expressions
    exprs.train.rma <- exprs(summ.rma)
    rma_obj <- list(xnorm=t(exprs.train.rma), rmadoc=rmadoc, sumdoc.rma=sumdoc.rma, nfeature=nrow(exprs.train.rma))
    class(rma_obj) <- "rmatrain"
    return(rma_obj)
}

rmaaddon <- function(rma_obj, affybatchtest, bg.correct=TRUE, parallel=TRUE) {
    if(class(rma_obj) != "rmatrain")
        stop("Input parameter 'rma_obj' has to be of class 'rmatrain'.")
    # perform RMA with add-on quantile normalization
    if (bg.correct) {
        cat("Performing background correction\n")
        affybatchtest <- affy::bg.correct.rma(affybatchtest)
    }
    cat("Performing add-on normalization/summarization")
    if (parallel) {
        suppressPackageStartupMessages(require("doParallel"))
        registerDoParallel(cores=max(detectCores()/2, 1))
        exprs.test.rma <- foreach (cel=1:length(affybatchtest), .combine="cbind") %dopar% {
            ab.add <- bapred::extractAffybatch(cel, affybatchtest)
            abo.nrm.rma  <- bapred::normalizeqntadd(ab.add, rma_obj$rmadoc$mqnts)
            eset <- bapred::summarizeadd2(abo.nrm.rma, rma_obj$sumdoc.rma)
            cat(".")
            exprs(eset)
        }
    }
    else {
        exprs.test.rma <- matrix(0, nrow=rma_obj$nfeature, ncol=length(affybatchtest))
        for (cel in 1:length(affybatchtest)) {
            ab.add <- bapred::extractAffybatch(cel, affybatchtest)
            abo.nrm.rma  <- bapred::normalizeqntadd(ab.add, rma_obj$rmadoc$mqnts)
            eset <- bapred::summarizeadd2(abo.nrm.rma, rma_obj$sumdoc.rma)
            exprs.test.rma[,cel] <- exprs(eset)
            cat(".")
        }
    }
    cat("Done.\n")
    return(list(xnorm=t(exprs.test.rma), abg=affybatchtest))
}
