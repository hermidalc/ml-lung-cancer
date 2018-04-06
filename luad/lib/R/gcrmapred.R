gcrmatrain <- function(affybatchtrain, affinities) {
    # perform GCRMA
    affybatchtrain <- gcrma::bg.adjust.gcrma(
        affybatchtrain, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE
    )
    cat("Performing normalization/summarization\n")
    affybatchtrain <- bapred::normalizeAffyBatchqntval(affybatchtrain, 'pmonly')
    # store parameters for add-on quantile normalization
    rmadoc <- Biobase::experimentData(affybatchtrain)@preprocessing[['val']]
    summ.rma <- bapred::summarizeval2(affybatchtrain)
    sumdoc.rma <- Biobase::experimentData(summ.rma)@preprocessing$val$probe.effects
    # extract gene expressions
    exprs.train.rma <- exprs(summ.rma)
    gcrma_obj <- list(xnorm=t(exprs.train.rma), rmadoc=rmadoc, sumdoc.rma=sumdoc.rma, nfeature=nrow(exprs.train.rma))
    class(gcrma_obj) <- "gcrmatrain"
    return(gcrma_obj)
}

gcrmaaddon <- function(gcrma_obj, affybatchtest, affinities, bg.correct=TRUE, parallel=TRUE) {
    if (class(gcrma_obj) != "gcrmatrain")
        stop("Input parameter 'gcrma_obj' has to be of class 'gcrmatrain'.")
    # perform GCRMA with add-on quantile normalization
    if (bg.correct) {
        affybatchtest <- gcrma::bg.adjust.gcrma(
            affybatchtest, affinity.info=affinities, type="fullmodel", verbose=TRUE, fast=FALSE
        )
    }
    cat("Performing add-on normalization/summarization")
    if (parallel) {
        suppressPackageStartupMessages(require("doParallel"))
        registerDoParallel(cores=max(detectCores()/2, 1))
        exprs.test.rma <- foreach (cel=1:length(affybatchtest), .combine="cbind") %dopar% {
            ab.add <- bapred::extractAffybatch(cel, affybatchtest)
            abo.nrm.rma  <- bapred::normalizeqntadd(ab.add, gcrma_obj$rmadoc$mqnts)
            eset <- bapred::summarizeadd2(abo.nrm.rma, gcrma_obj$sumdoc.rma)
            cat(".")
            exprs(eset)
        }
    }
    else {
        exprs.test.rma <- matrix(0, nrow=gcrma_obj$nfeature, ncol=length(affybatchtest))
        for (cel in 1:length(affybatchtest)) {
            ab.add <- bapred::extractAffybatch(cel, affybatchtest)
            abo.nrm.rma  <- bapred::normalizeqntadd(ab.add, gcrma_obj$rmadoc$mqnts)
            eset <- bapred::summarizeadd2(abo.nrm.rma, gcrma_obj$sumdoc.rma)
            exprs.test.rma[,cel] <- exprs(eset)
            cat(".")
        }
    }
    cat("Done.\n")
    return(list(xnorm=t(exprs.test.rma), abg=affybatchtest))
}
