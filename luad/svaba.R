svaba <- function(x, y, batch, mod, mod0, numsv, controls=NULL, algorithm="exact") {
    if(any(is.na(x)))
        stop("Data contains missing values.")
    if(!is.matrix(x))
        stop("'x' has to be of class 'matrix'.")
    if(!(is.factor(y) & all(levels(y)==(1:2))))
        stop("'y' has to be of class 'factor' with levels '1' and '2'.")
    if (numsv!=0) {
        if (!is.null(controls)) {
          svobj <- sva::sva(t(x), mod, mod0, method="supervised", n.sv=numsv, controls=controls)
        }
        else {
          svobj <- sva::sva(t(x), mod, mod0, n.sv=numsv)
        }
        nmod <- dim(mod)[2]
        mod <- cbind(mod, svobj$sv)
        gammahat <- (t(x) %*% mod %*% solve(t(mod) %*% mod))[, (nmod + 1):(nmod + numsv)]
        db = t(x) - gammahat %*% t(svobj$sv)
        xadj <- t(db)
        params <- list(xadj = xadj, xtrain = x, ytrain = y, svobj = svobj, algorithm = algorithm)
    }
    else {
        warning("Estimated number of factors was zero.")
        xadj <- x
        params <- list(xadj = xadj, xtrain = x, ytrain = y, svobj = NULL, algorithm = algorithm)
    }
    params$nbatches <- length(unique(batch))
    params$batch <- batch
    class(params) <- "svatrain"
    return(params)
}

# regress surrogate vars out of exprs to get batch corrected exprs
svabaxadj <- function(exprs, mod, svobj) {
    X <- cbind(mod, svobj$sv)
    Hat <- solve(t(X) %*% X) %*% t(X)
    beta <- (Hat %*% t(exprs))
    P <- ncol(mod)
    exprsadj <- exprs - t(as.matrix(X[,-c(1:P)]) %*% beta[-c(1:P),])
    return(exprsadj)
}
