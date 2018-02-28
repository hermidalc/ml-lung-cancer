normFact <- function(fact,X,ref,refType,k=20,t=0.5,ref2=NULL,refType2=NULL,t2=0.5,...) {
#
#    Function to normalize data X by factorizing X=A*t(B) and removing components having a R2(ref) value higher than threshold t.
#    If ref2 is defined, the components with R2(ref2) higher than threshold t2 are kept.
#
#    Inputs:
#    - fact : factorization method, 'SVD' or 'stICA'
#    - X :(matrix n*p) features*samples matrix to normalize
#    - ref: (vector n) variable representing the information we want to remove from  X
#    - refType : type of ref, 'categorical' or 'continuous' to indicates which linear model to use (class means or linear regression)
#    - k: rank of the low-rank decomposition
#    - t: scalar in [0,1], if R2(cmpt, ref)>t the cmpt is removed from  X   to normalize
#    - ref2: (vector n*l) ref2[,i] represents the ith information we want to not remove from  X
#    - refType2: refType2[i] gives the type of  ref2[,i]  , 'categorical' or 'continuous' to indicates which linear model to use (class means or linear regression)
#    - t2: (vector 1*l)   scalar(s) in [0,1], if R2(cmpt,  ref2[,i]  )> t2[i]   the cmpt is kept in  X  , if  t2   is a scalar this threshold is considered for all  ref2[,i]
#    - ... values to pass to factorization method (typically, alpha value if facotorization by stICA)
#
#  Outputs:
#
#    - Xn : matrix n*p, normalized version of  X
#    - R2 : R2[k,l] gives the R2 between  B[k,]   and the ref l ( ref   or  ref2  )
#    - bestSV : components of  B   correlating with  ref   (but not with  ref2  ), removed from  X   to normalize
#    - A :  A in the matrix factorization X=A*t(B)
#    - B : B in the matrix factorization X=A*t(B)
#
#    Renard E., Branders S. and Absil P.-A.: Independent Component Analysis to Remove Batch Effects from Merged Microarray Datasets (WABI2016)

    source("lib/R/unbiased_stICA.R")
    source("lib/R/R2.R")

    if (fact=='stICA') {
        obj = unbiased_stICA(X,k,...)
        A = obj$A
        B = obj$B
    }
    else if (fact == 'SVD') {
        obj = svd(X,nu=k,nv=k)
        A = obj$u %*% diag(obj$d[1:k],k)
        B = obj$v
    }
    else {
        stop("Factorization method should be SVD or stICA")
    }
    factR2 = R2(ref,B,refType)
    idx2remove = which(factR2$allR2>t)
    idx2keep = which(factR2$allR2<=t)
    R2 = factR2$allR2
    if (t < 0 | t > 1) { stop("t not in [0 1]") }
    if (!is.null(ref2)) {
        if (sum(t2 < 0 | t2 > 1)) { stop("t2 not in [0 1]") }
        factR2_2 = R2(ref2,B,refType2)
        idx2 = c()
        if (length(t2)!=length(refType2)) {
            if (length(t2)==1) {
                t2=rep(t2,length(refType2))
            }
            else {
                stop("length(t2) sould be equal to 1 or length(refType2)")
            }
        }
        for (i in 1:length(refType2)) {
             idx2 = c(idx2,which(factR2_2$allR2[,i]>t2[i]))
        }
        idx2keep2 = intersect(idx2remove,idx2)
        print(paste("Keeping",length(idx2keep2), "components with R2(ref2) higher than",t2))
        idx2remove = setdiff(idx2remove,idx2keep2)
        idx2keep = union(idx2keep,idx2keep2)
        R2 = cbind(R2,factR2_2$allR2)
    }
    U = A[,idx2keep]
    V = B[,idx2keep]
    print(paste("Removing",length(idx2remove),"components with R2(ref) higher than",t))
    Xn = U %*% t(V)
    return(list(Xn=Xn,R2=R2,U=U,V=V))
}
