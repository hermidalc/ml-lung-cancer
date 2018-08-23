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
        cat("Keeping", length(idx2keep2), "components with R2(ref2) higher than", t2, "\n")
        idx2remove = setdiff(idx2remove,idx2keep2)
        idx2keep = union(idx2keep,idx2keep2)
        R2 = cbind(R2,factR2_2$allR2)
    }
    U = A[,idx2keep]
    V = B[,idx2keep]
    cat("Removing", length(idx2remove), "components with R2(ref) higher than", t, "\n")
    Xn = U %*% t(V)
    return(list(Xn=Xn,R2=R2,U=U,V=V))
}

R2 <- function(poi,V,poiType,pval=FALSE) {
   #
   #   R2(poi,V,poiType)
   #
   # Args:
   # - V is a p*k matrix, where the rows corresponds to the samples
   # - poi is a matrix p*l, representing the phenotypes of interest
   # - poiType (1*l) is the types of poi: 'continuous' (then a linear
   # regression is used) or 'categorical' (then the mean by class is used)
   #
   # Outputs:
   # - R2(l), higher R^2 value between a column of V and poi(l)
   # - idxCorr(l), index of the column of V giving the higher R^2 value (if many,
   # takes the first one)
   # - allR2(k,l),  R2 value for column k of V with poi l
   #
   #    IF pval =TRUE, return also:   #
   # - pv(l) smaller p-value association between a column of V and poi(l)
   # - idxcorr2(l) index of the column of V giving the smaller p-value (if many,
   #                                                                          # takes the first one)
   # - allpv(k,l),  p-value for column k of V with poi l
   #
   # if missing information in poi, remove the corresponding samples in the R2 computation

   if (is.vector(V) ){ V = matrix(V,ncol=1)}
   if (is.vector(poi)){poi = matrix(poi,nrow =length(poi))}

   p = nrow(V)   # number of samples
   k = ncol(V)    # number of components
   l = length(poiType)  # number of cf/poi to test
   if (is.null(l)){stop("POI type(s) neeeded")}
   p2 = nrow(poi)
   l2 = ncol(poi)

   if( l2 != l){ # checking poi and poiType dimensions compatiblity
      if (p2 == l){  # if poi is transposed (l*p)
         poi = t(poi)
         warning("Transposing poi to match poiType dimension")
         p2 = nrow(poi)
      } else {
         print(poi)
         print(poiType)
         stop("poi dimensions doesn't match poiType dimension")
      }
   }

   if (p != p2){ # checking poi and V dimensions compatiblity
      if (p2 == k){
         warnings("Transposing V to match poi dimension")
         V =t(V)
         k = p
         p = p2
      } else {
         stop("poi and V dimensions incompatible")
      }
   }

   R2 = rep(-1,l)
   names(R2) = colnames(poi)
   idxcorr = R2
   R2_tmp <- matrix(rep(-1,k*l),k,l,dimnames=list(colnames(V),colnames(poi)))    # r2_tmp(k,l) hold the R2 value for column k of V with poi l

   if (pval){
      pv = R2
      idxcorr2 = R2
      pv_tmp <- R2_tmp   # r2_tmp(k,l) hold the R2 value for column k of V with poi l
   }

   for (cmpt in 1:k){    # for each column of V
      cmpt2an <- V[,cmpt]
      for (ipoi in 1:l){
         idx_finite = is.finite(as.factor(poi[,ipoi]))
         poi2an = poi[idx_finite,ipoi]
         cmpt2an_finite=cmpt2an[idx_finite]
         if (poiType[ipoi] == "continuous") {  # estimation by linear regression
            coefs <- coef(lm(cmpt2an_finite~as.numeric(poi2an)))
            cmpt2an_est <- coefs[2]*as.numeric(poi2an)+coefs[1]
            nc <- 2;
         } else if (poiType[ipoi]=="categorical"){  # estimation by classe mean
            classes <- unique(poi2an)
            nc <- length(classes)
            cmpt2an_est <- rep(NA,length(cmpt2an_finite))
            for (icl in 1:length(classes) ){
               idxClasse <- which(poi2an==classes[icl])
               cmpt2an_est[idxClasse] <- mean(cmpt2an_finite[idxClasse])
            }
         } else {
            stop("Incorrect poiType. Select 'continuous' or 'categorical'. ")
         }
         sse <- sum((cmpt2an_finite-cmpt2an_est)^2)
         sst <- sum((cmpt2an_finite-mean(cmpt2an_finite))^2)
         R2_tmp[cmpt,ipoi] <-  1 - sse/sst
         if (pval){
            F <- ((sst-sse)/(nc-1))/(sse/(p-nc))
            pv_tmp[cmpt,ipoi] = 1-pf(F,nc-1,p-nc);
            if (!is.finite(pv_tmp[cmpt,ipoi])) {
               warning(paste("Non finite p-value for component ",cmpt," (pv=",pv_tmp[cmpt,ipoi],", F=",F,"), assigning NA", sep=""))
               pv_tmp[cmpt,ipoi] <- NA
            }
         }
      }
   }

   for (ipoi in 1:l){
      if (pval){
         pv[ipoi] <- min(pv_tmp[,ipoi])
         idxcorr2[ipoi] <- which(pv_tmp[,ipoi] == pv[ipoi])[1]   # if more than one component gives the best R2, takes the first one
      }
      R2[ipoi] <- max(R2_tmp[,ipoi])
      idxcorr[ipoi] <- which(R2_tmp[,ipoi] == R2[ipoi])[1]   # if more than one component gives the best R2, takes the first one
   }

   if (pval){
      return(list(R2=R2,idxcorr=idxcorr,allR2 = R2_tmp, pv=pv,idxcorr2=idxcorr2,allpv = pv_tmp))
   } else {
      return(list(R2=R2,idxcorr=idxcorr,allR2 = R2_tmp))
   }

}

unbiased_stICA <- function(X,k=20,alpha=0.5) {

#
#    [A B W] = unbiased_stICA(X, k,alpha)
#
# Compute factorization of X =A*B' using Jade algorithm
#
# Inputs:
# - X (matrix p*n) is the matrix to factorise
# - alpha (scalar between 0 and 1) is the trade-off between spatial ICA (alpha = 0) et temporal ICA
# (alpha=1) (default alpha =0.5)
# - k (scalar integer) is the number of components to estimate (default = 20)
#
# Ouputs:
# - A (matrix p*k), B (matrix n*k) such that A*B' is an approximation of X
# - W orthogonal matrix k*k minimizing the objective function :
#       f(W) = sum_i ( alpha ||off(W*C_i( Dk^(1-alpha)*Vk)* W')||_F^2
#                   + (1-alpha)||off( W* C_i(Dk^alpha*Uk)*W') ||_F^2  )
#
#       where - C_i(X) is fourth order cumulant-like matrices of X
#             - X = U* D*V^T is the svd decomposition of X, U/D/V_k is
#              the U/D/V matrix keeping only the k first components

# Author: Emilie Renard, december 2013

# References:
#  E. Renard, A. E. Teschendorff, P.-A. Absil
# 'Capturing confounding sources of variation in DNA methylation data by
#  spatiotemporal independent component analysis', submitted to ESANN 2014
#  (Bruges)
#
#  based on code from F. Theis, see
#  F.J. Theis, P. Gruber, I. Keck, A. Meyer-Baese and E.W. Lang,
#  'Spatiotemporal blind source separation using double-sided approximate joint diagonalization',
#  EUSIPCO 2005 (Antalya), 2005.
#
#  Uses Cardoso's diagonalization algorithm based on iterative Given's
#  rotations (matlab-file RJD), see
#  J.-F. Cardoso and A. Souloumiac, 'Jacobi angles for simultaneous diagonalization',
#  SIAM J. Mat. Anal. Appl., vol 17(1), pp. 161-164, 1995

   library(JADE)
   library(corpcor)

   jadeCummulantMatrices <- function(X) {
      # calcs the n(n+1)/2 cum matrices used in JADE, see A. Cichocki and S. Amari. Adaptive blind signal and image
      # processing. John Wiley & Sons, 2002. book, page 173
      # (pdf:205), C.1
      # does not need whitened data X (n x t)
      # Args:
      #   matrix X
      # Returns:
      #   M as n x n x (n(n+1)/2) array, each n x n matrix is one of the n(n+1)/2 cumulant matrices used in JADE

      # adapted code from Matlab implementation of F. Theis, see
      #  F.J. Theis, P. Gruber, I. Keck, A. Meyer-Baese and E.W. Lang,
      #  'Spatiotemporal blind source separation using double-sided approximate joint diagonalization',
      #  EUSIPCO 2005 (Antalya), 2005.

      n <- nrow(X)
      t <- ncol(X)

      M <- array(0,c(n,n,n*(n+1)/2))
      scale <- matrix(1,n,1)/t  # for convenience

      R <- cov(t(X)) # covariance

      k <- 1
      for (p in 1:n){
         #case q=p
         C <- ((scale %*% (X[p,]*X[p,]))*X) %*% t(X)
         E <- matrix(0,n,n)
         E[p,p] <- 1
         M[,,k] <- C - R %*% E %*% R - sum(diag(E %*% R)) * R - R %*% t(E) %*% R
         k <- k+1
         #case q<p
         if (p > 1) {
            for (q in 1:(p-1)){
               C <- ((scale %*% (X[p,]*X[q,]))*X) %*% t(X) * sqrt(2)
               E <- matrix(0,n,n)
               E[p,q] <- 1/sqrt(2)
               E[q,p] <- E[p,q]
               M[,,k] <- C - R %*% E %*% R - sum(diag(E %*% R)) * R - R %*% t(E) %*% R
               k <- k+1
            }
         }
      }

      return(M)

   }

   p <- nrow(X)
   n <- ncol(X)

   dimmin <- min(n,p)

   if (dimmin < k) {
      k <- dimmin
   }
   if (alpha <0 | alpha >1){
      stop("alpha not in [0 1]")
   }

   # Remove the spatiotemporal mean
   Xc <- X - matrix(rep(colMeans(X,dims=1),p),nrow = p,byrow=T);
   Xc <- Xc - matrix(rep(rowMeans(Xc,dims=1),n),nrow = p);

   # SVD of Xc and dimension reduction: keeping only the k first
   # components
   udv <- svd(Xc,k,k)
   D <- diag(udv$d[1:k]); if (k==1) {D <- udv$d[1]}
   U <- udv$u;
   V <- udv$v;

   # Estimation of the cumulant matrices
   nummat <- k*(k+1)/2;
   M <- array(0,c(k,k,2*nummat));
   Bt <- D^(1-alpha) %*% t(V)
   if (alpha == 1) { Bt <- t(V)}
   At <- D^(alpha) %*% t(U)
   if (alpha == 0) { At <- t(U)}
   M[,,1:nummat] <- jadeCummulantMatrices(Bt);
   M[,,(nummat+1):(2*nummat)] <- jadeCummulantMatrices(At)

   # normalization within the groups in order to allow for comparisons using
   # alpha
   M[,,1:nummat] <- alpha*M[,,1:nummat]/mean(sqrt(apply(M[,,1:nummat]*M[,,1:nummat],3,sum)));
   M[,,(nummat+1):(2*nummat)] <- (1-alpha)*M[,,(nummat+1):(2*nummat)]/mean(sqrt(apply(M[,,(nummat+1):(2*nummat)]*M[,,(nummat+1):(2*nummat)],3,sum)));

   # Joint diagonalization
   Worth <- rjd(M,eps = 1e-06, maxiter = 1000);
   Wo <-t (Worth$V);

   # Computation of A and B
   A0 <- U %*% D^(alpha) %*% solve(Wo);
   B0 <- V%*% D^(1-alpha) %*% t(Wo);
   if (alpha == 1) { B0 <- V %*% t(Wo)}
   if (alpha == 0) { A0 <- U %*% solve(Wo)}

   # Add transformed means
   meanCol <- matrix(colMeans(X,dims=1),ncol =1); # spatial means
   meanRows <- matrix(rowMeans(X,dims=1),ncol = 1); # temporal means

   meanB <- pseudoinverse(A0) %*% (meanRows);
   meanA <- pseudoinverse(B0) %*% (meanCol);

   Bfin <- B0 + matrix(rep(meanB,n),nrow = n,byrow=T)
   Afin <- A0 + matrix(rep(meanA,p),nrow = p,byrow=T)

   return(list(A=Afin,B=Bfin,W=Wo))

}
