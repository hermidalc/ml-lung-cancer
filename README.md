# NCI LHC NSCLC Research and Data Analysis Repository

## Environment and Analysis Pipeline Setup

1. Install System Dependencies

```bash
sudo dnf install -y libxml-devel libxml2-devel
```

2. Install Anaconda3

https://www.anaconda.com/download/

https://docs.anaconda.com/anaconda/install/linux

2. Setup Conda Environment and Install R, Rpy2, Bioconductor, Limma, etc.

```bash
conda config --add channels conda-forge
conda config --add channels bioconda
conda install -v -y \
r-essentials \
r-devtools \
r-lintr \
r-pamr \
r-minqa \
r-nloptr \
r-rcppeigen \
r-lme4 \
r-corpcor \
rpy2 \
libiconv \
bioconductor-biocinstaller \
bioconductor-biobase \
bioconductor-affyplm \
bioconductor-simpleaffy \
bioconductor-limma \
bioconductor-sva \
bioconductor-vsn \
bioconductor-gcrma \
bioconductor-impute \
'icu=58.*' \
lxml \
natsort \
mlxtend

```
3. Install CRAN Packages (not available via Conda)

```R
options(repos=structure(c(CRAN="https://cloud.r-project.org/")))
install.packages("bapred")
```
4. Install Bioconductor Packages (not available via Conda)

```R
source("https://bioconductor.org/biocLite.R")
biocLite("hgu133plus2.db", suppressUpdates=TRUE)
biocLite("hgu133plus2cdf", suppressUpdates=TRUE)
biocLite("hgu133plus2probe", suppressUpdates=TRUE)
biocLite("pvca", suppressUpdates=TRUE)
biocLite("LiblineaR", suppressUpdates=TRUE)
biocLite("JADE", suppressUpdates=TRUE)
```
5. Install Brainarray Custom Microarray Annotation DBs and CDFs

```R
library(devtools)
install_url("http://brainarray.mbni.med.umich.edu/bioc/src/contrib/hgu133plus2hsentrezg.db_22.0.0.tar.gz")
install_url("http://brainarray.mbni.med.umich.edu/bioc/src/contrib/hgu133plus2hsentrezgcdf_22.0.0.tar.gz")
install_url("http://brainarray.mbni.med.umich.edu/bioc/src/contrib/hgu133plus2hsentrezgprobe_22.0.0.tar.gz")
```
