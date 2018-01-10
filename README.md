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
r-lintr \
r-pamr \
r-minqa \
r-nloptr \
r-rcppeigen \
r-lme4 \
r-corpcor \
rpy2 \
libiconv
conda install -v -y \
bioconductor-biocinstaller \
bioconductor-biobase \
bioconductor-affyplm \
bioconductor-simpleaffy \
bioconductor-limma \
bioconductor-sva \
bioconductor-vsn \
bioconductor-gcrma
conda install -v -y -c conda-forge 'icu=58.*' lxml natsort mlxtend

```
3. Install CRAN Packages (not available via Conda)

```R
options(repos=structure(c(CRAN="https://cloud.r-project.org/")))
install.packages("bapred")
```
4. Install Bioconductor Packages (not available via Conda)

```R
source("https://bioconductor.org/biocLite.R")
biocLite("hgu133plus2.db")
biocLite("hgu133plus2cdf")
biocLite("hgu133plus2probe")
biocLite("pvca")
biocLite("LiblineaR")
biocLite("JADE")
```
