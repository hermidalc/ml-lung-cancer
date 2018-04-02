# NCI LHC NSCLC Research and Data Analysis Repository

## Environment and Analysis Pipeline Setup

1. Install System Dependencies

```bash
sudo dnf install -y libxml-devel libxml2-devel mesa-libGL-devel mesa-libGLU-devel
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
r-rgl \
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

3. Setup for rJava (used by some R packages)

In .bashrc/.bash_profile:
```bash
JAVA_HOME=/usr/lib/jvm/java
export JAVA_HOME

LD_LIBRARY_PATH=/usr/lib/jvm/java/jre/lib/amd64:/usr/lib/jvm/java/jre/lib/amd64/server
export LD_LIBRARY_PATH
```
Then run:
```bash
source .bashrc
R CMD javareconf
```

4. Install CRAN Packages (not available via Conda)

```R
options(repos=structure(c(CRAN="https://cloud.r-project.org/")))
install.packages("rJava", type="source")
install.packages("FSelector")
install.packages("bapred")
```

5. Install Bioconductor Packages (not available via Conda)

```R
source("https://bioconductor.org/biocLite.R")
biocLite("hgu133plus2.db", suppressUpdates=TRUE)
biocLite("hgu133plus2cdf", suppressUpdates=TRUE)
biocLite("hgu133plus2probe", suppressUpdates=TRUE)
biocLite("pvca", suppressUpdates=TRUE)
biocLite("JADE", suppressUpdates=TRUE)
```

6. Install Brainarray Custom Microarray Annotation DBs and CDFs

```R
library(devtools)
install_url("http://brainarray.mbni.med.umich.edu/bioc/src/contrib/hgu133plus2hsentrezg.db_22.0.0.tar.gz")
install_url("http://brainarray.mbni.med.umich.edu/bioc/src/contrib/hgu133plus2hsentrezgcdf_22.0.0.tar.gz")
install_url("http://brainarray.mbni.med.umich.edu/bioc/src/contrib/hgu133plus2hsentrezgprobe_22.0.0.tar.gz")
```
