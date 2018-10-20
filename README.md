# ML Prediction in Lung Cancer

## Environment and Analysis Pipeline Setup

1. Install System Dependencies

```bash
sudo dnf install -y libxml-devel libxml2-devel mesa-libGL-devel mesa-libGLU-devel gcc-gfortran
```

2. Install Anaconda3 5.2.0

https://www.anaconda.com/download/
https://docs.anaconda.com/anaconda/install/linux

```bash
bash ~/Downloads/Anaconda3-5.2.0-Linux-x86_64.sh
```

3. Install Conda Packages

```bash
conda install \
r-base=3.5.1 \
r-essentials \
r-devtools \
r-argparse \
r-lintr \
r-lme4 \
r-minqa \
r-nloptr \
r-rcppeigen \
r-rgl \
r-rjava \
r-xml \
autopep8 \
libgfortran \
libiconv \
lxml \
natsort \
rpy2 \
seaborn
conda install -c conda-forge python-language-server mlxtend=0.13.0
conda install -c sebp scikit-survival
```

4. Install CRAN and Bioconductor Packages (not available/working via Conda)

```R
options(repos=structure(c(CRAN="https://cloud.r-project.org/")))
source("https://bioconductor.org/biocLite.R")
install.packages("corpcor")
install.packages("pamr")
install.packages("mlr")
install.packages("statmod")
install.packages("FSelector")
install.packages("FSelectorRcpp")
install.packages("Biocomb")
biocLite("Biobase", suppressUpdates=TRUE)
biocLite("affyio", suppressUpdates=TRUE)
biocLite("affy", suppressUpdates=TRUE)
biocLite("gcrma", suppressUpdates=TRUE)
biocLite("affyPLM", suppressUpdates=TRUE)
biocLite("impute", suppressUpdates=TRUE)
biocLite("limma", suppressUpdates=TRUE)
biocLite("edgeR", suppressUpdates=TRUE)
biocLite("sva", suppressUpdates=TRUE)
biocLite("pvca", suppressUpdates=TRUE)
biocLite("JADE", suppressUpdates=TRUE)
biocLite("MLSeq", suppressUpdates=TRUE)
biocLite("hgu133plus2.db", suppressUpdates=TRUE)
biocLite("hgu133plus2cdf", suppressUpdates=TRUE)
biocLite("hgu133plus2probe", suppressUpdates=TRUE)
biocLite("GO.db", suppressUpdates=TRUE)
install.packages("bapred")
install.packages("WGCNA")
```

5. Install Brainarray Custom Microarray Annotation DBs and CDFs

```R
library(devtools)
install_url("http://brainarray.mbni.med.umich.edu/bioc/src/contrib/hgu133plus2hsentrezg.db_22.0.0.tar.gz")
install_url("http://brainarray.mbni.med.umich.edu/bioc/src/contrib/hgu133plus2hsentrezgcdf_22.0.0.tar.gz")
install_url("http://brainarray.mbni.med.umich.edu/bioc/src/contrib/hgu133plus2hsentrezgprobe_22.0.0.tar.gz")
```
