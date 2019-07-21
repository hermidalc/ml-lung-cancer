# ML Prediction in Lung Cancer

## Environment and Analysis Pipeline Setup

1.  Install latest Miniconda3

<https://docs.conda.io/en/latest/miniconda.html>

```bash
bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
conda config --add channels conda-forge
conda update --all
conda config --set auto_activate_base false
```

2.  Create environment

```bash
conda create --name ml-bio-sklearn --yes
conda activate ml-bio-sklearn
conda config --env --add channels bioconda
conda config --env --add channels conda-forge
```

3.  Install conda packages

```bash
conda install \
autopep8 \
bioconductor-affy \
bioconductor-affyio \
bioconductor-affyplm \
bioconductor-biobase \
bioconductor-biocversion \
bioconductor-edger \
bioconductor-impute \
bioconductor-gcrma \
bioconductor-limma \
bioconductor-pvca \
bioconductor-sva \
cython \
ipykernel \
jedi \
joblib \
libgfortran \
libiconv \
lxml \
libxml2 \
matplotlib \
mlxtend \
natsort \
python-language-server \
r-base \
r-essentials \
r-devtools \
r-argparse \
r-arules \
r-bitops \
r-catools \
r-clue \
r-corpcor \
r-e1071 \
r-fnn \
r-fselector \
r-gdata \
r-gplots \
r-gtools \
r-lintr \
r-lme4 \
r-minqa \
r-mnormt \
r-nloptr \
r-pamr \
r-proc \
r-rcpparmadillo \
r-rcppeigen \
r-rematch2 \
r-rgl \
r-rjava \
r-rocr \
r-statmod \
r-styler \
r-tinytex \
r-wgcna \
r-xml \
rpy2 \
scikit-learn \
seaborn
```

4.  Switching between different BLAS implementations

By default conda-forge installed numpy, scipy, scikit-learn, numexpr packages
are built against OpenBLAS, but for your particular architecture others might
have better performance:

<https://conda-forge.org/docs/maintainer/knowledge_base.html#switching-blas-implementation>

```bash
conda install "libblas=*=*mkl"
conda install "libblas=*=*openblas"
conda install "libblas=*=*blis"
conda install "libblas=*=*netlib"
```

5.  Install CRAN and Bioconductor packages (not available/working via Conda)

```R
options(repos=structure(c(CRAN="https://cloud.r-project.org/")))
install.packages("Biocomb")
install.packages("bapred")
install.packages("languageserver")
library(BiocManager)
install("JADE", update=FALSE)
install("hgu133plus2.db", update=FALSE)
install("hgu133plus2cdf", update=FALSE)
install("hgu133plus2probe", update=FALSE)
install("GO.db", update=FALSE)
```

6.  Install Brainarray Custom Microarray Annotation DBs and CDFs

```R
library(devtools)
install_url("http://mbni.org/customcdf/23.0.0/entrezg.download/hgu133plus2hsentrezg.db_23.0.0.tar.gz")
install_url("http://mbni.org/customcdf/23.0.0/entrezg.download/hgu133plus2hsentrezgcdf_23.0.0.tar.gz")
install_url("http://mbni.org/customcdf/23.0.0/entrezg.download/hgu133plus2hsentrezgprobe_23.0.0.tar.gz")
```

7.  Create scikit-survival Environment and Install Package

```bash
conda create -n sksurv -c sebp python=3 scikit-survival
conda activate sksurv
```

Or from source if needing latest dev version:

```bash
cd ~/projects/github/hermidalc/scikit-survival
git fetch upstream
git pull upstream master
git push
python ci/list-requirements.py requirements/dev.txt > /tmp/requirements.txt
conda create -n sksurv -c sebp python=3 --file /tmp/requirements.txt
conda activate sksurv
git submodule update --init --recursive
python setup.py install
pytest tests/
```

## Old Anaconda3 5.2.0 Instructions

1.  Install Anaconda3 5.2.0

<https://www.anaconda.com/download/>
<https://docs.anaconda.com/anaconda/install/linux>

```bash
bash ~/Downloads/Anaconda3-5.2.0-Linux-x86_64.sh
conda update -n base conda
conda init bash
```

2.  Install Conda Packages

```bash
conda install \
r-base \
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
cython \
ipykernel \
jedi \
joblib \
libgfortran \
libiconv \
lxml \
libxml2 \
natsort \
rpy2 \
scikit-learn=0.19.2 \
seaborn
conda config --append channels conda-forge
conda config --append channels bioconda
conda install mlxtend=0.16.0 python-language-server
```

3.  Install CRAN and Bioconductor Packages (not available/working via Conda)

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
install.packages("languageserver")
library(devtools)
install_version("mvtnorm", version="1.0-8")
install.packages("WGCNA")
```

4.  Install Brainarray Custom Microarray Annotation DBs and CDFs

```R
library(devtools)
install_url("http://mbni.org/customcdf/22.0.0/entrezg.download/hgu133plus2hsentrezg.db_22.0.0.tar.gz")
install_url("http://mbni.org/customcdf/22.0.0/entrezg.download/hgu133plus2hsentrezgcdf_22.0.0.tar.gz")
install_url("http://mbni.org/customcdf/22.0.0/entrezg.download/hgu133plus2hsentrezgprobe_22.0.0.tar.gz")
```

5.  Create scikit-survival Environment and Install Package

```bash
conda create -n sksurv -c sebp python=3 scikit-survival
conda activate sksurv
```

Or from source if needing latest dev version:

```bash
cd ~/projects/github/hermidalc/scikit-survival
git fetch upstream
git pull upstream master
git push
python ci/list-requirements.py requirements/dev.txt > /tmp/requirements.txt
conda create -n sksurv -c sebp python=3 --file /tmp/requirements.txt
conda activate sksurv
git submodule update --init --recursive
python setup.py install
pytest tests/
```
