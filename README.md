# NCI LHC NSCLC Research and Data Analysis Repository

## Environment and Analysis Pipeline Setup

1. Install System Dependencies

sudo dnf install libxml-devel libxml2-devel

2. Install Anaconda

https://www.anaconda.com/download/

https://docs.anaconda.com/anaconda/install/linux

2. Setup Conda Environment and Install R, Rpy2, Bioconductor, Limma, etc.

```bash
conda config --add channels conda-forge
conda config --add channels bioconda
conda install -v -y r-essentials rpy2 bioconductor-biocinstaller bioconductor-biobase bioconductor-simpleaffy bioconductor-limma
conda install -v -y -c conda-forge 'icu=58.*' lxml
```

3. Install HG-U133Plus2 Bioconductor DB

```R
source("https://bioconductor.org/biocLite.R")
biocLite("hgu133plus2.db")
```
