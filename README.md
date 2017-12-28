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
conda install -v -y r-essentials r-lintr rpy2 r-pamr
conda install -v -y bioconductor-biocinstaller bioconductor-biobase bioconductor-simpleaffy bioconductor-limma bioconductor-sva
conda install -v -y -c conda-forge 'icu=58.*' lxml natsort mlxtend

```

3. Install HG-U133Plus2 Bioconductor DB

```R
source("https://bioconductor.org/biocLite.R")
biocLite("hgu133plus2.db")

```
