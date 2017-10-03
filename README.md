# NCI LHC NSCLC Research and Data Analysis Repository

## Environment and Analysis Pipeline Setup

1. Install System Dependencies

sudo dnf install libxml-devel libxml2-devel libxml++-devel

2. Install Anaconda

https://www.anaconda.com/download/

https://docs.anaconda.com/anaconda/install/linux

2. Setup Conda Environment and Install R, Rpy2, Bioconductor, Limma, etc.

```bash
conda install -v -y r-essentials rpy2
conda config --add channels bioconda
conda install -v -y -c bioconda bioconductor-biocinstaller bioconductor-biobase bioconductor-simpleaffy bioconductor-limma
```

3.

conda config --add channels conda-forge
