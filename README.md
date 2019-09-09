# ML Prediction in Lung Cancer

## Environment and Analysis Pipeline Setup

1.  Install latest Miniconda3

<https://docs.conda.io/en/latest/miniconda.html>

```bash
bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
```

Restart shell and then:

```bash
conda config --set auto_activate_base false
conda config --add channels conda-forge
conda config --set channel_priority strict
conda update --all
```

2.  Create `ml-bio-sklearn` environment

```bash
conda create --name ml-bio-sklearn --yes
conda activate ml-bio-sklearn
conda config --env --add channels bioconda
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
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
bioconductor-geoquery \
bioconductor-gsva \
bioconductor-limma \
bioconductor-pvca \
bioconductor-singscore \
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
pillow \
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

3.  Switching between different BLAS implementations

By default conda-forge installed numpy, scipy, scikit-learn, numexpr packages
are built against OpenBLAS, but for your particular CPU architecture others
might have better performance:

<https://conda-forge.org/docs/maintainer/knowledge_base.html#switching-blas-implementation>

```bash
conda install "libblas=*=*mkl"
conda install "libblas=*=*openblas"
conda install "libblas=*=*blis"
conda install "libblas=*=*netlib"
```

If you switch from `openblas` then you also need to create a `pinned` file in
your environment `conda-meta` directory, e.g.:

```bash
echo 'libblas[build=*mkl]' > <path to miniconda3>/envs/ml-bio-sklearn/conda-meta/pinned
```

4.  Install CRAN and Bioconductor packages (not available/working via Conda)

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

5.  Install Brainarray custom microarray annotations and CDFs

```R
library(devtools)
install_url("http://mbni.org/customcdf/23.0.0/entrezg.download/hgu133plus2hsentrezg.db_23.0.0.tar.gz")
install_url("http://mbni.org/customcdf/23.0.0/entrezg.download/hgu133plus2hsentrezgcdf_23.0.0.tar.gz")
install_url("http://mbni.org/customcdf/23.0.0/entrezg.download/hgu133plus2hsentrezgprobe_23.0.0.tar.gz")
```

6.  Create `sksurv` environment and scikit-survival installation

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
