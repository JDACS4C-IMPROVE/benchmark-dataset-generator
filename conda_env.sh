#!/bin/bash --login

set -e

# conda create -n imp_data python=3.7 pip --yes

conda install -c anaconda scikit-learn --yes  # includes numpy, scipy, joblib
conda install -c anaconda pandas --yes

conda install -c anaconda lightgbm --yes

conda install -c conda-forge ipdb --yes
conda install -c conda-forge python-lsp-server --yes

conda install -c anaconda pyarrow --yes
