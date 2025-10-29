# HGCAL Offline Reconstruction

Repository for HGCAL offline reconstruction using machine learning techniques. This repository contains code and tools for training and evaluating machine learning models on HGCAL simulated datasets.
A set of scripts to analyze the output and plot the clustering space is also included.

## Installation
The following installation instructions are taken from the README of the [pytorch_cmspepr](https://github.com/cms-pepr/pytorch_cmspepr) repository to install an environment based on CUDA 11.7 and Pytorch 2.0.

Pull and run docker container:
```
singularity pull docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
singularity run --nv pytorch_2.0.0-cuda11.7-cudnn8-devel.sif
```

Create a Python virtual environment within the docker container, activate the environment and install the required packages:
```
export PYTHONPATH="/opt/conda/lib/python3.10/site-packages"
python -m venv env
source env/bin/activate
pip install tqdm
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html # Make sure to pick the right torch and CUDA versions here
git clone git@github.com:cms-pepr/pytorch_cmspepr.git
cd pytorch_cmspepr
pip install -e .
```

Now install the code of this repository with:
```
cd ..
pip install -e .
```

Now you are all setup and ready to run the first training.

## Training
Example of how to run a training:
```bash
python src/scripts/train/train_taus_integrated_noise.py --root /eos/user/m/mmarcheg/ml4reco/data/taus_2025
```

## Evaluation
TBC.

## Validation plots
TBC.


## Installation with Python 3.11 (experimental)
These installation instructions work on the CERN `lxplus` cluster, or alternatively on any cluster where the `/cvmfs` partition is mounted.
In order to get a Python environment with Pytorch working on Python 3.11, activate the `cmsml:3.11` singularity image:
```bash
alias cmsml311='apptainer shell -B /afs -B /eos -B /eos/user/m/mmarcheg -B /eos/home-m/mmarcheg -B /afs/cern.ch/work/m/mmarcheg \
        -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} \
        --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.11 --bind /usr/local/cuda-12.4:/usr/local/cuda-12.4'
```
Create a Python virtual environment within the singularity container and activate the environment:
```
python -m venv --system-site-packages myenv
source myenv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
```

Install the package in editable mode (`-e`), then install the optional libraries for `pytorch_geometric` with python wheels:
```
pip install -e .
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

Install `pytorch_cmspepr` to install KNN graph operations in Pytorch:
```bash
git clone git@github.com:tklijnsma/pytorch_cmspepr.git
cd pytorch_cmspepr
pip install -e .
```
