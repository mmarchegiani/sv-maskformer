# HGCAL Offline Reconstruction

Repository for HGCAL offline reconstruction using machine learning techniques. This repository contains code and tools for training and evaluating machine learning models on HGCAL simulated datasets.
A set of scripts to analyze the output and plot the clustering space is also included.

## Installation

### Quick Start (Recommended)

The package can be installed directly via pip:

```bash
# For standard installation
pip install -e .

# For development installation with additional tools
pip install -e ".[dev]"
```

### Detailed Installation with Docker/Singularity

The following installation instructions are taken from the README of the [pytorch_cmspepr](https://github.com/cms-pepr/pytorch_cmspepr) repository to install an environment based on CUDA 11.7 and Pytorch 2.0.

Pull and run docker container:
```bash
singularity pull docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
singularity run --nv pytorch_2.0.0-cuda11.7-cudnn8-devel.sif
```

Create a Python virtual environment within the docker container, activate the environment and install the package:
```bash
export PYTHONPATH="/opt/conda/lib/python3.10/site-packages"
python -m venv env
source env/bin/activate

# Install this package in editable mode
pip install -e .

# Install PyTorch Geometric extensions
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Optionally install pytorch_cmspepr
git clone git@github.com:cms-pepr/pytorch_cmspepr.git
cd pytorch_cmspepr
pip install -e .
cd ..
```

Now you are all setup and ready to run the first training.

## GNN training
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

## Installation for maskformer training
Follow the installation instructions of the [hepattn](https://github.com/samvanstroud/hepattn/tree/main) repository. In particular, follow the [Setup](https://github.com/samvanstroud/hepattn/tree/main?tab=readme-ov-file#%EF%B8%8F-setup) of the apptainer shell.
To start a shell within the apptainer, run the following commands:
```bash
source setup_hepattn.sh
```

Now you can run using the environment of the `hepattn` repository alongside the custom code specific for the HGCAL in the `src` folder.

To run the model training:

```bash
python src/scripts/train/train_maskformer.py fit --config configs/base.yaml
```

To skip sanity checking of the validation dataloader and test with a single worker, run the training with these options:
```bash
python src/scripts/train/train_maskformer.py fit --config configs/base.yaml --data.num_workers 0 --trainer.num_sanity_val_steps 0
```

Note that the parameters in `configs/base.yaml` are a template and need to be tuned for the HGCAL reconstruction case.
