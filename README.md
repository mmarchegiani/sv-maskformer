# HGCAL Offline Reconstruction

Repository for HGCAL offline reconstruction using machine learning techniques. This repository contains code and tools for training and evaluating machine learning models on HGCAL simulated datasets.
A set of scripts to analyze the output and plot the clustering space is also included.

## GitLab configuration
In order to link your commits to your CERN account identity, you need to make sure that the parameters `user.name` and `user.email` are set in your Git configuration.
To show your `user.name` and `user.email`, you can run the following commands:

```bash
git config user.name
git config user.email
```

To change the `user.name` and `user.email` locally in your repository, you can run the following commands:

```bash
git config user.name "Your Name"
git config user.email "your.email@cern.ch"
```

In this way, the commits pushed to your repository will be linked to your CERN account identity.

## Installation
Activate the `cmsml:3.11` singularity image:
```bash
alias cmsml311='apptainer shell -B /afs -B /eos -B /eos/user/m/mmarcheg -B /eos/home-m/mmarcheg -B /afs/cern.ch/work/m/mmarcheg \
        -B /etc/sysconfig/ngbauth-submit -B ${XDG_RUNTIME_DIR} \
        --env KRB5CCNAME="FILE:${XDG_RUNTIME_DIR}/krb5cc" --nv /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.11 --bind /usr/local/cuda-12.4:/usr/local/cuda-12.4'
```
Create a Python virtual environment within the singularity container and activate the environment:
```
python -m venv --system-site-packages myenv
source myenv/bin/activate
```

Install the package in editable mode (`-e`), then install the optional libraries for `pytorch_geometric` with python wheels:
```
pip install -e .
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

## Training
Example of how to run a training:
```bash
python src/scripts/train/train_taus_integrated_noise.py --root /eos/user/m/mmarcheg/ml4reco/data/taus_2025
```

## Evaluation
TBC.

## Validation plots
TBC.


