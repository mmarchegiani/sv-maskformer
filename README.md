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
TBC.

## Training
Example of how to run a training:
```bash
python src/scripts/train/train_taus_integrated_noise.py --root /eos/user/m/mmarcheg/ml4reco/data/train
```

## Evaluation
TBC.

## Validation plots
TBC.


