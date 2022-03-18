# Deep Learning Utilities for Pathology: PyTorch-lightning implementation of multiple instance learning models

==== WORK IN PROGRESS ====

- Add bash scripts to run compilation of features from VISSL
- Add bash scripts to run training of all classifiers
- Add config files
- Improve documentation

==========================

dlup_lightning_mil offers a set MIL classification models built with PyTorch-lightning using utilities from DLUP
to ease the process of running Deep Learning classification/regression models on Whole Slide Images.

Tile supervision methods can be run with just this repository.

MIL methods require feature extraction using [hissl](https://github.com/nki-ai/hissl).

## Reproduce DeepSMILE (to be added)

The scripts in `~/dlup-lightning-mil/tools/reproduce_deepsmile` will reproduce the models
as described in DeepSMILE. It is expected you run `~/hissl/tools/reproduce_deepsmile/reproduce_deepsmile.sh` from [hissl](https://github.com/nki-ai/hissl) on
the same machine and hosting the repository in the same parent directory.

Running `~/dlup-lightning-mil/tools/reproduce_deepsmile$ bash reproduce_deepsmile.sh`
will run the following scripts:

- `0_check_singularity/check_singularity.sh`
  - Checks if a singularity image is available to run all following scripts
- `1_create_splits_for_tcga_crck`
  - Creates 5-fold train-val splits for TCGA-CRCk from the defined train set
  - Creates subsets of the training splits for low labelled data regime experiments
- `2_compile_features/compile_h5_features.sh`
  - Reorganizes the saved features from `hissl` to allow for easier and quicker loading
- `3_train_models/*`
  - Runs
    - Every combination of extractor and classifier
    - On both datasets
    - For all available labels
    - For all defined fractions of training data
  - Saves to `~/dlup-lightning-mil/logs`
    - model checkpoints
    - training loss
    - validation and test AUCs
    - validation and test predictions and labels
    - MIL attention or predicted score for each tile

## Upcoming features
- Analysis of results of runs
- Visualization of high- and low-attention tiles
- Plotting of graphs
- More MIL models & end-to-end MIL training


