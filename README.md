## HEALING PRODUCT OF GAUSSIAN PROCESS EXPERTS - ICML 2020

In this repository can be found implementation of the aggregations and weighting approaches for Gaussian Process PoEs proposed in our paper [1], along with several previously proposed PoEs.

## Aggregation Methods:

* PoE
* gPoE
* BCM
* rBCM
* Barycenter

Along with other baselines:

* Full GP
* Linear regression

## Weighting Methods
* Differential Entropy
* Softmax-Variance
* Uniform
* No-weights

## How to Run Experiments:
Please move into Code/bayesian_benchmarks_modular and run:
* python -m pytest bayesian_benchmarks/scripts/run_all_pytest.py -n X

where X is the number of experiments ran in parallel

Results can be viewed in Code/bayesian_benchmarks_modular/bayesian_benchmarks/results/view_results.ipynb

## Dependencies:
* Tensorflow 2.0
* GPflow 2.0.1
* Numpy 0.18.1
* Pandas 0.25.1
* pytest-xdist
* tqdm 4.32.1
* sklearn 0.21.2


## Our Paper
This repository complements our paper:

[1] Healing Products of Gaussian Process Experts, Samuel Cohen, Rendani Mbuvha, Tshilidzi Marwala, Marc Deisenroth
