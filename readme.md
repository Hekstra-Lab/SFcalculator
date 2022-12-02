## SFcalculator

This is the code repo related to paper [Towards automated crystallographic structure refinement with a differentiable pipeline](https://www.mlsb.io/papers_2022/Towards_automated_crystallographic_structure_refinement_with_a_differentiable_pipeline.pdf) on [Machine Learning in Structural Biology Workshop](https://www.mlsb.io/) at NeurIPS 2022.

Structure Factor Calculator implemented in `tensorflow2`, `pytorch` and `jax`. 

A differentiable pipeline connecting the protein atomic models and experimental structure factors, featuring a differentiable bulk solvent correction.

The symmetry-related nitty-gritty in both real space and reciprocal space are included.

Provided in three popular deep learning frameworks.

### Authors

Minhuan Li, minhuanli@g.harvard.edu

Doeke R. Hekstra, doeke_hekstra@harvard.edu

### Installation

#### Tensorflow 2 verision

1. Create a python environment with package manager you like ([mambaforge](https://github.com/mamba-org/mamba) recommended).

2. Install [Tensorflow2](https://www.tensorflow.org/install)

3. Clone this repository, then
    ```bash
    cd tensorflow2

    pip install .
    ```

#### Pytorch verision

1. Create a python environment with package manager you like ([mambaforge](https://github.com/mamba-org/mamba) recommended).

2. Install [Pytorch](https://pytorch.org/get-started/locally/)

3. Clone this repository, then
    ```bash
    cd pytorch

    pip install .
    ```

#### Jax verision

1. Create a python environment with package manager you like ([mambaforge](https://github.com/mamba-org/mamba) recommended).

2. Install [Jax](https://github.com/google/jax#installation)

3. Clone this repository, then
    ```bash
    cd jax

    pip install .
    ```
