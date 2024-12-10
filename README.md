# SEFI: SEgmentation-Free Integration of Spatial Transcriptomics and Morphological Features

## Overview

SEFI (SEgmentation-Free Integration) is a Python package for combining spatial transcriptomics data and morphological features. It uses a two-step clustering approach with k-means and hierarchical clustering.

## Installation

We recommend creating a conda environment for running the whole SEFI pipeline:
```shell
conda env create -n sefi -f environment.yml
```

To activate the environment:
```shell
conda activate sefi
```

## Usage

To run the SEFI pipeline, the user needs:
1. A csv file containing the ST spots data with columns: x, y, Gene
2. A csv file containing the morphological features extracted from patches centered on the ST spots with columns: x, y, Feature 1, Feature 2, ...

First, modify the `config.py` file to set the parameters for the processing[^1] and clustering steps:
```python
# Points2Regions processing parameters
POINTS2REGIONS_PARAMS = {
    'sigma': 16,              # Bandwidth parameter for kernel density estimation
    'min_genes_per_bin': 5,   # Minimum number of genes required per spatial bin
}

# MiniBatch k-means clustering parameters
KMEANS_PARAMS = {
    'batch_size': 2560,    # Large batch size for better stability
    'max_iter': 100,       # Maximum iterations for convergence
    'random_state': 0      # For reproducibility
}

# Hierarchical clustering parameters
HIERARCHICAL_PARAMS = {
    'percentile_threshold': 75,      # Default percentile if no clear jump found
    'significant_jump_ratio': 1.5,   # Minimum ratio to consider a jump significant
    'top_distances_fraction': 0.75,  # Consider only top fraction of distances
    'linkage_method': 'ward',        # Method for hierarchical clustering
    'linkage_metric': 'euclidean'    # Distance metric for hierarchical clustering
} 
``` 

[^1]: Processing of imaging-based spatial transcriptomics data is performed using Points2Regions. Refer to the [Points2Regions repository](https://github.com/wahlby-lab/Points2Regions) for more information.

Then, specify the paths to the ST spots `path/to/points.csv` and morphology embeddings `path/to/embeddings.csv` files in the `main.py` file, and run:
```shell
python main.py
```
The output `path/to/results/combined_clusters.csv` will be a csv file containing the cluster assignments for each spot based on the gene expression and morphological features.

## Reference
For more information and a specific example of SEFI applied to retinal MERFISH data, please refer to:

```

``` 