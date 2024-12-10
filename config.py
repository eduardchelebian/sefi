"""
Configuration parameters for SEFI (SEgmentation-Free Integration).
"""

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