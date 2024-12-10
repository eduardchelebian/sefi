import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import linkage, fcluster

from config import (
    KMEANS_PARAMS,
    HIERARCHICAL_PARAMS
)

def kmeans(merged):
    """
    Perform MiniBatch k-means clustering on the merged data, excluding spatial coordinates.
    
    Uses MiniBatchKMeans for memory efficiency when dealing with large datasets.
    The number of clusters is set to the number of unique genes in the dataset.
    
    Parameters
    ----------
    merged : dict
        Dictionary containing:
        - 'data': Combined gene expression and PCA-reduced embeddings DataFrame
        - 'n_genes': Number of genes (used as n_clusters)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'labels': Cluster assignments for each data point
        - 'centers': Cluster centroids in feature space
    """
    print("Performing MiniBatch k-means clustering...")
    
    # Skip spatial coordinates (x, y) and sample index
    feature_columns = merged['data'].columns[3:]
    clustering_data = merged['data'][feature_columns]
    
    # Initialize and fit MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=merged['n_genes'],
        batch_size=KMEANS_PARAMS['batch_size'],
        max_iter=KMEANS_PARAMS['max_iter'],
        random_state=KMEANS_PARAMS['random_state']
    )
    labels = kmeans.fit_predict(clustering_data)
    
    # Calculate cluster centers manually for consistency
    centers = np.zeros((merged['n_genes'], len(feature_columns)))
    for i in range(merged['n_genes']):
        mask = labels == i
        centers[i] = clustering_data[mask].mean(axis=0)
    
    return {
        'labels': labels,
        'centers': centers
    }

def find_optimal_threshold(linkage_matrix):
    """
    Find optimal threshold for cutting dendrogram using the elbow method.
    
    Uses a combination of techniques to find a good cutting point:
    1. Looks for significant jumps in merge distances
    2. Falls back to percentile-based threshold if no clear jumps found
    
    Parameters
    ----------
    linkage_matrix : np.ndarray
        The linkage matrix from hierarchical clustering
        
    Returns
    -------
    float
        Optimal distance threshold for cutting the dendrogram
    """
    # Get sorted merge distances
    distances = np.sort(linkage_matrix[:, 2])
    
    # Calculate distance ratios between consecutive merges
    diffs = np.diff(distances)
    ratios = diffs[1:] / diffs[:-1]
    
    # Look for significant jumps in the top fraction of distances
    start_idx = int(HIERARCHICAL_PARAMS['top_distances_fraction'] * len(distances))
    significant_jumps = ratios[start_idx:] > HIERARCHICAL_PARAMS['significant_jump_ratio']
    
    if np.any(significant_jumps):
        # Use midpoint between merges where significant jump occurs
        jump_idx = np.where(significant_jumps)[0][0] + start_idx
        threshold = (distances[jump_idx] + distances[jump_idx + 1]) / 2
    else:
        # Fall back to percentile-based threshold
        threshold = np.percentile(distances, HIERARCHICAL_PARAMS['percentile_threshold'])
    
    return threshold

def hierarchical(kmeans_clusters):
    """
    Perform hierarchical clustering on k-means cluster centers.
    
    This implements a two-stage clustering approach:
    1. Use k-means centers as input to reduce data dimensionality
    2. Apply hierarchical clustering to find natural groupings
    
    Parameters
    ----------
    kmeans_clusters : dict
        Output from kmeans() containing:
        - 'centers': Cluster centers from k-means
        - 'labels': Original k-means cluster assignments
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'merged_labels': Final cluster assignments for each point
        - 'linkage_matrix': Hierarchical clustering linkage matrix
        - 'distance_threshold': Optimal cutting threshold used
    """
    print("Performing hierarchical clustering...")
    
    # Compute linkage matrix using configured method and metric
    linkage_matrix = linkage(
        kmeans_clusters['centers'],
        method=HIERARCHICAL_PARAMS['linkage_method'],
        metric=HIERARCHICAL_PARAMS['linkage_metric']
    )
    
    # Find optimal cutting threshold
    distance_threshold = find_optimal_threshold(linkage_matrix)
    
    # Get cluster assignments
    merged_labels = fcluster(linkage_matrix, 
                           distance_threshold, 
                           criterion='distance')
    
    # Map cluster assignments back to original points
    final_labels = merged_labels[kmeans_clusters['labels']]
    
    # Print clustering statistics
    n_merged_clusters = len(np.unique(final_labels))
    print(f"    > Number of merged clusters: {n_merged_clusters}")
    
    return {
        'merged_labels': final_labels,
        'linkage_matrix': linkage_matrix,
        'distance_threshold': distance_threshold
    }
