import os
import pandas as pd
from preprocessing import load_data
from clustering import kmeans, hierarchical

def process_and_cluster(st_spots_path, morphology_embeddings_path, output_path):
    """
    Main function to run the SEFI pipeline.
    
    This function orchestrates the complete SEFI workflow:
    1. Loads and processes spatial transcriptomics and morphology data
    2. Performs initial k-means clustering for dimensionality reduction
    3. Applies hierarchical merging for final cluster assignments
    4. Saves results to specified output path
    
    Parameters
    ----------
    st_spots_path : str or list
        Path(s) to CSV file(s) containing spatial transcriptomics spots.
        Each file should have columns: x, y, Gene
    morphology_embeddings_path : str or list
        Path(s) to CSV file(s) containing morphological embeddings.
        Each file should have columns: x, y, Feature_1, Feature_2, ...
    output_path : str
        Path where to save the clustering results CSV file
        
    Returns
    -------
    tuple
        - numpy.ndarray: Final cluster assignments for each point
        - dict: Clustering metadata containing:
            - 'linkage_matrix': Hierarchical clustering linkage matrix
            - 'distance_threshold': Optimal cutting threshold used
    """
    # Load and process data
    print("Loading and processing data...")
    merged = load_data(st_spots_path, morphology_embeddings_path)
    
    # Perform k-means clustering
    kmeans_results = kmeans(merged)
    
    # Perform hierarchical clustering
    clustering_results = hierarchical(kmeans_results)
    
    # Save results
    print("Saving results...")
    results_df = pd.DataFrame({
        'x': merged['data']['x'],
        'y': merged['data']['y'],
        'kmeans_cluster': kmeans_results['labels'],
        'hierarchical_cluster': clustering_results['merged_labels']
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save cluster assignments
    results_df.to_csv(output_path, index=False)
    
    return clustering_results['merged_labels'], {
        'linkage_matrix': clustering_results['linkage_matrix'],
        'distance_threshold': clustering_results['distance_threshold']
    }




# Example usage with multiple samples
points_paths = [
    "path/to/points1.csv", 
    "path/to/points2.csv",
]

embeddings_paths = [
    "path/to/embeddings1.csv", 
    "path/to/embeddings2.csv",
]

output_path = "path/to/results/combined_clusters.csv"

# Run the SEFI pipeline
merged = process_and_cluster(points_paths, embeddings_paths, output_path)