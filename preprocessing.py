import pandas as pd
import numpy as np
from Points2Regions import points2regions
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import scanpy as sc
import anndata as ad
import scipy

from config import POINTS2REGIONS_PARAMS

def join_samples(points_paths: list, embeddings_paths: list) -> tuple:
    """
    Join multiple samples into single dataframes with offset coordinates.
    
    Combines multiple spatial transcriptomics samples by adding offsets to their
    x-coordinates to prevent overlap.
    
    Parameters
    ----------
    points_paths : list
        List of paths to CSV files containing gene positions and labels.
        Each file should have columns: x, y, Gene
    embeddings_paths : list
        List of paths to CSV files containing morphological embeddings.
        Each file should have columns: x, y, Feature_1, Feature_2, ...
        
    Returns
    -------
    tuple
        - Combined points DataFrame with columns: x, y, Gene, sample_idx
        - Combined embeddings DataFrame with columns: x, y, Feature_*, sample_idx
    """
    all_points = []
    all_embeddings = []
    offset = 0
    
    for points_path, emb_path in zip(points_paths, embeddings_paths):
        # Load data files
        points = pd.read_csv(points_path)
        embeddings = pd.read_csv(emb_path)
        
        # Apply x-coordinate offset
        points['x'] += offset
        embeddings['x'] += offset
        
        # Add sample index for tracking
        points['sample_idx'] = len(all_points)
        embeddings['sample_idx'] = len(all_embeddings)
        
        # Calculate next offset (1000 units after max x-coordinate)
        offset = max(points['x'].max(), embeddings['x'].max()) + 1000
        
        all_points.append(points)
        all_embeddings.append(embeddings)
    
    return pd.concat(all_points, ignore_index=True), pd.concat(all_embeddings, ignore_index=True)

def process(points_df: pd.DataFrame, embeddings_df: pd.DataFrame) -> dict:
    """
    Process data using Points2Regions and merge with morphological embeddings.
    
    This function performs several steps:
    1. Groups data by sample and applies Points2Regions
    2. Reduces dimensionality of morphological features using PCA
    3. Integrates gene expression with reduced morphological features
    
    Parameters
    ----------
    points_df : pd.DataFrame
        DataFrame containing gene positions and labels.
        Required columns: x, y, Gene, sample_idx
    embeddings_df : pd.DataFrame
        DataFrame containing morphological embeddings.
        Required columns: x, y, Feature_*, sample_idx
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'data': Combined feature matrix with spatial coordinates
        - 'n_genes': Number of genes in the dataset
    """
    # Process each sample separately
    all_gene_data = []
    sample_indices = points_df['sample_idx'].unique()
    
    for idx in sample_indices:
        # Extract sample data
        sample_points = points_df[points_df['sample_idx'] == idx]
        
        # Apply Points2Regions
        print(f"    > Processing Points2Regions for sample {idx+1}...")
        gene_positions = sample_points[['x', 'y']].to_numpy()
        adata = points2regions(
            xy=gene_positions,
            gene_labels=sample_points['Gene'].to_numpy(),
            sigma=POINTS2REGIONS_PARAMS['sigma'],
            n_clusters=100, # Not important
            bin_width=None,
            min_genes_per_bin=POINTS2REGIONS_PARAMS['min_genes_per_bin'],
            region_name="points2regions",
            return_anndata=True
        )
        
        # Create gene expression dataframe
        df_genes = pd.DataFrame(
            np.hstack([adata.obsm['spatial'].astype(int),
                      adata.X.toarray()]),
            columns=['x', 'y'] + list(adata.var_names)
        )
        df_genes['sample_idx'] = idx
        all_gene_data.append(df_genes)
    
    # Combine processed gene data
    combined_genes = pd.concat(all_gene_data, ignore_index=True)
    
    # Create AnnData object for normalization
    adata_final = ad.AnnData(
        X=combined_genes.drop(['x', 'y', 'sample_idx'], axis=1).values,
        obsm={'spatial': combined_genes[['x', 'y']].values}
    )
    
    # Normalize gene expression data
    sc.pp.normalize_total(adata_final)
    sc.pp.log1p(adata_final)
    
    # Process morphological features
    print("Applying PCA to embeddings...")
    feature_columns = [col for col in embeddings_df.columns if col.startswith('Feature')]
    scaled_embeddings = minmax_scale(embeddings_df[feature_columns], axis=0)
    pca = PCA(n_components=.95)  # Keep 95% of variance
    reduced_embeddings = pca.fit_transform(scaled_embeddings)
    print(f"    > Reduced embeddings from {len(feature_columns)} to {reduced_embeddings.shape[1]} dimensions")
    
    # Create coordinate-to-embedding mapping
    coord_to_embedding = {
        (x, y): emb for x, y, emb in zip(
            embeddings_df['x'],
            embeddings_df['y'],
            reduced_embeddings
        )
    }
    
    # Align embeddings with Points2Regions output
    aligned_embeddings = np.array([
        coord_to_embedding.get((x, y), np.zeros(reduced_embeddings.shape[1]))
        for x, y in adata_final.obsm['spatial']
    ])
    
    # Get gene expression matrix
    genes_matrix = (adata_final.X.toarray() if scipy.sparse.issparse(adata_final.X) 
                   else adata_final.X)
    
    # Combine gene expression and morphological features
    feature_matrix = np.hstack([genes_matrix, aligned_embeddings])
    
    return {
        'data': pd.DataFrame(
            np.hstack([
                adata_final.obsm['spatial'],  # Spatial coordinates
                feature_matrix
            ]),
            index=combined_genes.index,
            columns=['x', 'y'] + 
                    list(adata_final.var_names) + 
                    [f'PCA_{i+1}' for i in range(reduced_embeddings.shape[1])]
        ),
        'n_genes': len(adata_final.var_names)
    }

def load_data(points_paths: str | list, embeddings_paths: str | list) -> dict:
    """
    Load and process single or multiple samples.
    
    Parameters
    ----------
    points_paths : str or list
        Path(s) to CSV file(s) with gene positions and labels
    embeddings_paths : str or list
        Path(s) to CSV file(s) with morphological embeddings
        
    Returns
    -------
    dict
        Processed data dictionary containing combined features and metadata
    """
    # Handle tuple input for embeddings_paths
    if isinstance(embeddings_paths, tuple):
        embeddings_paths = embeddings_paths[0]
    
    # Convert single paths to lists
    if isinstance(points_paths, str):
        points_paths = [points_paths]
        embeddings_paths = [embeddings_paths]
    
    # Validate input
    if len(points_paths) != len(embeddings_paths):
        raise ValueError("Number of points files must match number of embeddings files")
    
    # Process single or multiple samples
    if len(points_paths) > 1:
        points_df, embeddings_df = join_samples(points_paths, embeddings_paths)
    else:
        points_df = pd.read_csv(points_paths[0])
        embeddings_df = pd.read_csv(embeddings_paths[0])
        points_df['sample_idx'] = 0
        embeddings_df['sample_idx'] = 0
    
    return process(points_df, embeddings_df)

