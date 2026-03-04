from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

def build_real_spot_graph(expr_df, coords_df, k=5, alpha=0.7, add_self_loop=True, use_mnn=True):
    """
    Build a real spot ↔ real spot adjacency matrix, combining expression similarity and spatial distance.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Expression matrix (n_spots, n_genes)

    coords_df : pd.DataFrame
        Coordinate matrix (n_spots, 2), containing x and y

    k : int
        Number of nearest neighbors per spot (for KNN/MNN)

    alpha : float (0~1)
    Fusion weight, final similarity = alpha * expression_similarity + (1 - alpha) * spatial_similarity

    add_self_loop : bool
        Whether to add self-loops (diagonal edges)

    use_mnn : bool
        Whether to use Mutual Nearest Neighbors

    Returns
    -------
    coo_matrix
        Sparse adjacency matrix with shape (n_spots, n_spots)
    """

    expr = expr_df.values
    coords = coords_df.values
    n = expr.shape[0]

    # 1. Expression similarity
    sim_expr = cosine_similarity(expr)
    sim_expr_df = pd.DataFrame(sim_expr)
    sim_expr_df.to_csv('sim_expr.csv')
    # 2. Spatial similarity (Gaussian kernel)
    dist_spatial = cdist(coords, coords, metric='euclidean')
    sigma = np.mean(dist_spatial)
    sim_spatial = np.exp(- dist_spatial ** 2 / (2 * sigma ** 2))
    # with np.errstate(divide='ignore'):
    #     sim_spatial = 1 / (dist_spatial + 1e-8)
    #
    # # Optional normalization
    # sim_spatial = sim_spatial / np.max(sim_spatial)
    # np.fill_diagonal(sim_spatial, 0)
    # sim_spatial = sim_spatial / np.max(sim_spatial)
    # np.fill_diagonal(sim_spatial, 0)

    # Save spatial similarity for inspection
    sim_spatial_df = pd.DataFrame(sim_spatial)
    sim_spatial_df.to_csv('sim_spatial.csv')
    # 3. Fused similarity
    sim_combined = alpha * sim_expr + (1 - alpha) * sim_spatial
    np.fill_diagonal(sim_combined, 0)  # Mask diagonal to avoid self-bias

    # 4. Build adjacency edges
    nn_idx = np.argsort(-sim_combined, axis=1)[:, :k]
    rows, cols, data = [], [], []

    for i in range(n):
        for j in nn_idx[i]:
            if use_mnn:
                if i in nn_idx[j]:  # Mutual nearest neighbors
                    rows.append(i)
                    cols.append(j)
                    data.append(sim_combined[i, j])
            else:
                rows.append(i)
                cols.append(j)
                data.append(sim_combined[i, j])

    # 5. Optionally add self-loops (set to 1.0)
    if add_self_loop:
        for i in range(n):
            rows.append(i)
            cols.append(i)
            data.append(1.0)

    # 6. Return COO sparse matrix
    adj = coo_matrix((data, (rows, cols)), shape=(n, n))
    return adj
def build_pseudo_real_graph(expr_real_df, expr_pseudo_df, k=5):
    """
    Build a pseudo spot ↔ real spot adjacency graph using Mutual Nearest Neighbors (MNN).

    Parameters
    ----------
    expr_real_df : pd.DataFrame
        Expression matrix for real spots (n_real, n_genes)
    expr_pseudo_df : pd.DataFrame
        Expression matrix for pseudo spots (n_pseudo, n_genes)
    k : int
        Number of neighbors in KNN

    Returns
    -------
    coo_matrix
        Sparse matrix with shape (n_total, n_total), suitable as adjacency for a heterogeneous graph.
        Pseudo spots occupy the latter part of rows/columns.
    """
    expr_real = expr_real_df.values
    expr_pseudo = expr_pseudo_df.values

    n_real = expr_real.shape[0]
    n_pseudo = expr_pseudo.shape[0]

    # Compute similarity between real and pseudo spots
    sim_real2pseudo = cosine_similarity(expr_real, expr_pseudo)  # shape (n_real, n_pseudo)
    sim_pseudo2real = sim_real2pseudo.T  # shape (n_pseudo, n_real)

    # Get top-k neighbors
    nn_real = np.argsort(-sim_real2pseudo, axis=1)[:, :k]
    nn_pseudo = np.argsort(-sim_pseudo2real, axis=1)[:, :k]

    rows, cols, data = [], [], []

    for i in range(n_real):
        for j in nn_real[i]:
            if i in nn_pseudo[j]:
                # Add edges: real i ↔ pseudo (n_real + j)
                rows.append(i)
                cols.append(n_real + j)
                data.append(sim_real2pseudo[i, j])
                # Symmetric edge
                rows.append(n_real + j)
                cols.append(i)
                data.append(sim_real2pseudo[i, j])

    # Return COO sparse matrix (to be used when assembling the heterogeneous graph)
    total_nodes = n_real + n_pseudo
    adj = coo_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes))
    return adj


def build_spot_gene_graph(expr_real, expr_pseudo, threshold=0.0):
    """
    Build a sparse adjacency matrix from real/pseudo spots to genes (column-normalized separately).

    Parameters
    ----------
    expr_real : pd.DataFrame
        Expression matrix for real spots, shape: (n_real_spots, n_genes)
    expr_pseudo : pd.DataFrame
        Expression matrix for pseudo spots, shape: (n_pseudo_spots, n_genes)
    threshold : float
        Expression threshold; edges are created only for values above this threshold

    Returns
    -------
    coo_matrix
        Adjacency matrix with shape (n_total_spots, n_genes)
    """
    # 1. Column-wise (per gene) normalization so each gene column sums to 1
    expr_real_norm = expr_real.div(expr_real.sum(axis=0), axis=1)
    expr_pseudo_norm = expr_pseudo.div(expr_pseudo.sum(axis=0), axis=1)

    # 2. Concatenate real and pseudo spot matrices
    expr_all = pd.concat([expr_real_norm, expr_pseudo_norm], axis=0)

    # 3. Build sparse matrix
    rows, cols, data = [], [], []
    for i in range(expr_all.shape[0]):
        for j in range(expr_all.shape[1]):
            val = expr_all.iat[i, j]
            if val > threshold:
                rows.append(i)
                cols.append(j)
                data.append(val)

    adj = coo_matrix((data, (rows, cols)), shape=(expr_all.shape[0], expr_all.shape[1]))
    return adj




