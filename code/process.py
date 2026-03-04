import numpy as np
import scanpy as sc
import pandas as pd

def to_categorical_np(y, num_classes=None):
    """Convert labels to one-hot encoding using numpy."""
    y = np.array(y, dtype=int)
    if num_classes is None:
        num_classes = np.max(y) + 1  # Assume labels are from 0 to num_classes-1
    return np.eye(num_classes)[y]

def random_mix(Xs, ys, nmix=5, n_samples=10000, seed=0):
    # Define empty lists
    Xs_new, ys_new = [], []

    # Convert labels to one-hot encoding using numpy
    ys_ = to_categorical_np(ys)

    rstate = np.random.RandomState(seed)
    fraction_all = rstate.rand(n_samples, nmix)  # Random fractions for mixing
    randindex_all = rstate.randint(len(Xs), size=(n_samples, nmix))  # Random selection of cells

    for i in range(n_samples):
        # fraction: random fraction across the "nmix" number of sampled cells
        fraction = fraction_all[i]
        fraction = fraction / np.sum(fraction)  # Normalize to sum to 1
        fraction = np.reshape(fraction, (nmix, 1))

        # Random selection of the single cell data by the index
        randindex = randindex_all[i]
        ymix = ys_[randindex]  # Get the one-hot encoded labels for the selected cells

        # Calculate the fraction of cell types in the cell mixture
        yy = np.sum(ymix * fraction, axis=0)  # Weighted sum of cell type fractions

        # Calculate weighted gene expression of the cell mixture
        XX = np.asarray(Xs[randindex]) * fraction  # Weighted gene expression
        XX_ = np.sum(XX, axis=0)  # Sum to get the final gene expression profile

        # Add cell type fraction & composite gene expression to the lists
        ys_new.append(yy)
        Xs_new.append(XX_)

    Xs_new = np.asarray(Xs_new)  # Convert list of samples to numpy array
    ys_new = np.asarray(ys_new)  # Convert list of labels to numpy array

    return Xs_new, ys_new

def random_mix_with_dominant(Xs, ys, nmix=5, n_samples=10000, n_dominant=30, dominant_ratio=0.7, seed=0):
    """
    Generate n_dominant dominant pseudo-spots for each cell type (dominant fraction > dominant_ratio
    and randomly sampled in the interval (dominant_ratio, 1.0)); the remaining pseudo-spots are
    generated using the original random_mix strategy.

    Returns
    -------
    Xs_new : np.ndarray
        Pseudo-spot expression matrix.
    ys_new : np.ndarray
        Pseudo-spot cell-type fraction matrix.
    """
    Xs_new, ys_new = [], []
    ys_ = to_categorical_np(ys)
    rstate = np.random.RandomState(seed)
    num_classes = ys_.shape[1]
    # 1. Dominant pseudo-spots for each cell type
    for cell_type in range(num_classes):
        for _ in range(n_dominant):
            # Sample dominant-class fraction from (dominant_ratio, 1.0)
            dom_ratio = rstate.uniform(dominant_ratio, 1.0)
            rest_total = 1 - dom_ratio
            rest = rstate.rand(num_classes - 1)
            rest = rest / rest.sum() * rest_total if rest_total > 0 else np.zeros(num_classes - 1)
            yy = np.zeros(num_classes)
            yy[cell_type] = dom_ratio
            yy[np.arange(num_classes) != cell_type] = rest
            # Generate mixed expression profile for this pseudo-spot
            n_dom = max(1, int(np.round(nmix * dom_ratio)))
            n_rest = nmix - n_dom
            dom_indices = np.where(np.argmax(ys_, axis=1) == cell_type)[0]
            rest_indices = np.where(np.argmax(ys_, axis=1) != cell_type)[0]
            if len(dom_indices) == 0 or (n_rest > 0 and len(rest_indices) == 0):
                continue
            dom_sample = rstate.choice(dom_indices, n_dom, replace=True)
            rest_sample = rstate.choice(rest_indices, n_rest, replace=True) if n_rest > 0 else []
            indices = np.concatenate([dom_sample, rest_sample])
            indices = indices.astype(int)
            fraction = np.zeros(nmix)
            fraction[:n_dom] = dom_ratio / n_dom
            if n_rest > 0:
                rest_fraction = rest / rest.sum() * rest_total if rest_total > 0 else np.zeros(n_rest)
                fraction[n_dom:] = rest_fraction[:n_rest] if len(rest_fraction) >= n_rest else rest_total / n_rest
            fraction = fraction / fraction.sum()  # 再归一化
            XX = np.asarray(Xs[indices]) * fraction[:, None]
            XX_ = np.sum(XX, axis=0)
            ys_new.append(yy)
            Xs_new.append(XX_)
    # 2. Remaining pseudo-spots using the standard random_mix strategy
    n_random = n_samples - num_classes * n_dominant
    if n_random > 0:
        Xs_rand, ys_rand = random_mix(Xs, ys, nmix=nmix, n_samples=n_random, seed=seed+42)
        Xs_new.extend(Xs_rand)
        ys_new.extend(ys_rand)
    Xs_new = np.asarray(Xs_new)
    ys_new = np.asarray(ys_new)
    return Xs_new, ys_new