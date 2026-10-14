"""Probability and spatial uncertainty."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def calculate_uncertainty(prob_mat, spe, sample_col="sample_id", k_spatial=15):
    """Calculate probability and spatial uncertainty for every cell.

    Parameters
    ----------
    prob_mat : pandas.DataFrame or array-like
        Cell-by-cell-type probabilities. Data-frame rows are aligned to
        ``spe.obs_names``.
    spe : anndata.AnnData
        Annotated object with coordinates in ``spe.obsm["spatial"]``.
    sample_col : str, default="sample_id"
        ``spe.obs`` column defining independent spatial samples.
    k_spatial : int, default=15
        Nearest neighbours used for spatial label discordance.

    Returns
    -------
    pandas.DataFrame
        Entropy, Gini impurity, top-two probability margin uncertainty, and
        within-sample spatial discordance for each cell.

    Notes
    -----
    Spatial discordance is missing for samples with at most ``k_spatial``
    cells.
    """
    probabilities = pd.DataFrame(prob_mat).reindex(spe.obs_names)
    values = probabilities.to_numpy(float)
    labels = probabilities.columns[
        np.nanargmax(np.nan_to_num(values, nan=-np.inf), axis=1)
    ].to_numpy()
    log_values = np.zeros_like(values)
    positive = values > 0
    log_values[positive] = np.log(values[positive])
    entropy = -np.nansum(values * log_values, axis=1) / np.log(values.shape[1])
    gini = (1 - np.nansum(values**2, axis=1)) / (1 - 1 / values.shape[1])
    ordered = np.sort(values, axis=1)[:, ::-1]
    margin = 1 - (ordered[:, 0] - ordered[:, 1])
    discordance = np.full(len(values), np.nan)
    coordinates = np.asarray(spe.obsm["spatial"])
    for sample in pd.unique(spe.obs[sample_col]):
        index = np.flatnonzero(spe.obs[sample_col].to_numpy() == sample)
        if len(index) <= k_spatial:
            continue
        neighbors = NearestNeighbors(n_neighbors=k_spatial + 1).fit(
            coordinates[index]
        ).kneighbors(return_distance=False)[:, 1:]
        discordance[index] = np.mean(labels[index][neighbors] != labels[index, None], axis=1)
    return pd.DataFrame({
        "cell_id": spe.obs_names, "entropy": entropy,
        "gini_impurity": gini, "margin_uncertainty": margin,
        "spatial_discordance": discordance,
    }, index=spe.obs_names)


def calculate_spatial_prior_labels(
    spe,
    prob_mat,
    k_spatial=50,
    lambda_=0.2,
    protect_threshold=0.85,
    out_col="knn_spatial_label",
):
    """Refine low-confidence predictions with a local spatial prior.

    Parameters
    ----------
    spe : anndata.AnnData
        Object with coordinates in ``spe.obsm["spatial"]``; modified in place.
    prob_mat : pandas.DataFrame or array-like
        Cell-by-cell-type probability matrix.
    k_spatial : int, default=50
        Number of spatial neighbours used to estimate the local prior.
    lambda_ : float, default=0.2
        Exponent controlling the influence of the neighbour prior.
    protect_threshold : float, default=0.85
        Predictions at or above this confidence are never changed.
    out_col : str, default="knn_spatial_label"
        Destination label column in ``spe.obs``.

    Returns
    -------
    anndata.AnnData
        The same object with spatially refined labels.
    """
    probabilities = pd.DataFrame(prob_mat).reindex(spe.obs_names)
    values = probabilities.to_numpy(float)
    classes = probabilities.columns.to_numpy()
    winner = np.nanargmax(np.nan_to_num(values, nan=-np.inf), axis=1)
    confidence = np.nanmax(values, axis=1)
    k = min(k_spatial, len(values) - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(
        spe.obsm["spatial"]
    ).kneighbors(return_distance=False)[:, 1:]
    result = classes[winner].astype(object)
    for i in range(len(values)):
        if confidence[i] >= protect_threshold:
            continue
        prior = np.bincount(winner[neighbors[i]], minlength=len(classes)) / k
        posterior = values[i] * prior**lambda_
        result[i] = classes[np.nanargmax(np.nan_to_num(posterior))]
    spe.obs[out_col] = result
    return spe
