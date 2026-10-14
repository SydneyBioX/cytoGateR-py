"""Hierarchical reference construction."""

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from ..preprocessing import expression_frame


def build_lineage_hierarchy(
    spe, label_col="cleaned_core_label", assay_name="exprs"
):
    """Build a complete-linkage hierarchy from cell-type pseudobulks.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated cells-by-markers object.
    label_col : str, default="cleaned_core_label"
        ``spe.obs`` column containing cleaned reference labels. Missing and
        ``"Unknown"`` labels are excluded.
    assay_name : str or None, default="exprs"
        Expression layer, or ``None``/``"X"`` for ``spe.X``.

    Returns
    -------
    dict
        ``linkage`` contains the SciPy complete-linkage matrix and ``labels``
        gives the corresponding cell-type leaf names.
    """
    frame = expression_frame(spe, assay_name)
    labels = spe.obs[label_col]
    keep = labels.notna() & (labels.astype(str) != "Unknown")
    groups = list(dict.fromkeys(labels[keep].astype(str)))
    averages = np.column_stack([
        frame.loc[keep & (labels.astype(str) == group)].mean(axis=0)
        for group in groups
    ])
    correlation = np.corrcoef(averages, rowvar=False)
    distance = np.nan_to_num(1 - correlation, nan=1.0)
    distance = np.clip((distance + distance.T) / 2, 0, None)
    np.fill_diagonal(distance, 0)
    return {"linkage": linkage(squareform(distance), method="complete"), "labels": groups}


def build_hierarchical_reference(
    spe,
    hc_tree,
    marker_stats=None,
    label_col="cleaned_core_label",
    top_n=5,
    assay_name="exprs",
    unknown_label="Unassigned",
):
    """Create a binary training reference for every hierarchy node.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated cells-by-markers object.
    hc_tree : dict
        Hierarchy returned by :func:`build_lineage_hierarchy`.
    marker_stats : dict or None, default=None
        Optional marker statistics from gating. When supplied, node-specific
        markers are ranked by weighted mean difference.
    label_col : str, default="cleaned_core_label"
        ``spe.obs`` column containing reference cell types.
    top_n : int or None, default=5
        Number of discriminating markers retained per node. ``None`` keeps all.
    assay_name : str or None, default="exprs"
        Expression layer, or ``None``/``"X"`` for ``spe.X``.
    unknown_label : str, default="Unassigned"
        Reference label excluded from training.

    Returns
    -------
    dict
        Entries named ``Node_*`` contain binary training data, labels, marker
        names, descendant cell types, and cluster IDs. ``_root`` identifies
        the root cluster.
    """
    frame = expression_frame(spe, assay_name)
    labels = spe.obs[label_col].astype(object)
    keep = labels.notna() & (labels != unknown_label)
    root, nodes = to_tree(hc_tree["linkage"], rd=True)
    leaf_names = hc_tree["labels"]

    def leaves(node):
        if node.is_leaf():
            return [leaf_names[node.id]]
        return leaves(node.left) + leaves(node.right)

    result = {}
    internal = [node for node in nodes if not node.is_leaf()]
    for number, node in enumerate(internal, 1):
        left_members, right_members = leaves(node.left), leaves(node.right)
        selected = keep & labels.isin(left_members + right_members)
        node_labels = np.where(labels[selected].isin(left_members), "Left", "Right")
        markers = list(marker_stats) if marker_stats is not None else list(frame.columns)
        if top_n is not None and marker_stats is not None:
            left_mean = frame.loc[selected].loc[node_labels == "Left", markers].mean()
            right_mean = frame.loc[selected].loc[node_labels == "Right", markers].mean()
            score = (left_mean - right_mean).abs() * np.array(
                [marker_stats[m].get("weight", 0.1) for m in markers]
            )
            markers = score.sort_values(ascending=False).index[:top_n].tolist()
        result[f"Node_{number}"] = {
            "train_data": frame.loc[selected, markers],
            "train_labels": node_labels,
            "markers": markers,
            "left_members": left_members,
            "right_members": right_members,
            "cluster_id": node.id,
        }
    result["_root"] = root.id
    return result
