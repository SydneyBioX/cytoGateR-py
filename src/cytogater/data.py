"""Small data conversion helpers."""

import pandas as pd


def lineage_table(cell_type, pos_markers, neg_markers=None):
    """Create the list-column lineage table expected by cytoGateR.

    Parameters
    ----------
    cell_type : sequence of str
        Cell-type names.
    pos_markers : sequence of sequences of str
        Positive marker names for each cell type.
    neg_markers : sequence of sequences of str or None, default=None
        Negative marker names. ``None`` creates an empty list for every type.

    Returns
    -------
    pandas.DataFrame
        Table with ``cell_type``, ``pos_markers``, and ``neg_markers`` columns.
    """
    if neg_markers is None:
        neg_markers = [[] for _ in cell_type]
    return pd.DataFrame({
        "cell_type": cell_type,
        "pos_markers": pos_markers,
        "neg_markers": neg_markers,
    })


def probabilities_from_obs(spe, prefix="P_"):
    """Read cell-type probability columns stored in ``adata.obs``.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated data object.
    prefix : str, default="P_"
        Prefix identifying probability columns.

    Returns
    -------
    pandas.DataFrame
        Probability columns with the prefix removed and underscores converted
        back to spaces in cell-type names.
    """
    columns = [x for x in spe.obs.columns if x.startswith(prefix)]
    out = spe.obs[columns].copy()
    out.columns = [x[len(prefix):].replace("_", " ") for x in columns]
    return out
