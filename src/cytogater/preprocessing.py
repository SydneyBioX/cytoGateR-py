"""Expression access and marker preprocessing."""

import numpy as np
import pandas as pd


def expression_frame(spe, assay_name="exprs"):
    """Return AnnData expression as a labelled dense data frame.

    Parameters
    ----------
    spe : anndata.AnnData
        Cells-by-markers data object.
    assay_name : str or None, default="exprs"
        Layer to read. Use ``None`` or ``"X"`` for ``spe.X``.

    Returns
    -------
    pandas.DataFrame
        Dense float matrix indexed by ``spe.obs_names`` with ``spe.var_names``
        as columns.
    """
    x = spe.X if assay_name in (None, "X") else spe.layers[assay_name]
    if hasattr(x, "toarray"):
        x = x.toarray()
    return pd.DataFrame(
        np.asarray(x, dtype=float), index=spe.obs_names, columns=spe.var_names
    )


def clean_lineage_table(lineage_table, spe):
    """Normalise a lineage table and remove unavailable markers.

    Parameters
    ----------
    lineage_table : pandas.DataFrame or mapping
        Table with cell types and positive/negative marker lists.
    spe : anndata.AnnData
        Object whose ``var_names`` define the available markers.

    Returns
    -------
    pandas.DataFrame
        Cleaned table containing only cell types with at least one positive
        marker present in ``spe``.
    """
    table = pd.DataFrame(lineage_table).copy()
    markers = set(map(str, spe.var_names))
    table["cell_type"] = table.cell_type.astype(str).str.strip()
    for column in ("pos_markers", "neg_markers"):
        table[column] = table[column].map(
            lambda values: [str(x) for x in values if str(x) in markers]
        )
    return table[table.pos_markers.map(bool)].reset_index(drop=True)
