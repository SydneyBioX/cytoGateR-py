"""Minimal validation matching the R package's required inputs."""

import pandas as pd


def validate_lineage_table(lineage_table):
    """Validate the required lineage-table columns.

    Parameters
    ----------
    lineage_table : pandas.DataFrame or mapping
        Candidate lineage table.

    Returns
    -------
    pandas.DataFrame
        Data-frame representation of the input.

    Raises
    ------
    ValueError
        If ``cell_type``, ``pos_markers``, or ``neg_markers`` is absent.
    """
    table = pd.DataFrame(lineage_table)
    missing = {"cell_type", "pos_markers", "neg_markers"} - set(table)
    if missing:
        raise ValueError("lineage_table missing columns: " + ", ".join(sorted(missing)))
    return table


def validate_adata(spe, assay_name="exprs"):
    """Validate that an AnnData expression layer exists.

    Parameters
    ----------
    spe : anndata.AnnData
        Object to validate.
    assay_name : str or None, default="exprs"
        Required layer, or ``None``/``"X"`` to use ``spe.X``.

    Returns
    -------
    anndata.AnnData
        The unchanged input object.

    Raises
    ------
    ValueError
        If the requested layer is absent.
    """
    if assay_name not in (None, "X") and assay_name not in spe.layers:
        raise ValueError(f"Assay '{assay_name}' not found in adata.layers.")
    return spe
