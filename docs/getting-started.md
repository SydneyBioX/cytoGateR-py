# Getting started

## Installation

Install cytoGateR from PyPI:

```bash
pip install cytogater
```

Install an optional feature set when it is needed:

```bash
pip install "cytogater[plot]"      # Matplotlib and seaborn plots
pip install "cytogater[torch]"     # neural-network model
pip install "cytogater[notebook]"  # JupyterLab and notebook execution
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/SydneyBioX/cytoGateR-py.git
cd cytoGateR-py
python -m pip install -e ".[test,plot,notebook]"
pytest
```

## AnnData layout

Python `AnnData` stores cells in rows and markers in columns. Place the
expression matrix in the layer passed as `assay_name`; the default is:

```python
adata.layers["exprs"]
```

The corresponding marker names come from `adata.var_names`. Spatial plotting
functions expect two-dimensional coordinates in:

```python
adata.obsm["spatial"]
```

## Lineage table

The lineage table is a pandas data frame with `cell_type`, `pos_markers`, and
`neg_markers` columns. Marker columns contain Python lists:

```python
import pandas as pd

lineage_table = pd.DataFrame(
    {
        "cell_type": ["T cell", "B cell"],
        "pos_markers": [["CD3e"], ["CD20"]],
        "neg_markers": [["CD20"], ["CD3e"]],
    }
)
```

Markers absent from `adata.var_names` are removed during validation. A cell
type must retain at least one positive marker.

## Outputs

Gating functions return a dictionary containing the annotated `AnnData`, the
cell-by-cell-type probability matrix, labels, and fitted gating objects. Common
outputs are also stored directly on the `AnnData` object:

- labels and `P_*` probabilities in `adata.obs`;
- probability matrices in `adata.obsm`;
- probability-matrix column names in `adata.uns`.

## R compatibility

The Python implementation uses a deterministic two-component Gaussian-mixture
fit. The R package uses `mclust`, so fitted cutoffs can differ slightly while
the workflow and output structure remain aligned. Python's `lambda_` parameter
corresponds to R's `lambda`, and `num_trees` corresponds to `num.trees`.
