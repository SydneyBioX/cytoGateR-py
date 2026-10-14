---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: "0.13"
    jupytext_version: "1.16"
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Basic annotation pipeline

This example creates a small spatial dataset, performs marker-priority tree
gating, selects high-confidence labels, and visualises the result. It is
executed from start to finish whenever the documentation is built.

## Create an AnnData object

For demonstration purposes, half the cells resemble T cells and half resemble
B cells. Real data can be loaded from any source that produces an `AnnData`
object with cells in rows and markers in columns.

```{code-cell} ipython3
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cytogater

rng = np.random.default_rng(42)
n_cells = 600
truth = np.repeat(["T cell", "B cell"], n_cells // 2)

expression = np.column_stack(
    [
        np.where(truth == "T cell", rng.normal(4, 0.7, n_cells), rng.normal(1, 0.5, n_cells)),
        np.where(truth == "B cell", rng.normal(4, 0.7, n_cells), rng.normal(1, 0.5, n_cells)),
    ]
)

adata = ad.AnnData(expression)
adata.var_names = ["CD3e", "CD20"]
adata.layers["exprs"] = expression.copy()
adata.obs["truth"] = pd.Categorical(truth)
adata.obsm["spatial"] = rng.uniform(0, 100, size=(n_cells, 2))
adata
```

## Define marker knowledge

Positive markers support a cell-type assignment; negative markers penalise an
assignment when they are expressed.

```{code-cell} ipython3
lineage_table = pd.DataFrame(
    {
        "cell_type": ["T cell", "B cell"],
        "pos_markers": [["CD3e"], ["CD20"]],
        "neg_markers": [["CD20"], ["CD3e"]],
    }
)
lineage_table
```

## Run tree gating

`run_tree_gating()` fits marker distributions, constructs a marker-priority
tree for each cell type, and calculates a probability for every cell and cell
type.

```{code-cell} ipython3
result = cytogater.run_tree_gating(
    adata,
    lineage_table,
    assay_name="exprs",
    min_cells=50,
    min_score=0.1,
    workers=1,
)

result["prob_mat"].head()
```

The hard labels and probabilities are available both in the returned result
and on the annotated object.

```{code-cell} ipython3
adata.obs[["truth", "cell_type_hard", "prob_best"]].head()
```

## Select confident reference cells

Probability cutoffs can identify an initial reference set for supervised
prediction. Cells that do not uniquely exceed a cell-type cutoff remain
`Unknown`.

```{code-cell} ipython3
result = cytogater.apply_cutoff_labels(result)
adata.obs["cutoff_label"].value_counts()
```

## Inspect the spatial labels

```{code-cell} ipython3
:tags: [remove-input]

ax = cytogater.plot_labelled_cells(
    adata,
    "cell_type_hard",
    s=9,
    title="Tree-gating labels",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
```

The selected labels can now be used with
{func}`cytogater.train_custom_knn`,
{func}`cytogater.train_custom_randomforest`, or the hierarchical prediction
workflow documented in the {doc}`api`.
