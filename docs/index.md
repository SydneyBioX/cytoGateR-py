# cytoGateR Python

**Marker-aware cell annotation for spatial proteomics.**

cytoGateR combines interpretable marker gating, supervised prediction, and
uncertainty measurements for single-cell spatial data stored in
[`AnnData`](https://anndata.readthedocs.io/). The Python API follows the R
package's function names, parameters, defaults, and result structure wherever
Python syntax permits.

```{image} https://i.imgur.com/BmRpXzR.png
:width: 180px
:align: right
:alt: cytoGateR logo
```

Install the current release from PyPI:

```bash
pip install cytogater
```

Optional plotting, neural-network, and notebook dependencies are available as
extras:

```bash
pip install "cytogater[plot]"
pip install "cytogater[torch]"
pip install "cytogater[notebook]"
```

## What cytoGateR provides

- marker GMMs, soft gating, and marker-priority tree gating;
- confidence-based reference selection;
- weighted kNN, random-forest, and optional PyTorch prediction;
- hierarchical reference construction and recursive hierarchical kNN;
- classification metrics, probability uncertainty, and spatial uncertainty;
- Matplotlib plots for labels, markers, probabilities, metrics, and trees.

## Minimal workflow

```python
import cytogater

result = cytogater.run_tree_gating(
    adata,
    lineage_table,
    assay_name="exprs",
)

labels = result["hard_label"]
probabilities = result["prob_mat"]
```

See {doc}`getting-started` for the expected data layout and {doc}`tutorial` for
a complete, executable example.

```{toctree}
:maxdepth: 2
:hidden:

getting-started
tutorial
api
```
