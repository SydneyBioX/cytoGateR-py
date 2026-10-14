"""Matplotlib visualisations corresponding to the R plotting API."""

import numpy as np
import pandas as pd

from .preprocessing import expression_frame


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_probability_map(
    spe, probability, title=None, spatial_key="spatial", ax=None, **kwargs
):
    """Plot a probability or continuous value in spatial coordinates.

    Parameters
    ----------
    spe : anndata.AnnData
        Object containing spatial coordinates.
    probability : str or array-like
        ``spe.obs`` column name or one value per cell.
    title : str or None, default=None
        Plot title; defaults to ``probability``.
    spatial_key : str, default="spatial"
        Key in ``spe.obsm`` containing x/y coordinates.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on; a new figure is created when omitted.
    **kwargs
        Additional arguments passed to ``Axes.scatter``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the scatter plot and colour bar.
    """
    ax = ax or _plt().subplots()[1]
    coordinates = np.asarray(spe.obsm[spatial_key])
    values = spe.obs[probability] if isinstance(probability, str) else probability
    artist = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=values, **kwargs)
    ax.set(title=title or str(probability), aspect="equal")
    _plt().colorbar(artist, ax=ax)
    return ax


def plot_labelled_cells(
    spe, label_col, spatial_key="spatial", ax=None, title=None, **kwargs
):
    """Plot categorical cell labels in spatial coordinates.

    Parameters
    ----------
    spe : anndata.AnnData
        Object containing labels and spatial coordinates.
    label_col : str
        Categorical label column in ``spe.obs``.
    spatial_key : str, default="spatial"
        Key in ``spe.obsm`` containing x/y coordinates.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.
    title : str or None, default=None
        Plot title; defaults to ``label_col``.
    **kwargs
        Additional arguments passed to ``Axes.scatter``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the spatial label plot.
    """
    ax = ax or _plt().subplots()[1]
    coordinates = np.asarray(spe.obsm[spatial_key])
    labels = pd.Categorical(spe.obs[label_col])
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels.codes, **kwargs)
    ax.set(title=title or label_col, aspect="equal")
    return ax


def plot_marker_density(spe, marker, assay_name="exprs", label_col=None, ax=None):
    """Plot the expression density of one marker.

    Parameters
    ----------
    spe : anndata.AnnData
        Cells-by-markers object.
    marker : str
        Marker in ``spe.var_names``.
    assay_name : str or None, default="exprs"
        Expression layer, or ``None``/``"X"`` for ``spe.X``.
    label_col : str or None, default=None
        Optional ``spe.obs`` column used to draw one density per label.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the density histogram(s).
    """
    ax = ax or _plt().subplots()[1]
    frame = expression_frame(spe, assay_name)
    if label_col is None:
        ax.hist(frame[marker].dropna(), bins=40, density=True, alpha=0.7)
    else:
        for label, index in spe.obs.groupby(label_col, observed=True).groups.items():
            ax.hist(frame.loc[index, marker].dropna(), bins=40, density=True,
                    histtype="step", label=str(label))
        ax.legend()
    ax.set(xlabel=marker, ylabel="Density")
    return ax


def plot_confusion_matrix(conf_mat, title="Confusion matrix", ax=None):
    """Display a confusion/count matrix as a heatmap.

    Parameters
    ----------
    conf_mat : pandas.DataFrame or array-like
        Square count matrix. Data-frame labels are used as tick labels.
    title : str, default="Confusion matrix"
        Plot title.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the heatmap and colour bar.
    """
    ax = ax or _plt().subplots()[1]
    matrix = np.asarray(conf_mat)
    image = ax.imshow(matrix, cmap="Blues")
    labels = getattr(conf_mat, "columns", range(matrix.shape[1]))
    rows = getattr(conf_mat, "index", range(matrix.shape[0]))
    ax.set(xticks=range(len(labels)), xticklabels=labels,
           yticks=range(len(rows)), yticklabels=rows, title=title)
    _plt().colorbar(image, ax=ax)
    return ax


def plot_class_metrics(class_metrics, title="Per-class precision/recall/F1", subtitle=None, ax=None):
    """Plot per-class precision, recall, and F1 as grouped bars.

    Parameters
    ----------
    class_metrics : pandas.DataFrame or mapping
        Table with ``class``, ``precision``, ``recall``, and ``f1`` columns.
    title : str, default="Per-class precision/recall/F1"
        Plot title.
    subtitle : str or None, default=None
        Accepted for R API compatibility; currently unused.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the bar plot.
    """
    del subtitle
    ax = ax or _plt().subplots()[1]
    table = pd.DataFrame(class_metrics).set_index("class")
    table[["precision", "recall", "f1"]].plot.bar(ax=ax)
    ax.set(title=title, ylim=(0, 1), ylabel="Score")
    return ax


def plot_label_counts(spe, label_col, ax=None):
    """Plot the number of cells assigned to each label.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated data object.
    label_col : str
        Label column in ``spe.obs``.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the count bars.
    """
    ax = ax or _plt().subplots()[1]
    spe.obs[label_col].value_counts().plot.bar(ax=ax)
    ax.set(ylabel="Cells", title=label_col)
    return ax


def plot_label_agreement(agreement, ax=None):
    """Plot per-label agreement rates.

    Parameters
    ----------
    agreement : pandas.DataFrame or mapping
        Table with ``label`` and ``agreement`` columns, such as the output of
        :func:`cytogater.label_agreement_rates`.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the agreement bars.
    """
    ax = ax or _plt().subplots()[1]
    table = pd.DataFrame(agreement)
    ax.bar(table["label"], table["agreement"])
    ax.set(ylim=(0, 1), ylabel="Agreement")
    return ax


def plot_label_confusion_matrix(spe, label_col1, label_col2, ax=None):
    """Calculate and plot the confusion matrix between two label columns.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated data object.
    label_col1, label_col2 : str
        Label columns in ``spe.obs``.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the confusion heatmap.
    """
    from .metrics import label_confusion_matrix
    return plot_confusion_matrix(
        label_confusion_matrix(spe, label_col1, label_col2), ax=ax
    )


def plot_score_hist(res, ax=None):
    """Plot distributions of cell-type probabilities from a gating result.

    Parameters
    ----------
    res : dict
        Result containing a ``prob_mat`` entry.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing overlaid probability histograms.
    """
    ax = ax or _plt().subplots()[1]
    pd.DataFrame(res["prob_mat"]).plot.hist(alpha=0.4, bins=30, ax=ax)
    return ax


def plot_label_cardinality(label_mat, ax=None):
    """Plot how many cutoff labels are positive for each cell.

    Parameters
    ----------
    label_mat : pandas.DataFrame or array-like
        Cell-by-cell-type boolean label matrix.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing counts by label cardinality.
    """
    ax = ax or _plt().subplots()[1]
    pd.DataFrame(label_mat).sum(axis=1).value_counts().sort_index().plot.bar(ax=ax)
    ax.set(xlabel="Labels per cell", ylabel="Cells")
    return ax


def plot_ct_marker_intensity(spe, markers=None, label_col="cell_type_hard", assay_name="exprs", ax=None):
    """Plot mean marker expression for each annotated cell type.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated cells-by-markers object.
    markers : sequence of str or None, default=None
        Markers to display; ``None`` uses all markers.
    label_col : str, default="cell_type_hard"
        Cell-type column in ``spe.obs``.
    assay_name : str or None, default="exprs"
        Expression layer, or ``None``/``"X"`` for ``spe.X``.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the cell-type-by-marker heatmap.

    Notes
    -----
    ``plot_pseudobulk_heatmap`` and ``plot_label_dotplot`` are aliases of this
    function in the current Python implementation.
    """
    ax = ax or _plt().subplots()[1]
    frame = expression_frame(spe, assay_name)
    markers = list(frame.columns) if markers is None else markers
    means = frame[markers].groupby(spe.obs[label_col], observed=True).mean()
    image = ax.imshow(means, aspect="auto", cmap="viridis")
    ax.set(xticks=range(len(markers)), xticklabels=markers,
           yticks=range(len(means)), yticklabels=means.index)
    _plt().colorbar(image, ax=ax)
    return ax


plot_pseudobulk_heatmap = plot_ct_marker_intensity
plot_label_dotplot = plot_ct_marker_intensity


def tree_to_df(node, parent=None, id="root"):
    """Convert a gating tree to a labelled node table.

    Parameters
    ----------
    node : dict
        Root node returned by :func:`cytogater.build_fullcoverage_tree`.
    parent : str or None, default=None
        Parent identifier used internally during recursion.
    id : str, default="root"
        Identifier assigned to the current node.

    Returns
    -------
    pandas.DataFrame
        One row per node with its parent, label, split statistics, depth, and
        cell count.
    """
    is_leaf = node["type"] == "leaf"
    cutoff = node.get("cutoff")
    separation = node.get("sep_score")
    cutoff_label = float("nan") if cutoff is None else float(cutoff)
    separation_label = float("nan") if separation is None else float(separation)
    n_cells = len(node.get("cells", []))
    label = (
        f"leaf\nn={n_cells}"
        if is_leaf
        else f"{node['marker']} > {cutoff_label:.2f}\nsep={separation_label:.2f}"
    )
    rows = [{
        "id": id,
        "parent": parent,
        "type": node["type"],
        "label": label,
        "marker": node.get("marker"),
        "cutoff": cutoff,
        "sep_score": separation,
        "depth": node.get("depth"),
        "n_cells": n_cells,
    }]
    if node["type"] == "node":
        return pd.concat(
            [
                pd.DataFrame(rows),
                tree_to_df(node["left"], id, id + "_L"),
                tree_to_df(node["right"], id, id + "_R"),
            ],
            ignore_index=True,
        )
    return pd.DataFrame(rows)


def print_celltype_tree(node, indent=""):
    """Print a compact text representation of a gating tree.

    Parameters
    ----------
    node : dict
        Root node returned by :func:`cytogater.build_fullcoverage_tree`.
    indent : str, default=""
        Initial indentation, extended recursively for child nodes.

    Returns
    -------
    None
        The tree is written to standard output.
    """
    text = indent + (
        f"{node['marker']} > {node['cutoff']:.3g}"
        if node["type"] == "node" else "leaf"
    )
    print(text)
    if node["type"] == "node":
        print_celltype_tree(node["left"], indent + "  ")
        print_celltype_tree(node["right"], indent + "  ")


def plot_celltype_tree(tree, title="", ax=None):
    """Plot a top-down marker-priority gating tree.

    Parameters
    ----------
    tree : dict
        Root node returned by :func:`cytogater.build_fullcoverage_tree`.
    title : str, default=""
        Plot title.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the tree diagram.
    """
    ax = ax or _plt().subplots()[1]
    table = tree_to_df(tree).set_index("id", drop=False)
    positions = {}
    next_leaf = [0]

    def assign_positions(node, node_id="root", depth=0):
        if node["type"] == "leaf":
            x = float(next_leaf[0])
            next_leaf[0] += 1
        else:
            left_x = assign_positions(node["left"], node_id + "_L", depth + 1)
            right_x = assign_positions(node["right"], node_id + "_R", depth + 1)
            x = (left_x + right_x) / 2
        positions[node_id] = (x, -float(depth))
        return x

    assign_positions(tree)
    for row in table.itertuples():
        if row.parent is None:
            continue
        parent_x, parent_y = positions[row.parent]
        child_x, child_y = positions[row.id]
        middle_y = (parent_y + child_y) / 2
        ax.plot(
            [parent_x, parent_x, child_x, child_x],
            [parent_y, middle_y, middle_y, child_y],
            color="#4b5563",
            linewidth=0.8,
            zorder=1,
        )

    for row in table.itertuples():
        x, y = positions[row.id]
        ax.text(
            x,
            y,
            row.label,
            ha="center",
            va="center",
            fontsize=8,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#4b5563",
                "linewidth": 0.4,
            },
            zorder=2,
        )

    ax.set_title(title)
    ax.margins(x=0.12, y=0.18)
    ax.axis("off")
    return ax


def plot_marker_priority_tree(lineage_table, marker_stats, cell_type_name, ax=None):
    """Plot positive-marker priority weights for one cell type.

    Parameters
    ----------
    lineage_table : pandas.DataFrame
        Cell-type marker table.
    marker_stats : dict
        Per-marker statistics from :func:`cytogater.fit_marker_stats`.
    cell_type_name : str
        Cell type selected from ``lineage_table.cell_type``.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing marker separation weights.
    """
    row = pd.DataFrame(lineage_table).query("cell_type == @cell_type_name").iloc[0]
    values = [marker_stats[x]["weight"] for x in row.pos_markers]
    ax = ax or _plt().subplots()[1]
    ax.bar(row.pos_markers, values)
    ax.set(title=cell_type_name, ylabel="Separability weight")
    return ax


def plot_rand_cell_probs(spe=None, prob_mat=None, n=20, seed=None, ax=None):
    """Plot stacked probabilities for a random subset of cells.

    Parameters
    ----------
    spe : anndata.AnnData or None, default=None
        Accepted for R API compatibility; currently unused.
    prob_mat : pandas.DataFrame or array-like
        Cell-by-cell-type probability matrix.
    n : int, default=20
        Maximum cells sampled for display.
    seed : int or None, default=None
        Random seed used for sampling.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing stacked probability bars.
    """
    del spe
    frame = pd.DataFrame(prob_mat)
    sample = frame.sample(min(n, len(frame)), random_state=seed)
    ax = ax or _plt().subplots()[1]
    sample.plot.bar(stacked=True, ax=ax)
    return ax
