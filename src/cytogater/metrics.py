"""Label confidence and classification metrics."""

import numpy as np
import pandas as pd


def compute_custom_labels(
    prob_mat, base_thresh=0.4, quantile_cut=0.75, iqr_mult=1.5, flag_fn=None
):
    """Flag cells whose winning probability is unusually high.

    Parameters
    ----------
    prob_mat : array-like
        Cell-by-cell-type probability matrix.
    base_thresh : float, default=0.4
        Absolute lower bound for the winning probability.
    quantile_cut : float, default=0.75
        Within-cell probability quantile used in the adaptive cutoff.
    iqr_mult : float, default=1.5
        Number of within-cell interquartile ranges added to the quantile.
    flag_fn : callable or None, default=None
        Optional function receiving one probability row and returning a
        boolean. When supplied, it replaces the default cutoff rule.

    Returns
    -------
    numpy.ndarray
        Boolean confidence flag for every cell.
    """
    values = np.asarray(prob_mat, dtype=float)
    flags = []
    for row in values:
        finite = row[np.isfinite(row)]
        if flag_fn is not None:
            flags.append(bool(flag_fn(row)))
        elif not len(finite):
            flags.append(False)
        else:
            q = np.quantile(finite, quantile_cut)
            iqr = np.quantile(finite, 0.75) - np.quantile(finite, 0.25)
            flags.append(np.max(finite) > max(base_thresh, q + iqr_mult * iqr))
    return np.asarray(flags)


def assign_confident_labels(prob_mat, confidence_flags, unknown_label="Unknown"):
    """Assign winning cell types only to cells flagged as confident.

    Parameters
    ----------
    prob_mat : pandas.DataFrame
        Cell-by-cell-type probability matrix.
    confidence_flags : array-like of bool
        One confidence decision per cell.
    unknown_label : str, default="Unknown"
        Label used for unconfident rows or rows without finite probabilities.

    Returns
    -------
    numpy.ndarray
        One label per probability-matrix row.
    """
    values = np.asarray(prob_mat, dtype=float)
    columns = np.asarray(prob_mat.columns)
    safe = np.where(np.isfinite(values), values, -np.inf)
    labels = columns[safe.argmax(axis=1)].astype(object)
    labels[~np.isfinite(values).any(axis=1)] = unknown_label
    labels[~np.asarray(confidence_flags, dtype=bool)] = unknown_label
    return labels


def custom_labels(
    spe,
    prob_mat,
    colname="custom_label",
    base_thresh=0.4,
    quantile_cut=0.75,
    iqr_mult=1.5,
    unknown_label="Unknown",
    flag_fn=None,
):
    """Store confidence-filtered probability labels in an AnnData object.

    Parameters
    ----------
    spe : anndata.AnnData
        Object to annotate in place.
    prob_mat : pandas.DataFrame
        Cell-by-cell-type probability matrix.
    colname : str, default="custom_label"
        Destination column in ``spe.obs``.
    base_thresh, quantile_cut, iqr_mult : float
        Confidence-rule parameters passed to :func:`compute_custom_labels`.
    unknown_label : str, default="Unknown"
        Label assigned to cells that do not pass the confidence rule.
    flag_fn : callable or None, default=None
        Optional custom confidence function.

    Returns
    -------
    anndata.AnnData
        The same object, with ``spe.obs[colname]`` added.
    """
    flags = compute_custom_labels(
        prob_mat, base_thresh, quantile_cut, iqr_mult, flag_fn
    )
    spe.obs[colname] = assign_confident_labels(prob_mat, flags, unknown_label)
    return spe


def class_metrics_from_fit(fit, test_truth=None, test_pred=None):
    """Calculate per-class precision, recall, and F1 scores.

    Parameters
    ----------
    fit : dict
        Model result. If it contains ``cv_class_metrics``, that table is
        returned directly; otherwise ``test_truth`` and ``test_pred`` may be
        read from it.
    test_truth, test_pred : array-like or None, default=None
        Optional true and predicted labels overriding values in ``fit``.

    Returns
    -------
    pandas.DataFrame
        One row per class with counts, support, precision, recall, and F1.

    Raises
    ------
    ValueError
        If neither explicit labels nor labels in ``fit`` are available.
    """
    if "cv_class_metrics" in fit:
        return fit["cv_class_metrics"]
    truth = test_truth if test_truth is not None else fit.get("test_truth")
    pred = test_pred if test_pred is not None else fit.get("test_pred")
    if truth is None or pred is None:
        raise ValueError("Provide test_truth/test_pred or metrics in fit.")
    truth, pred = np.asarray(truth).astype(str), np.asarray(pred).astype(str)
    rows = []
    for cls in sorted(set(truth) | set(pred)):
        tp = int(np.sum((truth == cls) & (pred == cls)))
        fp = int(np.sum((truth != cls) & (pred == cls)))
        fn = int(np.sum((truth == cls) & (pred != cls)))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        rows.append({
            "class": cls, "tp": tp, "fp": fp, "fn": fn,
            "support": int(np.sum(truth == cls)), "precision": precision,
            "recall": recall, "f1": f1,
        })
    return pd.DataFrame(rows)


def calculate_f1(spe, ref_col="ref_broad", pred_col="pred_broad"):
    """Summarise annotation precision, recall, and F1 from AnnData labels.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated data object.
    ref_col : str, default="ref_broad"
        ``spe.obs`` column containing reference labels.
    pred_col : str, default="pred_broad"
        ``spe.obs`` column containing predicted labels.

    Returns
    -------
    pandas.DataFrame
        Per-class metrics excluding unassigned, undefined, and unknown labels.
    """
    table = class_metrics_from_fit(
        {}, spe.obs[ref_col].astype(str), spe.obs[pred_col].astype(str)
    )
    table = table[~table["class"].str.contains(
        "unassigned|undefined|unknown", case=False, regex=True
    )]
    return pd.DataFrame({
        "Category": table["class"],
        "Precision": table.precision.round(3),
        "Recall": table.recall.round(3),
        "F1_Score": table.f1.round(3),
        "Cell_Count": table.support,
    }).reset_index(drop=True)


def label_confusion_matrix(spe, label_col1, label_col2, drop_na=True):
    """Cross-tabulate two label columns using their combined label set.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated data object.
    label_col1, label_col2 : str
        Names of the two ``spe.obs`` label columns.
    drop_na : bool, default=True
        Remove rows missing either label before tabulation.

    Returns
    -------
    pandas.DataFrame
        Square confusion/count matrix with identical row and column levels.
    """
    frame = spe.obs[[label_col1, label_col2]]
    if drop_na:
        frame = frame.dropna()
    levels = sorted(set(frame[label_col1].astype(str)) | set(frame[label_col2].astype(str)))
    return pd.crosstab(
        pd.Categorical(frame[label_col1], levels),
        pd.Categorical(frame[label_col2], levels),
        dropna=False,
    )


def label_agreement_rates(spe, label_col1, label_col2, drop_na=True):
    """Calculate per-label intersection-over-union agreement.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated data object.
    label_col1, label_col2 : str
        Names of the two ``spe.obs`` label columns.
    drop_na : bool, default=True
        Remove rows missing either label.

    Returns
    -------
    pandas.DataFrame
        Label, exact-match count, union count, and agreement rate.
    """
    frame = spe.obs[[label_col1, label_col2]]
    if drop_na:
        frame = frame.dropna()
    a, b = frame[label_col1].astype(str), frame[label_col2].astype(str)
    rows = []
    for label in sorted(set(a) | set(b)):
        match = int(((a == label) & (b == label)).sum())
        union = int(((a == label) | (b == label)).sum())
        rows.append({
            "label": label, "match_n": match, "union_n": union,
            "agreement": match / union if union else np.nan,
        })
    return pd.DataFrame(rows)


def prob_quantile_cutoff(x, prob=0.98):
    """Calculate a finite-value probability quantile cutoff.

    Parameters
    ----------
    x : array-like
        Probability values.
    prob : float, default=0.98
        Quantile between zero and one.

    Returns
    -------
    float
        Requested quantile, ignoring missing values.
    """
    return float(np.nanquantile(x, prob))


def prob_mad_cutoff(x, mad_mult=3):
    """Calculate a median-plus-MAD probability cutoff.

    Parameters
    ----------
    x : array-like
        Probability values.
    mad_mult : float, default=3
        Number of unscaled median absolute deviations above the median.

    Returns
    -------
    float
        Robust probability cutoff.
    """
    x = np.asarray(x, dtype=float)
    median = np.nanmedian(x)
    # R uses stats::mad(..., constant = 1) in prob_mad_cutoff().
    mad = np.nanmedian(np.abs(x - median))
    return float(median + mad_mult * mad)


def probability_label_matrix(prob_mat, cutoff_fn=prob_quantile_cutoff):
    """Threshold every cell-type probability column independently.

    Parameters
    ----------
    prob_mat : pandas.DataFrame or array-like
        Cell-by-cell-type probability matrix.
    cutoff_fn : callable, default=prob_quantile_cutoff
        Function mapping one probability column to a scalar cutoff.

    Returns
    -------
    pandas.DataFrame
        Nullable boolean matrix indicating which cell-type cutoffs each cell
        exceeds. Missing input values and invalid cutoffs remain missing.
    """
    frame = pd.DataFrame(prob_mat)
    cutoffs = frame.apply(lambda column: cutoff_fn(column))
    cutoffs[~np.isfinite(cutoffs)] = np.nan
    label_matrix = frame.ge(cutoffs, axis="columns").astype("boolean")
    missing = frame.isna()
    missing.loc[:, cutoffs.isna()] = True
    return label_matrix.mask(missing)


def apply_cutoff_labels(
    res,
    cutoff_fn=prob_quantile_cutoff,
    label_col="cutoff_label",
    unknown_label="Unknown",
):
    """Assign labels to cells that exceed exactly one probability cutoff.

    Parameters
    ----------
    res : dict
        Gating result containing ``prob_mat`` and ``spe``.
    cutoff_fn : callable, default=prob_quantile_cutoff
        Per-cell-type cutoff function.
    label_col : str, default="cutoff_label"
        Destination column in ``res["spe"].obs``.
    unknown_label : str, default="Unknown"
        Label for cells exceeding zero or multiple cutoffs.

    Returns
    -------
    dict
        The input result with annotated ``spe`` and a nullable boolean
        ``label_mat`` entry added.
    """
    labels = probability_label_matrix(res["prob_mat"], cutoff_fn)
    positive_counts = labels.sum(axis=1, skipna=True)
    names = pd.Series(unknown_label, index=labels.index, dtype=object)
    single = positive_counts == 1
    if single.any():
        names.loc[single] = labels.loc[single].fillna(False).idxmax(axis=1)
    res["spe"].obs[label_col] = names.reindex(res["spe"].obs_names).to_numpy()
    res["label_mat"] = labels
    return res
