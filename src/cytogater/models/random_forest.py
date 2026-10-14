"""Random-forest consensus cleaning and unknown-cell prediction."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from ..preprocessing import expression_frame


def train_custom_randomforest(
    spe,
    label_col="cutoff_label",
    assay_name="exprs",
    unknown_label="Unknown",
    num_trees=200,
    mtry=None,
    seed=None,
    features="all",
    cv_folds=5,
    repeats=10,
    agreement_thresh=0.8,
    num_threads=2,
    **kwargs,
):
    """Train the R-equivalent probability random forest.

    ``num.trees`` may be supplied through ``kwargs`` for direct R-call migration.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated cells-by-markers object, modified in place.
    label_col : str, default="cutoff_label"
        ``spe.obs`` column containing initial reference labels.
    assay_name : str or None, default="exprs"
        Expression layer, or ``None``/``"X"`` for ``spe.X``.
    unknown_label : str, default="Unknown"
        Label excluded from reference training.
    num_trees : int, default=200
        Trees in the final random forest.
    mtry : int, float, str, or None, default=None
        Features considered at each split; ``None`` uses ``"sqrt"``.
    seed : int or None, default=None
        Random seed for cross-validation and model fitting.
    features : "all" or sequence of str, default="all"
        Marker features used by the model.
    cv_folds : int, default=5
        Requested stratified folds, capped by the smallest eligible class.
    repeats : int, default=10
        Repeated cross-validation rounds.
    agreement_thresh : float, default=0.8
        Minimum correct out-of-fold fraction required to retain a reference.
    num_threads : int, default=2
        Parallel threads used by scikit-learn.
    **kwargs
        Compatibility arguments. ``num.trees`` overrides ``num_trees``.

    Returns
    -------
    dict
        Annotated ``spe``, fitted model, retained-cell probabilities,
        agreement rates, and feature names. Cleaned labels are stored in
        ``spe.obs["cleaned_core_label"]``.

    Raises
    ------
    ValueError
        If no label has at least two reference cells.
    """
    num_trees = kwargs.pop("num.trees", num_trees)
    frame = expression_frame(spe, assay_name)
    features_used = (
        list(frame.columns)
        if isinstance(features, str) and features == "all"
        else [x for x in features if x in frame]
    )
    labels = spe.obs[label_col]
    core = labels.notna() & (labels != unknown_label)
    x, y = frame.loc[core, features_used], labels[core].astype(str)
    class_counts = y.value_counts()
    eligible = y.isin(class_counts[class_counts >= 2].index).to_numpy()
    if not eligible.any():
        raise ValueError(
            "Random-forest consensus cleaning needs at least one label "
            "with two or more cells."
        )
    x_cv, y_cv = x.iloc[eligible], y.iloc[eligible]
    folds = min(cv_folds, int(y_cv.value_counts().min()))
    cv = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=seed
    )
    cleaner = RandomForestClassifier(
        n_estimators=100, max_features=mtry or "sqrt",
        n_jobs=num_threads, random_state=seed,
    )
    classes = np.unique(y_cv)
    probabilities = np.zeros((len(x), len(classes)))
    counts = np.zeros(len(x))
    match_counts = np.zeros(len(x))
    eligible_positions = np.flatnonzero(eligible)
    for train, test in cv.split(x_cv, y_cv):
        cleaner.fit(x_cv.iloc[train], y_cv.iloc[train])
        fold = cleaner.predict_proba(x_cv.iloc[test])
        positions = [np.flatnonzero(classes == value)[0] for value in cleaner.classes_]
        original_test = eligible_positions[test]
        probabilities[np.ix_(original_test, positions)] += fold
        counts[original_test] += 1
        fold_labels = cleaner.classes_[fold.argmax(axis=1)]
        match_counts[original_test] += (
            fold_labels == y_cv.iloc[test].to_numpy()
        )
    probabilities[eligible] /= counts[eligible, None]
    # R increments a match counter for every repeat's out-of-fold hard
    # prediction, then divides by repeats. Using only the winner of averaged
    # probabilities incorrectly collapses agreement to 0 or 1.
    rates = np.zeros(len(x))
    rates[eligible] = match_counts[eligible] / counts[eligible]
    retained = eligible & (rates >= agreement_thresh)
    cleaned = labels.copy()
    cleaned.loc[x.index[~retained]] = unknown_label
    spe.obs["cleaned_core_label"] = cleaned
    model = RandomForestClassifier(
        n_estimators=num_trees, max_features=mtry or "sqrt",
        n_jobs=num_threads, random_state=seed,
    ).fit(x.loc[retained], y.loc[retained])
    model.cytogater_features_ = features_used
    return {
        "spe": spe, "model": model,
        "core_prob_mat": pd.DataFrame(
            probabilities[retained], index=x.index[retained], columns=classes
        ),
        "agreement_rates": pd.Series(rates[retained], index=x.index[retained]),
        "features_used": features_used,
    }


def predict_unknown_with_randomforest(
    spe,
    model,
    assay_name="exprs",
    label_col="custom_label",
    out_col="soft_tree_label_filled",
    pred_col="rf_pred",
    unknown_label="Unknown",
    threshold=0.5,
    unassigned_label="Unassigned",
):
    """Fill unknown labels using a trained random forest.

    Parameters
    ----------
    spe : anndata.AnnData
        Object containing cells to predict; modified in place.
    model : sklearn.ensemble.RandomForestClassifier
        Model returned in ``train_custom_randomforest(...)["model"]``.
    assay_name : str or None, default="exprs"
        Expression layer, or ``None``/``"X"`` for ``spe.X``.
    label_col : str, default="custom_label"
        Existing labels; only missing or ``unknown_label`` cells are replaced.
    out_col : str, default="soft_tree_label_filled"
        Destination for combined existing and predicted labels.
    pred_col : str, default="rf_pred"
        Destination for predictions on replaced cells only.
    unknown_label : str, default="Unknown"
        Label marking cells that require prediction.
    threshold : float, default=0.5
        Minimum winning class probability.
    unassigned_label : str, default="Unassigned"
        Label used below the confidence threshold.

    Returns
    -------
    dict
        Annotated ``spe`` and a probability matrix for predicted cells.
        ``prob_mat`` is ``None`` when no cells require prediction.
    """
    frame = expression_frame(spe, assay_name)
    # AnnData commonly round-trips string observation columns as Categoricals.
    # Prediction may introduce a model class or ``Unassigned`` that is not an
    # existing category, so perform replacement on an object-backed Series.
    labels = spe.obs[label_col].astype(object).copy()
    replace = labels.isna() | (labels == unknown_label)
    if not replace.any():
        return {"spe": spe, "prob_mat": None}
    features = getattr(model, "cytogater_features_", list(frame.columns))
    values = model.predict_proba(frame.loc[replace, features])
    probabilities = pd.DataFrame(
        values, index=frame.index[replace], columns=model.classes_
    )
    confidence = probabilities.max(axis=1)
    predicted = probabilities.idxmax(axis=1).where(
        confidence >= threshold, unassigned_label
    )
    spe.obs[pred_col] = pd.Series(index=spe.obs_names, dtype=object)
    spe.obs.loc[replace, pred_col] = predicted
    labels.loc[replace] = predicted
    spe.obs[out_col] = labels
    spe.obs["rf_confidence"] = pd.Series(confidence, index=confidence.index)
    return {"spe": spe, "prob_mat": probabilities}


def rf_metric_table(fit):
    """Extract per-class classification metrics from a model result.

    Parameters
    ----------
    fit : dict
        Result containing cached cross-validation metrics or true/predicted
        test labels accepted by :func:`cytogater.class_metrics_from_fit`.

    Returns
    -------
    dict
        Mapping with a ``class`` data frame of precision, recall, and F1.
    """
    from ..metrics import class_metrics_from_fit
    class_table = class_metrics_from_fit(fit)
    return {"class": class_table}


def rf_metric_text(fit, digits=3):
    """Format random-forest class metrics as readable text lines.

    Parameters
    ----------
    fit : dict
        Model result accepted by :func:`rf_metric_table`.
    digits : int, default=3
        Decimal places used for metric values.

    Returns
    -------
    list of str
        One precision/recall/F1 summary per class.
    """
    table = rf_metric_table(fit)["class"]
    return [
        f"{row['class']}: precision={row.precision:.{digits}f}, "
        f"recall={row.recall:.{digits}f}, f1={row.f1:.{digits}f}"
        for _, row in table.iterrows()
    ]
