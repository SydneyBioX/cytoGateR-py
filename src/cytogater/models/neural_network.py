"""Optional PyTorch classifier with the R package's public workflow."""

import numpy as np
import pandas as pd

from ..preprocessing import expression_frame


def train_custom_dl(
    spe,
    label_col="cutoff_label",
    assay_name="exprs",
    unknown_label="Unknown",
    features="all",
    hidden_dims=(64, 32),
    dropout=0.3,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    cv_folds=5,
    repeats=3,
    agreement_thresh=0.8,
    val_frac=0.15,
    patience=10,
    verbose_training=False,
    class_balanced_loss=True,
    seed=None,
    **kwargs,
):
    """Train a small feed-forward probability network.

    PyTorch is an optional dependency and is imported only when this function
    is called.

    Parameters
    ----------
    spe : anndata.AnnData
        Annotated cells-by-markers object.
    label_col : str, default="cutoff_label"
        ``spe.obs`` column containing training labels.
    assay_name : str or None, default="exprs"
        Expression layer, or ``None``/``"X"`` for ``spe.X``.
    unknown_label : str, default="Unknown"
        Label excluded from training.
    features : "all" or sequence of str, default="all"
        Marker features used by the network.
    hidden_dims : sequence of int, default=(64, 32)
        Width of each hidden layer.
    dropout : float, default=0.3
        Dropout probability after each hidden activation.
    epochs : int, default=50
        Training epochs.
    lr : float, default=1e-3
        Adam learning rate.
    weight_decay : float, default=1e-4
        Adam L2 weight decay.
    batch_size : int, default=256
        Mini-batch size.
    cv_folds, repeats, agreement_thresh, val_frac, patience
        Accepted for compatibility with the R workflow; currently unused.
    verbose_training : bool, default=False
        Print the loss after each epoch.
    class_balanced_loss : bool, default=True
        Weight cross-entropy inversely by class frequency.
    seed : int or None, default=None
        PyTorch random seed.
    **kwargs
        Additional options retained in the returned model metadata.

    Returns
    -------
    dict
        Input ``spe``, fitted network metadata, and feature names. The model
        mapping includes the network, class levels, scaling values, and device.
    """
    import torch
    from torch import nn

    if seed is not None:
        torch.manual_seed(seed)
    frame = expression_frame(spe, assay_name)
    features_used = (
        list(frame.columns)
        if isinstance(features, str) and features == "all"
        else [x for x in features if x in frame]
    )
    labels = spe.obs[label_col]
    keep = labels.notna() & (labels != unknown_label)
    x = frame.loc[keep, features_used].to_numpy(np.float32)
    classes, y = np.unique(labels[keep].astype(str), return_inverse=True)
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    x = (x - mean) / scale
    layers, width = [], x.shape[1]
    for hidden in hidden_dims:
        layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
        width = hidden
    layers.append(nn.Linear(width, len(classes)))
    del cv_folds, repeats, agreement_thresh, val_frac, patience
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if class_balanced_loss:
        counts = np.bincount(y)
        weights = len(y) / (len(counts) * counts)
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    else:
        loss_fn = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x), torch.tensor(y, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        if verbose_training:
            print(f"epoch {_ + 1}/{epochs}: loss={loss.item():.4f}")
    fitted = {
        "net": model, "class_levels": classes, "feature_names": features_used,
        "center": mean, "scale": scale, "device": device,
        "training_options": kwargs,
    }
    return {"spe": spe, "model": fitted, "features_used": features_used}


def predict_unknown_with_dl(
    spe,
    model,
    assay_name="exprs",
    label_col="custom_label",
    out_col="soft_tree_label_filled",
    pred_col="dl_pred",
    unknown_label="Unknown",
    threshold=0.5,
    unassigned_label="Unassigned",
):
    """Fill unknown labels using a trained PyTorch network.

    Parameters
    ----------
    spe : anndata.AnnData
        Object containing cells to predict; modified in place.
    model : dict
        Network metadata returned in ``train_custom_dl(...)["model"]``.
    assay_name : str or None, default="exprs"
        Expression layer, or ``None``/``"X"`` for ``spe.X``.
    label_col : str, default="custom_label"
        Existing labels; only missing or ``unknown_label`` cells are replaced.
    out_col : str, default="soft_tree_label_filled"
        Destination for combined existing and predicted labels.
    pred_col : str, default="dl_pred"
        Destination for predictions on replaced cells only.
    unknown_label : str, default="Unknown"
        Label marking cells that require prediction.
    threshold : float, default=0.5
        Minimum winning softmax probability.
    unassigned_label : str, default="Unassigned"
        Label used below the confidence threshold.

    Returns
    -------
    dict
        Annotated ``spe`` and a probability matrix for predicted cells.
        ``prob_mat`` is ``None`` when no cells require prediction.
    """
    import torch

    frame = expression_frame(spe, assay_name)
    # Avoid pandas rejecting predicted classes absent from a categorical
    # label column's currently registered categories.
    labels = spe.obs[label_col].astype(object).copy()
    replace = labels.isna() | (labels == unknown_label)
    if not replace.any():
        return {"spe": spe, "prob_mat": None}
    x = frame.loc[replace, model["feature_names"]].to_numpy(np.float32)
    x = (x - model["center"]) / model["scale"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model["net"] = model["net"].to(device)
    model["device"] = device
    model["net"].eval()
    with torch.no_grad():
        probability = torch.softmax(
            model["net"](torch.tensor(x).to(device)), dim=1
        ).cpu().numpy()
    probs = pd.DataFrame(
        probability, index=frame.index[replace], columns=model["class_levels"]
    )
    confidence = probs.max(axis=1)
    predicted = probs.idxmax(axis=1).where(confidence >= threshold, unassigned_label)
    spe.obs[pred_col] = pd.Series(index=spe.obs_names, dtype=object)
    spe.obs.loc[replace, pred_col] = predicted
    labels.loc[replace] = predicted
    spe.obs[out_col] = labels
    spe.obs["dl_confidence"] = pd.Series(confidence, index=confidence.index)
    return {"spe": spe, "prob_mat": probs}
