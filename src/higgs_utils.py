"""Helper functions for the Higgs Boson ML challenge project.

Everything here is stateless (no information is learned from the data), so it can be
used inside cross-validation without leaking information between folds.
"""
import numpy as np
import pandas as pd

MISSING = -999.0
BR = 10.0  # regularisation constant of the competition metric
ID_COL, WEIGHT_COL, LABEL_COL = "EventId", "Weight", "Label"
FRACS = np.round(np.arange(0.01, 0.60, 0.005), 3)  # grid of "fraction of events called signal"


# ----------------------------------------------------------------------------- metric
def ams(s, b, br=BR):
    """Approximate Median Significance (vectorised). s, b = weighted signal / background selected."""
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    radicand = 2.0 * ((s + b + br) * np.log1p(s / (b + br)) - s)
    return np.sqrt(np.maximum(radicand, 0.0))


def ams_from_labels(y_true, y_pred, w):
    """AMS for hard predictions. y_true / y_pred are 0/1 arrays (1 = signal), w = event weights."""
    y_true, y_pred, w = map(np.asarray, (y_true, y_pred, w))
    sel = y_pred == 1
    s = w[sel & (y_true == 1)].sum()
    b = w[sel & (y_true == 0)].sum()
    return float(ams(s, b))


def ams_curve(y, score, w, fracs=FRACS):
    """AMS obtained by calling the top `frac` of events (highest score) signal, for every frac."""
    y, score, w = map(np.asarray, (y, score, w))
    order = np.argsort(-score, kind="stable")
    ys, ws = y[order], w[order]
    cs = np.cumsum(ws * (ys == 1))
    cb = np.cumsum(ws * (ys == 0))
    k = np.clip((np.asarray(fracs) * len(y)).astype(int) - 1, 0, len(y) - 1)
    return ams(cs[k], cb[k]), score[order][k]


def best_threshold(y, score, w, fracs=FRACS, smooth=1):
    """Best AMS over the grid. smooth=1 -> argmax of a +-1-step moving average (less noisy)."""
    curve, thr = ams_curve(y, score, w, fracs)
    sm = curve.copy()
    if smooth:
        pad = np.pad(curve, smooth, mode="edge")
        sm = np.convolve(pad, np.ones(2 * smooth + 1) / (2 * smooth + 1), mode="valid")
    i = int(np.argmax(sm))
    return {"ams": float(curve[i]), "frac": float(fracs[i]), "thr": float(thr[i]),
            "ams_max_raw": float(curve.max())}


# ----------------------------------------------------------------------------- data
def load_train(path):
    df = pd.read_csv(path)
    y = (df[LABEL_COL] == "s").astype(int).to_numpy()
    w = df[WEIGHT_COL].to_numpy()
    X = df.drop(columns=[ID_COL, WEIGHT_COL, LABEL_COL])
    return df[ID_COL].to_numpy(), X, y, w


def load_test(path):
    df = pd.read_csv(path)
    return df[ID_COL].to_numpy(), df.drop(columns=[ID_COL])


def to_nan(X):
    """Replace the special -999 marker by NaN (stateless)."""
    return pd.DataFrame(X).replace(MISSING, np.nan)


def class_balanced_weights(y, w):
    """Rescale event weights so that signal and background have the same total weight.
    The relative weights inside each class are preserved."""
    y, w = np.asarray(y), np.asarray(w, dtype=float)
    out = w.copy()
    n = len(y)
    for c in (0, 1):
        out[y == c] = w[y == c] / w[y == c].sum() * (n / 2.0)
    return out


def _wrap(a):
    """Wrap an angle difference into [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def add_features(X):
    """Physics-motivated extra features. Input: raw DataFrame (may contain -999). Output: DataFrame
    with NaN for undefined values. Stateless, therefore safe inside cross-validation."""
    X = to_nan(X)
    F = X.copy()
    F["dphi_tau_lep"] = np.abs(_wrap(X["PRI_tau_phi"] - X["PRI_lep_phi"]))
    F["dphi_tau_met"] = np.abs(_wrap(X["PRI_tau_phi"] - X["PRI_met_phi"]))
    F["dphi_lep_met"] = np.abs(_wrap(X["PRI_lep_phi"] - X["PRI_met_phi"]))
    F["deta_tau_lep"] = np.abs(X["PRI_tau_eta"] - X["PRI_lep_eta"])
    F["met_over_tau_pt"] = X["PRI_met"] / (X["PRI_tau_pt"] + 1e-6)
    F["met_over_sumet"] = X["PRI_met"] / (X["PRI_met_sumet"] + 1e-6)
    F["mmc_over_vis"] = X["DER_mass_MMC"] / (X["DER_mass_vis"] + 1e-6)
    F["mmc_minus_vis"] = X["DER_mass_MMC"] - X["DER_mass_vis"]
    F["mt_over_vis"] = X["DER_mass_transverse_met_lep"] / (X["DER_mass_vis"] + 1e-6)
    F["pth_over_sumpt"] = X["DER_pt_h"] / (X["DER_sum_pt"] + 1e-6)
    F["jet_pt_ratio"] = X["PRI_jet_subleading_pt"] / (X["PRI_jet_leading_pt"] + 1e-6)
    F["lep_tau_pt_sum"] = X["PRI_lep_pt"] + X["PRI_tau_pt"]
    return F


def drop_raw_phi(X):
    """Remove absolute azimuth angles (physically uninformative: the detector is rotation-symmetric)."""
    X = pd.DataFrame(X)
    return X.drop(columns=[c for c in X.columns if c.endswith("_phi")])


def add_features_no_phi(X):
    return drop_raw_phi(add_features(X))
