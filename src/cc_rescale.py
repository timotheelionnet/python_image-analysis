# cc_rescale.py

# These functions re-scale Hoechst and EdU values (independently) 
# for cell-cycle analysis

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


__all__ = ["rescale_hoechst", "rescale_edu"]


def rescale_hoechst(
    df: pd.DataFrame,
    hoechst_col: str = "ch1_sum_int",
    group_cols = ("experiment", "plasmid"),
    method: str = "biological",
    *,
    min_group_n: int = 20,
    log_eps: float = 1e-9,
    random_state: int = 0,
    return_params: bool = False,
):
    """
    Rescale Hoechst per group so G1≈1 and G2≈2 ('biological') or z-score per group.

    Parameters
    ----------
    df : DataFrame
        Must contain hoechst_col and group_cols.
    hoechst_col : str
        Column with Hoechst intensity (linear scale).
    group_cols : str | list/tuple of str
        Columns that define groups (e.g., ('experiment','plasmid')).
    method : {'biological', 'zscore'}
        'biological' maps (G1 -> 1, G2 -> 2) via 2-component GMM in log10 space.
        'zscore' does (x - mean)/std in log10 space.
    min_group_n : int
        Minimum cells required to attempt GMM; otherwise fallback to robust quantile mapping.
    log_eps : float
        Small offset added before log10 to avoid log(0).
    random_state : int
        Random seed for GMM.
    return_params : bool
        If True, also returns a DataFrame of per-group parameters used.

    Returns
    -------
    scaled_series : pd.Series
        Scaled values aligned to df.index (name f'{hoechst_col}_scaled').
    params_df (optional) : pd.DataFrame
        Per-group parameters (mu_G1, mu_G2, method_used, fallback_flag).
    """

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Precompute log10 safely
    log_vals = np.log10(df[hoechst_col].values + log_eps)
    out = pd.Series(index=df.index, dtype=float, name=f"{hoechst_col}_scaled")

    param_rows = []

    def _robust_mus(x):
        q25, q75 = np.nanquantile(x, [0.25, 0.75])
        if np.isclose(q25, q75):
            q25, q75 = np.nanmin(x), np.nanmax(x)
        return float(min(q25, q75)), float(max(q25, q75))

    # Work group-by-group
    gobj = df.groupby(list(group_cols), sort=False)
    for keys, idx in gobj.groups.items():
        # Ensure keys is always a tuple so we can safely unpack later
        key_tuple = keys if isinstance(keys, tuple) else (keys,)

        idx = list(idx)
        x = log_vals[idx]

        used = method
        fallback = False
        mu1 = mu2 = np.nan
        scaled = np.full_like(x, np.nan, dtype=float)

        if np.all(~np.isfinite(x)) or np.isclose(np.nanstd(x), 0.0):
            param_rows.append(tuple(key_tuple) + (mu1, mu2, used, True, len(x)))
            out.iloc[idx] = scaled
            continue

        if method == "zscore":
            m, s = np.nanmean(x), np.nanstd(x)
            if s == 0 or not np.isfinite(s):
                fallback = True
            else:
                scaled = (x - m) / s

        elif method == "biological":
            if len(x) >= min_group_n and np.isfinite(x).sum() >= 2:
                try:
                    gm = GaussianMixture(n_components=2, random_state=random_state)
                    gm.fit(x.reshape(-1, 1))
                    mus = np.sort(gm.means_.flatten())
                    mu1, mu2 = float(mus[0]), float(mus[1])
                    if np.isclose(mu1, mu2):
                        raise ValueError("GMM means collapsed")
                    scaled = 1 + (x - mu1) / (mu2 - mu1)
                except Exception:
                    mu1, mu2 = _robust_mus(x)
                    fallback = True
                    if np.isclose(mu1, mu2):
                        scaled[:] = np.nan
                    else:
                        scaled = 1 + (x - mu1) / (mu2 - mu1)
            else:
                mu1, mu2 = _robust_mus(x)
                fallback = True
                if np.isclose(mu1, mu2):
                    scaled[:] = np.nan
                else:
                    scaled = 1 + (x - mu1) / (mu2 - mu1)
        else:
            raise ValueError("method must be 'biological' or 'zscore'")

        out.iloc[idx] = scaled
        param_rows.append(tuple(key_tuple) + (mu1, mu2, used, fallback, len(x)))

    if return_params:
        params_cols = list(group_cols) + ["mu_G1", "mu_G2", "method_used", "fallback_flag", "n"]
        params_df = pd.DataFrame(param_rows, columns=params_cols)
        return out, params_df

    return out

def rescale_edu(
    df: pd.DataFrame,
    edu_col: str = "ch4_sum_int",
    group_cols = ("experiment", "plasmid"),
    method: str = "biological",  # 'biological' | 'zscore' | 'minmax'
    *,
    min_group_n: int = 20,
    log_eps: float = 1e-9,
    random_state: int = 0,
    return_params: bool = False,
    clip_01: bool = False,   # keep values in [0,1] for biological/minmax
):
    """
    Rescale EdU per group for cross-experiment comparability.

    Methods
    -------
    biological: Map non-S -> 0 and S -> 1 via 2-comp GMM on log10(EdU+eps).
                Fallback to robust quantiles q10->0, q90->1.
    zscore:     (x - mean)/std in log10 space.
    minmax:     Robust min-max in log10 space using q1->0, q99->1.

    Returns
    -------
    scaled_series : pd.Series (name f'{edu_col}_scaled')
    params_df (optional): summary per group (mu_low, mu_high or q_low, q_high, etc.)
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    x_log = np.log10(df[edu_col].values + log_eps)
    out = pd.Series(index=df.index, dtype=float, name=f"{edu_col}_scaled")
    param_rows = []

    def _robust_pair(x, qlo=0.10, qhi=0.90):
        lo, hi = np.nanquantile(x, [qlo, qhi])
        if np.isclose(lo, hi):
            lo, hi = np.nanmin(x), np.nanmax(x)
        return float(min(lo, hi)), float(max(lo, hi))

    gobj = df.groupby(list(group_cols), sort=False)
    for keys, idx in gobj.groups.items():
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        idx = list(idx)
        x = x_log[idx]

        scaled = np.full_like(x, np.nan, dtype=float)
        used = method
        fallback = False

        # default params placeholders:
        mu_low = mu_high = np.nan
        q_low = q_high = np.nan
        m = s = np.nan

        # Degenerate / tiny-variance guard
        if np.all(~np.isfinite(x)) or np.isclose(np.nanstd(x), 0.0):
            param_rows.append(tuple(key_tuple) + (used, True, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, len(x)))
            out.iloc[idx] = scaled
            continue

        if method == "zscore":
            m, s = np.nanmean(x), np.nanstd(x)
            if s == 0 or not np.isfinite(s):
                fallback = True
            else:
                scaled = (x - m) / s

        elif method == "minmax":
            q_low, q_high = _robust_pair(x, 0.01, 0.99)
            if np.isclose(q_low, q_high):
                fallback = True
            else:
                scaled = (x - q_low) / (q_high - q_low)
                if clip_01:
                    scaled = np.clip(scaled, 0.0, 1.0)

        elif method == "biological":
            # GMM to separate non-S (low) vs S (high)
            if len(x) >= min_group_n and np.isfinite(x).sum() >= 2:
                try:
                    gm = GaussianMixture(n_components=2, random_state=random_state)
                    gm.fit(x.reshape(-1, 1))
                    mus = np.sort(gm.means_.flatten())
                    mu_low, mu_high = float(mus[0]), float(mus[1])
                    if np.isclose(mu_low, mu_high):
                        raise ValueError("GMM means collapsed")
                    scaled = (x - mu_low) / (mu_high - mu_low)
                    if clip_01:
                        scaled = np.clip(scaled, 0.0, 1.0)
                except Exception:
                    # Fallback to quantiles q10->0, q90->1
                    q_low, q_high = _robust_pair(x, 0.10, 0.90)
                    fallback = True
                    if np.isclose(q_low, q_high):
                        scaled[:] = np.nan
                    else:
                        scaled = (x - q_low) / (q_high - q_low)
                        if clip_01:
                            scaled = np.clip(scaled, 0.0, 1.0)
            else:
                q_low, q_high = _robust_pair(x, 0.10, 0.90)
                fallback = True
                if np.isclose(q_low, q_high):
                    scaled[:] = np.nan
                else:
                    scaled = (x - q_low) / (q_high - q_low)
                    if clip_01:
                        scaled = np.clip(scaled, 0.0, 1.0)
        else:
            raise ValueError("method must be 'biological', 'zscore', or 'minmax'")

        out.iloc[idx] = scaled
        # Store params. For clarity we record both sets; unused remain NaN.
        param_rows.append(tuple(key_tuple) + (
            used, fallback, mu_low, mu_high, q_low, q_high, m, s, len(x)
        ))

    if return_params:
        cols = list(group_cols) + [
            "method_used", "fallback_flag",
            "mu_low", "mu_high",         # biological (GMM)
            "q_low", "q_high",           # biological fallback or minmax
            "mean", "std",               # zscore
            "n"
        ]
        params_df = pd.DataFrame(param_rows, columns=cols)
        return out, params_df

    return out