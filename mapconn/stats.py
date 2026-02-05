from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd

from .constants import STATS


def _remove_global(mapconn_curves: pd.DataFrame) -> pd.DataFrame:
    """
    Remove global connectivity (= value at 0th percentile) from mapconn curves.
    """

    df = (
        mapconn_curves.T.groupby("map", sort=False, group_keys=False)
        .apply(lambda x: x - x.values[0, :])
        .T
    )
    return df


def _calc_mapconn_stats(
    mapconn_curves: pd.DataFrame,
    stats: Union[str, Sequence[str]] = "all",
    remove_global: bool = True,
    force_dict: bool = False,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Calculate statistics from mapconn curves.
    """

    # checks
    if not isinstance(mapconn_curves, pd.DataFrame):
        raise ValueError("mapconn_curves must be a pandas DataFrame")
    if stats == "all":
        stats = STATS
    if isinstance(stats, str):
        stats = [stats]
    if not isinstance(stats, list):
        raise ValueError(f"stats must be a string or list of strings, got {type(stats)}")
    if not all(stat in STATS for stat in stats):
        raise ValueError(f"stats must be one or more of {STATS}, got {stats}")

    # maps
    maps = mapconn_curves.columns.get_level_values(0).unique()

    # map-wise curves
    curves_mapwise = {m: mapconn_curves.loc[:, m] for m in maps}

    # ids
    ids = mapconn_curves.index

    # percentiles
    percentiles = np.array(mapconn_curves.columns.get_level_values(-1).unique())
    percentiles = (percentiles / 100).astype(float)

    # calculate stats
    out = {stat: pd.DataFrame(index=ids, columns=maps) for stat in stats}
    if any(["=" in stat for stat in stats]):
        stats_pct = [stat for stat in stats if stat.startswith("pct==")]
        stats_auc_thresh = [stat for stat in stats if stat.startswith("auc<=")]
        stats_auc2_thresh = [stat for stat in stats if stat.startswith("auc2<=")]
        stats_poly2_thresh = [stat for stat in stats if stat.startswith("poly2<=")]
    else:
        stats_auc_thresh, stats_auc2_thresh, stats_poly2_thresh, stats_pct = [], [], [], []
    for m, curve in curves_mapwise.items():

        # remove global
        if remove_global:
            curve = curve - curve.values[:, percentiles == 0]

        # AUC
        if "auc" in stats:
            out["auc"][m] = np.apply_along_axis(
                fast_auc,
                axis=1,
                arr=curve,
                percentiles=percentiles,
            )
        # AUC with percentile restriction
        if stats_auc_thresh:
            for stat in stats_auc_thresh:
                idx_bool = percentiles <= float(stat.split("=")[1]) / 100
                out[stat][m] = np.apply_along_axis(
                    fast_auc,
                    axis=1,
                    arr=curve.values[:, idx_bool],
                    percentiles=percentiles[idx_bool],
                )

        # AUC square
        if "auc2" in stats:
            out["auc2"][m] = np.apply_along_axis(
                fast_auc2,
                axis=1,
                arr=curve,
                percentiles=percentiles,
            )
        # AUC square with percentile restriction
        if stats_auc2_thresh:
            for stat in stats_auc2_thresh:
                idx_bool = percentiles <= float(stat.split("=")[1]) / 100
                out[stat][m] = np.apply_along_axis(
                    fast_auc2,
                    axis=1,
                    arr=curve.values[:, idx_bool],
                    percentiles=percentiles[idx_bool],
                )

        # 2nd degree polynomial fit
        if "poly2" in stats:
            out["poly2"][m] = np.apply_along_axis(
                poly,
                axis=1,
                arr=curve,
                percentiles=percentiles,
                degree=2,
            )
        # polynomial with percentile restriction
        if stats_poly2_thresh:
            for stat in stats_poly2_thresh:
                idx_bool = percentiles <= float(stat.split("=")[1]) / 100
                out[stat][m] = np.apply_along_axis(
                    poly,
                    axis=1,
                    arr=curve.values[:, idx_bool],
                    percentiles=percentiles[idx_bool],
                    degree=2,
                )

        # Peak connectivity
        if "peak_conn" in stats:
            out["peak_conn"][m] = curve.max(axis=1)

        # Peak percentile
        if "peak_pct" in stats:
            out["peak_pct"][m] = curve.idxmax(axis=1)

        # Connectivity at percentile
        if stats_pct:
            for stat in stats_pct:
                out[stat][m] = curve.loc[:, float(stat.split("==")[1])]

    # return
    if len(out) == 1 and not force_dict:
        return out[list(out.keys())[0]]
    else:
        return out


def auc(
    curve: Union[pd.Series, np.ndarray], percentiles: Sequence[float], square_curve: bool = False
) -> float:
    """Compute AUC for a curve over percentiles."""

    if not isinstance(curve, (pd.Series, np.ndarray)):
        raise ValueError(f"curve must be a pandas Series or numpy array, got {type(curve)}")
    if not np.ndim(curve) == 1:
        raise ValueError(f"curve must be 1D, got {np.ndim(curve)}D")
    if not len(curve) == len(percentiles):
        raise ValueError(
            f"curve and percentiles must have the same length, got {len(curve)} and {len(percentiles)}"
        )

    if square_curve:
        return fast_auc2(curve, percentiles)
    else:
        return fast_auc(curve, percentiles)


def fast_auc(curve: Union[pd.Series, np.ndarray], percentiles: Sequence[float]) -> float:
    """Compute trapezoidal AUC ignoring NaNs."""
    # handle curve
    curve = np.asarray(curve)
    isnan = np.isnan(curve)
    curve = curve[~isnan]

    # handle percentiles
    percentiles = np.asarray(percentiles)
    percentiles = percentiles[~isnan]

    return np.trapz(curve, x=percentiles)


def fast_auc2(curve: Union[pd.Series, np.ndarray], percentiles: Sequence[float]) -> float:
    """Compute squared-transformed AUC ignoring NaNs."""
    # handle curve
    curve = np.asarray(curve)
    isnan = np.isnan(curve)
    curve = curve[~isnan]
    curve = np.tanh(curve)
    curve = curve**2 * np.sign(curve)

    # handle percentiles
    percentiles = np.asarray(percentiles)
    percentiles = percentiles[~isnan]

    return np.trapz(curve, x=percentiles)


def poly(
    curve: Union[pd.Series, np.ndarray], percentiles: Sequence[float], degree: int = 2
) -> float:
    """Fit polynomial to curve and return leading coefficient."""
    # handle curve
    curve = np.asarray(curve)
    isnan = np.isnan(curve)
    curve = curve[~isnan]

    # handle percentiles
    percentiles = np.asarray(percentiles)
    percentiles = percentiles[~isnan]

    return np.polyfit(percentiles, curve, degree)[0]
