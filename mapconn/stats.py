import numpy as np
import pandas as pd

from .constants import STATS

def _remove_global(mapconn_curves):
    """
    Remove global connectivity (= value at 0th percentile) from mapconn curves.
    """
    
    df = (
        mapconn_curves
        .T
        .groupby("map", sort=False, group_keys=False)
        .apply(lambda x: x - x.values[0,:])
        .T
    )
    return df


def _calc_mapconn_stats(mapconn_curves, stats="all", force_dict=False):
    """
    Calculate statistics from mapconn curves.
    """
    
    # checks
    if not isinstance(mapconn_curves, pd.DataFrame):
        raise ValueError("mapconn_curves must be a pandas DataFrame")
    if stats == "all":
        stats = list(STATS)
    if isinstance(stats, (str, int)):
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
    
    # calculate stats
    out = {stat: pd.DataFrame(index=ids, columns=maps) for stat in stats}
    stats_pct = [stat for stat in stats if stat in percentiles]
    for m, curve in curves_mapwise.items():
    
        # AUC
        if "auc" in stats:
            out["auc"][m] = np.apply_along_axis(
                auc, 
                axis=1, 
                arr=curve, 
                percentiles=percentiles
            )
            
        # Peak connectivity
        if "peak_conn" in stats:
            out["peak_conn"][m] = curve.max(axis=1) - curve.loc[:, 0]
            
        # Peak percentile
        if "peak_pct" in stats:
            out["peak_pct"][m] = curve.idxmax(axis=1)
        
        # Connectivity at percentile
        if stats_pct:
            for pct in stats_pct:
                out[pct][m] = curve.loc[:, pct] - curve.loc[:, 0]
    
    # return
    if len(out) == 1 and not force_dict:
        return out[list(out.keys())[0]]
    else:
        return out
      
        
def auc(curve, percentiles):
    
    if not isinstance(curve, (pd.Series, np.ndarray)):
        raise ValueError(f"curve must be a pandas Series or numpy array, got {type(curve)}")
    if not np.ndim(curve) == 1:
        raise ValueError(f"curve must be 1D, got {np.ndim(curve)}D")
    if not len(curve) == len(percentiles):
        raise ValueError(f"curve and percentiles must have the same length, got {len(curve)} and {len(percentiles)}")
    
    curve = np.array(curve)
    percentiles = np.array(percentiles)
    
    return np.trapz(curve - curve[percentiles==0], x=percentiles)