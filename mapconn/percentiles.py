import logging
import operator
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from nilearn.connectome import sym_matrix_to_vec
from scipy.stats import percentileofscore

from .constants import MapPctThreshold
from .matrix import _n_sym_matrix_tri_elem_from_shape

logger = logging.getLogger(__name__)


def _calc_mappct_masks(
    map_data: Union[np.ndarray, pd.DataFrame],
    map_data_is_pct: bool = False,
    percentiles: Optional[Sequence[float]] = None,
    n_jobs: int = -1,
    verbose: bool = True,
    pct_kind: str = "strict",
    pct_threshold: MapPctThreshold = "overequal",
    return_df: bool = True,
    dtype: Optional[Union[np.dtype, type]] = None,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
    """
    Compute map-percentile masks from map data.

    Returns (percentile_data, mask_matrix) when `return_df=True`.
    """
    if percentiles is None:
        percentiles = np.arange(0, 100, 5)
    # data
    if map_data.ndim == 1:
        map_data_arr = np.atleast_2d(map_data)
    elif map_data.ndim == 2:
        map_data_arr = np.array(map_data)
    else:
        raise ValueError("map_data must be a 1D or 2D array")
    if verbose:
        logger.info(
            "Got map data with %s maps and %s values", map_data_arr.shape[0], map_data_arr.shape[1]
        )

    # threshold function
    threshold_funs = {
        "over": operator.gt,
        "overequal": operator.ge,
        "below": operator.lt,
        "belowequal": operator.le,
    }
    if pct_threshold not in threshold_funs:
        raise ValueError(f"Invalid threshold function '{pct_threshold}'")
    threshold_fun = threshold_funs[pct_threshold]

    # calculate map percentiles
    if not map_data_is_pct:
        if verbose:
            logger.info("Calculating map percentiles")
        map_data_arr = np.apply_along_axis(
            values_to_percentiles, axis=1, arr=map_data_arr, kind=pct_kind
        ).astype(dtype)
    else:
        if verbose:
            logger.info("Assuming percentiles already calculated")

    # maps
    n_maps = map_data_arr.shape[0]

    # parcels
    n_parcels = map_data_arr.shape[1]
    n_parcels_flat = _n_sym_matrix_tri_elem_from_shape(n_parcels)

    # percentiles
    percentiles = np.array(percentiles)
    n_percentiles = len(percentiles)

    # calculate masks: array with shape (n_maps * n_percentiles, n_parcels)
    mappct_masks_flat = []
    for i_map in range(n_maps):
        tmp_flat = np.full((n_percentiles, n_parcels_flat), False)
        for i_pct, pct in enumerate(percentiles):
            v = threshold_fun(map_data_arr[i_map, :], pct)
            tmp_flat[i_pct, :] = sym_matrix_to_vec(np.outer(v, v), discard_diagonal=True)
        mappct_masks_flat.append(tmp_flat)
    mappct_masks_flat_arr = np.concatenate(mappct_masks_flat, axis=0, dtype=bool)

    # to df and return
    if return_df:
        out = (
            pd.DataFrame(map_data_arr, index=map_data.index, columns=map_data.columns, dtype=dtype),
            pd.DataFrame(
                mappct_masks_flat_arr,
                index=pd.MultiIndex.from_product(
                    [map_data.index, percentiles], names=["map", "pct"]
                ),
                dtype=bool,
            ),
        )
    else:
        out = (map_data_arr, mappct_masks_flat_arr)
    return out


def values_to_percentiles(
    values: Union[np.ndarray, Sequence[float]],
    population: Optional[Union[np.ndarray, Sequence[float]]] = None,
    kind: str = "strict",
) -> np.ndarray:
    """Convert values to percentile ranks within a population."""
    if population is None:
        population = values
    return percentileofscore(a=population, score=values, kind=kind, nan_policy="omit")
