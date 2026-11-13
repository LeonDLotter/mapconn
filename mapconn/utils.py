import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix

from .matrix import _sym_matrix_shape_from_n_tri_elem

logger = logging.getLogger(__name__)


def _format_map_data(
    map_data: Union[pd.DataFrame, pd.Series, np.ndarray, xr.DataArray, Sequence[Any]],
    parcel_labels: Optional[Sequence[Any]] = None,
    as_df: bool = True,
    n_parcels: Optional[int] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Bring map data into shape (n_maps, n_parcels).

    Accepts DataFrames (maps x parcels), Series (one map, parcels as index), xarray DataArrays,
    and 1D (one map) or 2D (maps x parcels) arrays/sequences. If `n_parcels` is given and the
    data is unambiguously oriented as (n_parcels, n_maps), it is transposed. If `as_df=True`,
    returns a DataFrame; missing map names default to "map1", "map2", ... and missing parcel
    labels to `parcel_labels` (if given and matching) or 0..n_parcels-1. Otherwise returns a
    2D array.
    """
    if isinstance(map_data, xr.DataArray):
        map_data = map_data.to_pandas()
    if isinstance(map_data, pd.Series):
        name = map_data.name if map_data.name is not None else "map1"
        map_data = map_data.to_frame(name=name).T
    if not isinstance(map_data, pd.DataFrame):
        arr = np.asarray(map_data)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim != 2:
            raise ValueError(
                f"map_data must be 1D (one map) or 2D (n_maps, n_parcels), got shape {arr.shape}"
            )
        if n_parcels is not None and arr.shape[1] != n_parcels and arr.shape[0] == n_parcels:
            logger.info(f"Transposing map_data of shape {arr.shape} to (n_maps, n_parcels)")
            arr = arr.T
        if not as_df:
            return arr
        if parcel_labels is not None and len(parcel_labels) != arr.shape[1]:
            parcel_labels = None
        map_data = pd.DataFrame(
            arr,
            index=[f"map{i+1}" for i in range(arr.shape[0])],
            columns=parcel_labels,
        )
    elif (
        n_parcels is not None and map_data.shape[1] != n_parcels and map_data.shape[0] == n_parcels
    ):
        logger.info(f"Transposing map_data of shape {map_data.shape} to (n_maps, n_parcels)")
        map_data = map_data.T
    return map_data if as_df else map_data.values


def _format_map_data_null(
    map_data_null: Union[dict, np.ndarray, pd.DataFrame, Sequence[Any]],
    map_data: pd.DataFrame,
    n_nulls: Optional[int] = None,
) -> Tuple[dict, int]:
    """
    Bring user-provided null maps into a dict {map name: array (n_nulls, n_parcels)}, ordered
    like `map_data` (n_maps, n_parcels).

    Accepts a dict keyed by map names (or, with a warning, any keys in map order), a 2D array
    (n_nulls, n_parcels) for a single map, or a 3D array (n_maps, n_nulls, n_parcels).
    Returns the dict and the number of nulls to use (at most the number available).
    """
    maps = list(map_data.index)
    n_maps, n_parcels = map_data.shape

    # to dict
    if not isinstance(map_data_null, dict):
        arr = np.asarray(map_data_null)
        if arr.ndim == 2 and n_maps == 1:
            map_data_null = {maps[0]: arr}
        elif arr.ndim == 3 and arr.shape[0] == n_maps:
            map_data_null = {m: arr[i] for i, m in enumerate(maps)}
        else:
            raise ValueError(
                "map_data_null must be a dict {map: array (n_nulls, n_parcels)}, a 2D array "
                "(n_nulls, n_parcels) for a single map, or a 3D array (n_maps, n_nulls, n_parcels); "
                f"got shape {arr.shape} for {n_maps} map(s)"
            )

    # match keys to maps
    if set(map_data_null.keys()) == set(maps):
        map_data_null = {m: map_data_null[m] for m in maps}
    elif len(map_data_null) == n_maps:
        logger.warning(
            "Keys of map_data_null do not match map names; assigning null maps in order: "
            f"{dict(zip(map_data_null.keys(), maps))}"
        )
        map_data_null = dict(zip(maps, map_data_null.values()))
    else:
        missing = [m for m in maps if m not in map_data_null]
        raise ValueError(f"map_data_null is missing null maps for: {missing}")

    # shapes
    for m, arr in map_data_null.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(
                f"Null maps for '{m}' must be 2D (n_nulls, n_parcels), got {arr.shape}"
            )
        if arr.shape[1] != n_parcels and arr.shape[0] == n_parcels:
            arr = arr.T
        if arr.shape[1] != n_parcels:
            raise ValueError(
                f"Null maps for '{m}' have {arr.shape[1]} parcels, map data has {n_parcels}"
            )
        map_data_null[m] = arr

    # number of nulls
    n_available = min(arr.shape[0] for arr in map_data_null.values())
    if n_nulls is None or n_nulls > n_available:
        if n_nulls is not None:
            logger.info(f"Using {n_available} provided null maps (n_nulls={n_nulls} requested)")
        n_nulls = n_available

    return map_data_null, n_nulls


# convert flat mappercentile data to parcel-format data
def _mappct_flat_to_parcels(
    mappct_data_flat: pd.DataFrame, parcel_labels: Optional[Sequence[Any]] = None
) -> pd.DataFrame:
    """Convert flat map-percentile masks to parcel-format masks."""
    n_parcels = _sym_matrix_shape_from_n_tri_elem(mappct_data_flat.shape[1])

    diagonal = np.full(n_parcels, False)
    mappct_data = np.full((mappct_data_flat.shape[0], n_parcels), False)
    for i_v in range(mappct_data.shape[0]):
        mat = vec_to_sym_matrix(mappct_data_flat.values[i_v, :], diagonal=diagonal)
        mappct_data[i_v, :] = mat.sum(axis=0) > 0

    return pd.DataFrame(
        mappct_data, columns=parcel_labels, index=mappct_data_flat.index, dtype=bool
    )


# convert parcel-format mappercentile data to flat data
def _mappct_parcels_to_flat(
    mappct_data: pd.DataFrame, parcel_pair_labels: Optional[Sequence[Any]] = None
) -> pd.DataFrame:
    """Convert parcel-format map-percentile masks to flat masks."""
    return pd.DataFrame(
        np.stack(
            [
                sym_matrix_to_vec(
                    np.outer(mappct_data.values[i, :], mappct_data.values[i, :]),
                    discard_diagonal=True,
                )
                for i in range(mappct_data.shape[0])
            ]
        ),
        index=mappct_data.index,
        columns=parcel_pair_labels,
    )


def _construct_flat_label_pairs(
    labels: Sequence[Any], discard_diagonal: bool = True
) -> List[Tuple[Any, Any]]:
    """Construct parcel label pairs for flattened matrix indices."""
    idc = np.tril_indices(len(labels), -1 if discard_diagonal else 0)
    return [(labels[i], labels[j]) for i, j in zip(*idc)]


def reduce_df_index(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Drop index levels with a single unique value."""
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise ValueError("df must be a pandas DataFrame or Series")

    idc_one_level = [i for i in df.index.names if df.index.get_level_values(i).nunique() == 1]
    df = df.droplevel(idc_one_level)

    return df
