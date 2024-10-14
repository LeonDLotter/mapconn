import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy.stats import percentileofscore
from nilearn.connectome import sym_matrix_to_vec

from .matrix import _n_sym_matrix_tri_elem_from_shape
from .utils import over, below, overequal, belowequal


def _calc_mappct_masks(map_data=None, map_data_is_pct=False, percentiles=np.arange(0, 100, 5), 
                       n_jobs=-1, verbose=True, pct_kind="strict", pct_threshold="overequal",
                       return_df=True, dtype=None):
    
    # data
    if map_data.ndim == 1:
        map_data_arr = np.atleast_2d(map_data)
    elif map_data.ndim == 2:
        map_data_arr = np.array(map_data)
    else:
        raise ValueError("map_data must be a 1D or 2D array")
    if verbose:
        print(f"Got map data with {map_data_arr.shape[0]} maps and {map_data_arr.shape[1]} values")
    
    # threshold function
    threshold_funs = {
        "over": over,
        "overequal": overequal,
        "below": below,
        "belowequal": belowequal
    }
    if pct_threshold in threshold_funs:
        threshold_fun = threshold_funs[pct_threshold]
    else:
        raise ValueError(f"Invalid threshold function '{pct_threshold}'")

    # calculate map percentiles
    if not map_data_is_pct:
        if verbose:
            print("Calculating map percentiles")
        map_data_arr = np.apply_along_axis(
            values_to_percentiles, 
            axis=1, 
            arr=map_data_arr, 
            kind=pct_kind
        ).astype(dtype)
    else:
        if verbose:
            print("Assuming percentiles already calculated")
    
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
        #tmp = np.full((n_percentiles, n_parcels), False)
        tmp_flat = np.full((n_percentiles, n_parcels_flat), False)
        for i_pct, pct in enumerate(percentiles):
            v = threshold_fun(map_data_arr[i_map, :], pct)
            #tmp[i_pct, :] = v
            tmp_flat[i_pct, :] = sym_matrix_to_vec(np.outer(v, v), discard_diagonal=True)
        #mappct_masks.append(tmp)
        mappct_masks_flat.append(tmp_flat)
    #mappct_masks_arr = np.concatenate(mappct_masks, axis=0)
    mappct_masks_flat_arr = np.concatenate(mappct_masks_flat, axis=0, dtype=bool)
        
    # to df and return
    if return_df:
        out = (
            pd.DataFrame(
                map_data_arr,
                index=map_data.index,
                columns=map_data.columns,
                dtype=dtype
            ),
            # pd.DataFrame(
            #     mappct_masks_arr,
            #     index=pd.MultiIndex.from_product([map_data.index, percentiles], names=["map", "pct"])
            # ),
            pd.DataFrame(
                mappct_masks_flat_arr,
                index=pd.MultiIndex.from_product([map_data.index, percentiles], names=["map", "pct"]),
                dtype=bool
            )
        )
    else:
        out = (map_data_arr, mappct_masks_flat_arr)
    return out
    
    
def values_to_percentiles(values, population=None, kind="strict"):
    if population is None:
        population = values
    return percentileofscore(
        a=population,
        score=values,
        kind=kind,
        nan_policy="omit"
    )
    
    
# def calculate_mappercentiles(map_data, percentiles=np.arange(0, 100, 5), flat=True,
#                              n_jobs=-1, verbose=True, threshold="overequal"):
    
#     # data
#     if map_data.ndim == 1:
#         map_data_array = np.atleast_2d(map_data)
#     elif map_data.ndim == 2:
#         map_data_array = np.array(map_data)
#     else:
#         raise ValueError("map_data must be a 1D or 2D array")
    
#     # percentiles
#     percentiles = np.array(percentiles)
    
#     # threshold function
#     if threshold == "overequal":
#         threshold_fun = overequal
#     elif threshold == "below":
#         threshold_fun = below
#     else:
#         raise ValueError(f"Invalid threshold function '{threshold}'")
    
#     # get quantiles in parallel
#     mappct = Parallel(n_jobs=n_jobs)(
#         delayed(_parcels_by_percentile)(map_data_array[i, :], percentiles, flat, threshold_fun=threshold_fun) 
#         for i in tqdm(range(map_data_array.shape[0]), disable=not verbose, desc="Calculating map percentiles")
#     )
    
#     # sort into df and return
#     # TODO: MAKE NICE
#     if isinstance(map_data, pd.DataFrame):
#         if isinstance(map_data.index, pd.MultiIndex):
#             keys = ["_".join([str(i) for i in idx]) for idx in map_data.index.to_flat_index()]
#         else:
#             keys = map_data.index
#     elif isinstance(map_data, pd.Series):
#         keys = [map_data.name]
#     else:
#         keys = [f"map{i+1}" for i in range(map_data_array.shape[0])]
#     mappct = pd.concat(mappct, keys=keys, names=["map", "pct"])
    
#     return mappct


# percentile thresholds for one parcel vector (e.g., one parcellated PET map)
# def _parcels_by_percentile(map_vector, percentiles=np.arange(0, 100, 5), flat=True, 
#                            threshold_fun=overequal):
#     data_array = np.squeeze(map_vector)
#     notna = ~np.isnan(data_array)
    
#     if flat:
#         parcels_over_quantile = np.full((len(percentiles), _n_sym_matrix_tri_elem_from_shape(len(data_array))), False)
#         for i, q in enumerate(percentiles):
#             temp = np.full(len(data_array), False)
#             temp[notna] = threshold_fun(data_array[notna], np.nanquantile(data_array, q / 100))
#             parcels_over_quantile[i, :] = sym_matrix_to_vec(np.outer(temp, temp), discard_diagonal=True)
#     else:
#         parcels_over_quantile = np.full((len(percentiles), len(data_array)), False)
#         for i, q in enumerate(percentiles):
#             parcels_over_quantile[i, notna] = threshold_fun(data_array[notna], np.nanquantile(data_array, q / 100))
    
#     parcels_by_pct = pd.DataFrame(
#         parcels_over_quantile, 
#         index=percentiles,
#         columns=map_vector.index 
#                 if (not flat) and isinstance(map_vector, (pd.Series, pd.DataFrame)) else None
#     )
#     return parcels_by_pct
