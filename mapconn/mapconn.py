import numpy as np
import pandas as pd
import pickle
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from nilearn.connectome import vec_to_sym_matrix
from nispace.nulls import generate_null_maps
from nispace.stats.misc import null_to_p, permute_groups
from nispace.stats.effectsize import cohen_paired
from scipy.stats import ttest_rel, median_abs_deviation
from itertools import product
import xarray as xr

from .matrix import (_get_matrix_estimator, _vectorize_sym_matrices,
                     _n_sym_matrix_tri_elem_from_shape, _sym_matrix_shape_from_n_tri_elem,
                     matrix_sac)
from .percentiles import _calc_mappct_masks
from .utils import _construct_flat_label_pairs, _mappct_flat_to_parcels, reduce_df_index
from .stats import _calc_mapconn_stats, _remove_global
from .constants import STATS

logger = logging.getLogger(__name__)

ConnAggregation = Literal["mean", "median"]
MapPctThreshold = Literal["over", "overequal", "below", "belowequal"]

class MapConn:
    """
    Class for storing and analyzing map-connectivity ("NEOFC") data.
    """
    
    def __init__(self, 
                 flat_connectivity_matrices: Optional[pd.DataFrame] = None, 
                 flat_mappercentile_masks: Optional[pd.DataFrame] = None, 
                 map_data: Optional[Union[pd.DataFrame, np.ndarray, xr.DataArray]] = None,
                 mappct_data: Optional[Union[pd.DataFrame, np.ndarray, xr.DataArray]] = None,
                 mapconn_curves: Optional[pd.DataFrame] = None,
                 parcel_labels: Optional[Sequence[Any]] = None,
                 n_parcels: Optional[int] = None,
                 conn_aggregation: ConnAggregation = "mean",
                 mappct_thresh: MapPctThreshold = "overequal",
                 r_to_z: bool = False,
                 mapconn_stats: Optional[Dict[str, pd.DataFrame]] = None,
                 loo_stats: Optional[Dict[str, Any]] = None,
                 n_jobs: int = -1,
                 dtype: Union[np.dtype, type] = np.float32,
                 get_stats: bool = False
                 ) -> None:
        """
        Initialize the MapConn class.

        Expected data layout:
        - `mapconn_curves`: DataFrame indexed by subject/instance id, with
            MultiIndex columns named ("map", "pct").
        - `flat_connectivity_matrices`: DataFrame of shape
            $(n_{ids}, n_{edges})$ with flattened lower-triangle connectivity.
        - `flat_mappercentile_masks`: DataFrame of shape
            $(n_{maps} \times n_{percentiles}, n_{edges})$ aligned to `mapconn_curves`.
        """
        self._conn_data = flat_connectivity_matrices
        self._mappct_masks = flat_mappercentile_masks
        self._map_data = map_data
        self._mappct_data = mappct_data
        self._mapconn_curves = mapconn_curves
        self._parcel_labels = parcel_labels
        self._n_parcels = n_parcels 
        self._r_to_z = r_to_z
        self._n_jobs = n_jobs
        self._mapconn_stats = mapconn_stats
        self._dtype = dtype
        self._conn_agg = conn_aggregation   
        self._mappct_thresh = mappct_thresh
        self._loo = loo_stats if loo_stats is not None else {}
        
        # input validation of mapconn_curves
        if mapconn_curves is None:
            raise ValueError("mapconn_curves must be provided")
        else:
            # dataframe
            if not isinstance(mapconn_curves, pd.DataFrame):
                raise ValueError("mapconn_curves must be a pandas DataFrame")
            if mapconn_curves.columns.names != ["map", "pct"]:
                raise ValueError("mapconn_curves columns must be named 'map' and 'pct'")
            # percentiles
            percentiles = mapconn_curves.columns.get_level_values("pct").astype(float).unique()
            if not all(percentiles >= 0) or not all(percentiles <= 100):
                raise ValueError("All second-level column names in mapconn_curves (= percentiles) "
                                 "must be numbers between 0 and 100.")
            self._percentiles = sorted(percentiles)   
            # maps 
            maps = mapconn_curves.columns.get_level_values("map").unique().to_list()
            self._maps = maps
            self._n_maps = len(maps)
            # subjects (or something like this)
            self._ids = mapconn_curves.index.to_list()
            # dtype conversion
            self._mapconn_curves = mapconn_curves.astype(self._dtype)
            
        # input validation of parcel_labels/n_parcels
        if parcel_labels is not None and n_parcels is not None:
            if len(parcel_labels) != n_parcels:
                raise ValueError("n_parcels and parcel_labels must correspond")
        elif n_parcels is not None and parcel_labels is None:
            self._parcel_labels = list(range(n_parcels))
        elif parcel_labels is not None and n_parcels is None:
            self._n_parcels = len(parcel_labels)
            
        # input validation of other df's  
        if flat_connectivity_matrices is not None or flat_mappercentile_masks is not None:
            if n_parcels is None and parcel_labels is None:
                raise ValueError("if flat_connectivity_matrices or flat_mappercentile_data is provided, "
                                 "n_parcels or parcel_labels should be provided for input validation")
            for df_name, df in [("flat_connectivity_matrices", flat_connectivity_matrices), 
                                ("flat_mappercentile_data", flat_mappercentile_masks)]:
                if df is not None:
                    # checks
                    if not isinstance(df, pd.DataFrame):
                        raise ValueError(f"{df_name} must be a pandas DataFrame")
                    if df.shape[1] != _n_sym_matrix_tri_elem_from_shape(self._n_parcels):
                        raise ValueError(f"The number of columns in {df_name} must correspond "
                                         "to the number of parcels")
                    if ("connectivity" in df_name) and (df.shape[0] != mapconn_curves.shape[0]):
                        raise ValueError("The number of rows in flat_connectivity_matrices must "
                                        "match the number of rows in mapconn_curves")
                    if ("mappercentile" in df_name) and (df.shape[0] != mapconn_curves.shape[1]):
                        raise ValueError("The number of rows in flat_mappercentile_data must "
                                         "match the number of columns in mapconn_curves")
                    # dtype conversion
                    df = df.astype(self._dtype)
                    
        # precompute stats
        if get_stats:
            self.get_stats()
    
    def __getitem__(self, key: Any) -> pd.DataFrame:
        """
        Allows slicing of the mapconn_curves DataFrame directly via the instance.
        """
        return self._mapconn_curves.loc[key]
        
    def get_curves(self, 
                   maps: Optional[Sequence[Any]] = None, 
                   percentiles: Optional[Sequence[float]] = None, 
                   ids: Optional[Sequence[Any]] = None, 
                   remove_global: bool = True
                   ) -> pd.DataFrame:
        """
        Returns mapconn curves.

        Output shape: $(n_{ids}, n_{maps} \times n_{percentiles})$ with
        MultiIndex columns named ("map", "pct").
        """
        mapconn_curves = self._mapconn_curves
        maps = maps if maps is not None else self._maps
        percentiles = percentiles if percentiles is not None else self._percentiles
        ids = ids if ids is not None else self._ids
        
        if remove_global:
            mapconn_curves = _remove_global(mapconn_curves)

        return mapconn_curves.loc[ids, (maps, percentiles)]
    
    def get_parcel_labels(self, flat_pairs: bool = False) -> Union[List[Any], List[Tuple[Any, Any]]]:
        """
        Returns the parcel labels. If flat_pairs is True, returns parcel pairs corresponding 
        to columns of the flat data format.
        """
        if flat_pairs:
            return _construct_flat_label_pairs(self._parcel_labels, discard_diagonal=True)
        else:
            return self._parcel_labels

    def get_connectivity_matrices(self, 
                                  ids: Optional[Sequence[Any]] = None, 
                                  flat: bool = True, 
                                  flat_col_names: bool = True, 
                                  fill_diagonal: float = np.nan
                                  ) -> Union[pd.DataFrame, List[np.ndarray]]:
        """
        Returns connectivity matrices.

        - If `flat=True`: DataFrame with shape $(n_{ids}, n_{edges})$.
          Columns optionally labeled as (parcelA, parcelB).
        - If `flat=False`: list of $(n_{parcels}, n_{parcels})$ arrays.
        """
        # get conn data
        if self._conn_data is None:
            raise ValueError("No connectivity matrices available")
        ids = ids if ids is not None else self._ids
        conn_data = self._conn_data.loc[ids, :]
        # return
        if flat:
            if flat_col_names:
                conn_data.columns = pd.MultiIndex.from_tuples(self.get_parcel_labels(True),
                                                              names=["parcelA", "parcelB"])
        else:
            diagonal = np.full(_sym_matrix_shape_from_n_tri_elem(self._conn_data.shape[1]), fill_diagonal)
            conn_data = [vec_to_sym_matrix(conn_data.values[i,:], diagonal=diagonal) 
                         for i in range(conn_data.shape[0])]
        
        return conn_data

    def get_mappercentile_masks(self, maps: Optional[Sequence[Any]] = None, 
                                percentiles: Optional[Sequence[float]] = None, 
                                flat: bool = True, 
                                flat_col_names: bool = True
                                ) -> pd.DataFrame:
        """
        Returns map-percentile masks.

        - If `flat=True`: DataFrame with shape $(n_{maps} \times n_{percentiles}, n_{edges})$.
        - If `flat=False`: parcel-format DataFrame with shape
          $(n_{maps} \times n_{percentiles}, n_{parcels}, n_{parcels})$.
        """
        if self._mappct_masks is None:
            raise ValueError("No mappercentile data available")
        if flat:
            mappct_masks = self._mappct_masks
            if flat_col_names:
                mappct_masks.columns = pd.MultiIndex.from_tuples(self.get_parcel_labels(True),
                                                                names=["parcelA", "parcelB"])
        else:
            mappct_masks = _mappct_flat_to_parcels(self._mappct_masks, parcel_labels=self._parcel_labels)
        maps = maps if maps is not None else self._maps
        percentiles = percentiles if percentiles is not None else self._percentiles
        return mappct_masks.loc[(maps, percentiles), :]
    
    def get_map_data(self, maps: Optional[Sequence[Any]] = None, pct: bool = False) -> pd.DataFrame:
        """
        Return map data for the requested maps.

        If `pct=True`, returns map-percentile data.
        """
        map_data = self._map_data if not pct else self._mappct_data
        if map_data is None:
            raise ValueError("No map data available")
        maps = maps if maps is not None else self._maps
        return map_data.loc[maps]
    
    def get_stats(self, 
                  stats: Optional[Union[str, Sequence[str]]] = None, 
                  maps: Optional[Sequence[Any]] = None, 
                  percentiles: Optional[Sequence[float]] = None, 
                  ids: Optional[Sequence[Any]] = None, 
                  recalculate: bool = False, 
                  force_dict: bool = False
                  ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Return curve statistics (e.g., AUC, poly2) for selected maps/ids.

        Returns a DataFrame or dict of DataFrames keyed by stat.
        """
        if stats is None:
            stats = ["auc", "poly2"]
        if isinstance(stats, str):
            stats = [stats]
        if not all(stat in STATS for stat in stats):
            raise ValueError(f"stats must be one or more of {STATS}, got {stats}")
        # get stats
        mapconn_stats = self._mapconn_stats
        # recalculate
        if mapconn_stats is None or recalculate or any(stat not in mapconn_stats.keys() for stat in stats):
            recalculate = True
        # maybe recalculate
        elif mapconn_stats is not None:
            # get mapconn curves for reference
            mapconn_curves = self.get_curves(maps=maps, percentiles=percentiles, ids=ids)    
            # check if all stats and data are available
            if not all(stat in mapconn_stats.keys() for stat in stats):
                recalculate = True
            if not np.array_equal(mapconn_stats[stats[0]].index, 
                                  mapconn_curves.index):
                recalculate = True
            if not np.array_equal(mapconn_stats[stats[0]].columns.unique(), 
                                  mapconn_curves.columns.get_level_values("map").unique()):
                recalculate = True
                
        if recalculate:
            mapconn_curves = self.get_curves(maps=maps, percentiles=percentiles, ids=ids, remove_global=False)
            mapconn_stats = _calc_mapconn_stats(mapconn_curves, stats=stats, force_dict=True, remove_global=True)
            self._mapconn_stats = mapconn_stats
        else:
            mapconn_stats = {stat: mapconn_stats[stat] for stat in stats}
            
        if not force_dict:
            if len(mapconn_stats.keys()) == 1:
                mapconn_stats = mapconn_stats[stats[0]]
        
        return mapconn_stats
    
    def get_loo(self, 
                what: str = "parcels", 
                stats: Optional[Union[str, Sequence[str]]] = None, 
                maps: Optional[Sequence[Any]] = None, 
                percentiles: Optional[Sequence[float]] = None, 
                ids: Optional[Sequence[Any]] = None, 
                return_difference: bool = True, 
                relative_difference: bool = False, 
                recalculate: bool = False, 
                force_dict: bool = False, 
                n_jobs: int = -1, 
                verbose: bool = True, 
                **kwargs
                ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Returns the regional importance of parcels or connections based on leave-one-out analysis.
        """
        if stats is None:
            stats = ["auc", "poly2"]
        # delta and mapconn_stats as "secret" keyword arguments
        delta = kwargs.pop("delta", False)
        mapconn_stats = kwargs.pop("mapconn_stats", None)
        
        # load existing loo stats
        if what not in ["parcels", "connections"]:
            raise ValueError("what must be one of 'parcels', 'connections'")
        loo_key = f"loo-{what}_delta-{delta}"
        loo = self._loo.get(loo_key, None)
        
        # get stats
        if mapconn_stats is None:
            mapconn_stats = self.get_stats(stats=stats, maps=maps, percentiles=percentiles, ids=ids, 
                                           force_dict=True)
        stats = list(mapconn_stats.keys())
        maps = mapconn_stats[stats[0]].columns.to_list()
        ids = mapconn_stats[stats[0]].index.to_list()
        if percentiles is None:
            percentiles = self._percentiles
            
        # check if loo stats are already available and complete and return if so
        if loo is not None and not recalculate and return_difference:
            if all(stat in loo.keys() for stat in stats) and \
                all(m in loo[stats[0]].columns for m in maps) and \
                all(id in loo[stats[0]].index.get_level_values("id").unique() for id in ids):
                
                if verbose:
                    logger.info("Returning existing LOO stats")
                if not force_dict:
                    if len(stats) == 1:
                        loo = loo[stats[0]]
                return loo
        
        # get reference data
        map_data = self.get_map_data(maps, pct=False)
        
        # get connectivity matrices
        flat_conn_matrices = self.get_connectivity_matrices(ids=ids, flat=True)
        
        # index to iterate
        if what == "parcels":
            index = map_data.columns.to_list()
        else:
            index = flat_conn_matrices.columns.to_list()
        
        # parallelization function            
        def par_fun(what: str, loo_idx: Any) -> Dict[str, pd.DataFrame]:
            """Run a single leave-one-out iteration for a parcel or connection."""
            
            if what == "parcels":
                # drop parcel from map data
                map_data_loo = map_data.copy()
                map_data_loo[loo_idx] = np.nan
                # connectivity matrices remain the same
                flat_conn_matrices_loo = flat_conn_matrices
            else:
                # drop connection
                flat_conn_matrices_loo = flat_conn_matrices.copy()
                flat_conn_matrices_loo[loo_idx] = np.nan
                # map data remains the same
                map_data_loo = map_data
            
            # run 
            mapconn_kwargs = dict(
                flat_connectivity_matrices=flat_conn_matrices_loo,
                map_data=map_data_loo,
                matrix_ids=ids,
                percentiles=percentiles,
                n_jobs=1,
                verbose=False
            )
            if not delta:
                mapconn_stats_loo = MapConn.from_flat_matrix(**mapconn_kwargs) \
                    .get_stats(force_dict=True)
            else:
                mapconn_stats_loo = MapConnInv.from_flat_matrix(**mapconn_kwargs, get_pvalues=False) \
                    .get_delta_stats(force_dict=True)
            
            # return difference between full stats and loo stats
            diff = {}
            for stat in stats:
                if return_difference:
                    diff[stat] = mapconn_stats[stat] - mapconn_stats_loo[stat]
                else:
                    diff[stat] = mapconn_stats_loo[stat] # that's not the difference but the stats w/o current parcel
                if what == "parcels":
                    diff[stat] = diff[stat].assign(parcel=loo_idx).reset_index(names="id")
                else:
                    diff[stat] = diff[stat].assign(parcelA=loo_idx[0], parcelB=loo_idx[1]).reset_index(names="id")
            return diff
                
        # parallelize
        diff_list = Parallel(n_jobs=n_jobs)(
            delayed(par_fun)(what, idx)
            for idx in tqdm(index, disable=not verbose, desc=f"Calculating LOO ({what})")
        )
        # combine difference results
        diff_dfs = {
            stat: (
                pd.concat([diff_list[i][stat] for i in range(len(diff_list))], axis=0)
                .set_index(["parcel", "id"] if what == "parcels" else ["parcelA", "parcelB", "id"])
            )
            for stat in stats
        }
        
        # save difference stats
        if return_difference:
            self._loo[loo_key] = diff_dfs
        
        # to relative difference
        if relative_difference:
            diff_dfs = {stat: diff_dfs[stat] / mapconn_stats[stat].rename_axis(index="id") for stat in stats}
        
        # return
        if not force_dict:
            if len(stats) == 1:
                diff_dfs = diff_dfs[stats[0]]
        return diff_dfs
    
    def get_matrix_sac(self, 
                       distmat: np.ndarray, 
                       ids: Optional[Sequence[Any]] = None, 
                       **kwargs
                       ) -> pd.DataFrame:
        """
        Returns the spatial autocorrelation of the connectivity matrices.
        """
        if distmat is None:
            raise ValueError("Need distance matrix to calculate spatial autocorrelation")
        if ids is None:
            ids = self._ids
        conn_matrices = self.get_connectivity_matrices(ids=ids, flat=False)
        sac = pd.DataFrame(index=pd.Index(ids, name="id"), columns=["sa_lambda", "sa_infinity"])
        for i, mat in enumerate(conn_matrices):
            sac.loc[ids[i], :] = matrix_sac(mat, distmat, **kwargs)
        return sac
    
    def get_summary(self, 
                    level: str = "group", 
                    stats: Optional[Union[str, Sequence[str]]] = None, 
                    maps: Optional[Sequence[Any]] = None, 
                    percentiles: Optional[Sequence[float]] = None, 
                    ids: Optional[Sequence[Any]] = None, 
                    agg_stats: Optional[Sequence[str]] = None, 
                    reduce_index: bool = True
                    ) -> pd.DataFrame:
        """
        Returns concatenated dataframes of all available summary data (no curves).
        """
        if level not in ["group", "individual"]:
            raise ValueError(f"level must be one of 'group' | 'individual', got {level}")
        if agg_stats is None:
            agg_stats = ["mean", "std", "min", "max"]
        stats = self.get_stats(stats=stats, maps=maps, percentiles=percentiles, ids=ids, force_dict=True)
        df = (
            pd.concat(stats.values(), keys=stats.keys(), axis=0)
            .reset_index(names=["curve_stat", "id"])
            .assign(metric="original", variable="val")
            .set_index(["curve_stat", "metric", "variable", "id"])
        )
        if level == "group":
            df = (
                df.groupby(["curve_stat", "metric"], sort=False)
                .agg(agg_stats)
                .stack(future_stack=True)
            )
            df.index.names = df.index.names[:-1] + ["variable"]
            
        if reduce_index:
            df = reduce_df_index(df)
        return df
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Pickle the mapconn instance to a file.
        """            
        path = Path(path)
        save_gzip = path.suffix == ".gz"
        if save_gzip:
            with gzip.open(path, "wb", compresslevel=9) as f:
                pickle.dump(self, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)
                
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MapConn":
        """
        Load the mapconn instance from a pickled file.
        """
        path = Path(path)
        save_gzip = path.suffix == ".gz"
        if save_gzip:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)

    @classmethod
    def from_flat_matrix(cls, 
                         flat_connectivity_matrices: Union[np.ndarray, pd.DataFrame], 
                         map_data: Optional[Union[np.ndarray, pd.DataFrame, xr.DataArray]] = None, 
                         map_data_is_pct: bool = False, 
                         flat_mappercentile_masks: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                         map_operations: Optional[Sequence[str]] = None,
                         matrix_ids: Optional[Sequence[Any]] = None, 
                         parcel_labels: Optional[Sequence[Any]] = None, 
                         r_to_z: bool = False, 
                         percentiles: Optional[Sequence[float]] = None,
                         mappercentile_threshold: MapPctThreshold = "overequal", 
                         conn_aggregation: ConnAggregation = "mean", 
                         n_jobs: int = -1, 
                         verbose: bool = True, 
                         dtype: Union[np.dtype, type] = np.float32
                         ) -> "MapConn":
        """
        Create an instance from flattened connectivity matrices.

        Parameters:
        - `flat_connectivity_matrices`: array/DataFrame with shape $(n_{ids}, n_{edges})$.
        - `map_data`: array/DataFrame with shape $(n_{maps}, n_{parcels})$.
        - `percentiles`: values in $[0, 100]$ used to compute masks.
        """
        if map_operations is None:
            map_operations = []
        if percentiles is None:
            percentiles = np.arange(0, 100, 5)
        # Ensure flat_matrices is a numpy array
        if not isinstance(flat_connectivity_matrices, (np.ndarray, pd.DataFrame)):
            raise ValueError("flat_connectivity_matrices must be a numpy ndarray or pandas DataFrame")
        
        # Check if matrix_ids are provided, else generate default ids
        if matrix_ids is None:
            if isinstance(flat_connectivity_matrices, pd.DataFrame):
                matrix_ids = flat_connectivity_matrices.index
            else:
                matrix_ids = [f"mat{i+1}" for i in range(flat_connectivity_matrices.shape[0])]
        elif isinstance(matrix_ids, str):
            matrix_ids = [matrix_ids]
        # Ensure matrix_ids length matches the number of matrices
        if len(matrix_ids) != flat_connectivity_matrices.shape[0]:
            raise ValueError("Length of matrix_ids must match the number of matrices")
        # Ensure parcel_labels are provided, else generate default labels
        if parcel_labels is None:
            parcel_labels = \
                list(range(_sym_matrix_shape_from_n_tri_elem(flat_connectivity_matrices.shape[1])))

        # store flat connectivity matrices in df
        flat_connectivity_matrices = pd.DataFrame(
            flat_connectivity_matrices,
            index=matrix_ids,
            dtype=dtype
        )
        
        # Compute mapconn curves using the flat connectivity data and map data or mappercentile data
        mapconn_curves, mappct_data, flat_mappercentile_masks = calculate_mapconn(
            flat_connectivity_matrices=flat_connectivity_matrices, 
            mappct_masks_flat=flat_mappercentile_masks, 
            map_data=map_data,
            map_data_is_pct=map_data_is_pct,
            r_to_z=r_to_z,
            conn_agg=conn_aggregation,
            percentiles=percentiles,
            mappercentile_threshold=mappercentile_threshold,
            return_mappct=True,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype
        )

        # Create and return the MAPCONN instance
        return cls(flat_connectivity_matrices=flat_connectivity_matrices, 
                   flat_mappercentile_masks=flat_mappercentile_masks, 
                   mapconn_curves=mapconn_curves, 
                   parcel_labels=parcel_labels,
                   n_parcels=len(parcel_labels),
                   map_data=map_data if not map_data_is_pct else None,
                   mappct_data=mappct_data,
                   mappct_thresh=mappercentile_threshold, 
                   conn_aggregation=conn_aggregation,
                   r_to_z=r_to_z,
                   n_jobs=n_jobs,
                   dtype=dtype,
                   get_stats=True)

    @classmethod
    def from_matrix(cls, 
                    connectivity_matrices: Union[np.ndarray, pd.DataFrame, List[np.ndarray]], 
                    map_data: Optional[Union[np.ndarray, pd.DataFrame, xr.DataArray]] = None, 
                    map_data_is_pct: bool = False, 
                    flat_mappercentile_masks: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                    matrix_ids: Optional[Sequence[Any]] = None, 
                    parcel_labels: Optional[Sequence[Any]] = None, 
                    r_to_z: bool = False, 
                    percentiles: Optional[Sequence[float]] = None,
                    mappercentile_threshold: MapPctThreshold = "overequal", 
                    conn_aggregation: ConnAggregation = "mean", 
                    n_jobs: int = -1, 
                    verbose: bool = True, 
                    dtype: Union[np.dtype, type] = np.float32
                    ) -> "MapConn":
        """
        Create an instance from connectivity matrices.

        `connectivity_matrices` can be a list or array with shape
        $(n_{ids}, n_{parcels}, n_{parcels})$.
        """
        
        if percentiles is None:
            percentiles = np.arange(0, 100, 5)
        # perform input validation on 2d-format connectivity matrices
        if isinstance(connectivity_matrices, (np.ndarray, pd.DataFrame)):
            if connectivity_matrices.ndim == 2:
                connectivity_matrices = [np.array(connectivity_matrices)]
            elif connectivity_matrices.ndim == 3:
                connectivity_matrices = [connectivity_matrices[i,:,:] for i in range(connectivity_matrices.shape[0])]
        if connectivity_matrices[0].shape[-1] != connectivity_matrices[0].shape[-2]:
            raise ValueError(f"connectivity_matrices must be symmetric (not shape {connectivity_matrices[0].shape}); "
                             "if 3d or list of 2d arrays, first dimension is assumed to be the number of matrices")
        
        # flatten
        conn_data_flat = _vectorize_sym_matrices(connectivity_matrices, discard_diagonal=True)
        conn_data_flat = conn_data_flat.astype(dtype)
                   
        # get parcel labels
        if parcel_labels is None:
            if isinstance(connectivity_matrices, pd.DataFrame):
                parcel_labels = connectivity_matrices.columns.to_list()

        # return MAPCONN object
        return cls.from_flat_matrix(flat_connectivity_matrices=conn_data_flat, 
                                    matrix_ids=matrix_ids,
                                    r_to_z=r_to_z,
                                    map_data=map_data,
                                    map_data_is_pct=map_data_is_pct,
                                    flat_mappercentile_masks=flat_mappercentile_masks,
                                    percentiles=percentiles,
                                    mappercentile_threshold=mappercentile_threshold,
                                    conn_aggregation=conn_aggregation,
                                    parcel_labels=parcel_labels,
                                    n_jobs=n_jobs,
                                    verbose=verbose,
                                    dtype=dtype)
        
        
    @classmethod
    def from_timeseries(cls, 
                        timeseries_data: Union[np.ndarray, pd.DataFrame, xr.DataArray, List[Any]], 
                        map_data: Optional[Union[np.ndarray, pd.DataFrame, xr.DataArray]] = None, 
                        map_data_is_pct: bool = False, 
                        flat_mappercentile_masks: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                        connectivity_estimator: Union[str, Dict[str, Any]] = 'correlation', 
                        zscore: bool = True, 
                        timeseries_ids: Optional[Sequence[Any]] = None, 
                        parcel_labels: Optional[Sequence[Any]] = None,
                        percentiles: Optional[Sequence[float]] = None, 
                        mappercentile_threshold: MapPctThreshold = "overequal", 
                        conn_aggregation: ConnAggregation = "mean",
                        n_jobs: int = -1, 
                        verbose: bool = True, 
                        dtype: Union[np.dtype, type] = np.float32
                        ) -> "MapConn":
        """
        Create an instance from time series data.

        `timeseries_data` may be a list of 2D arrays with shape
        $(n_{time}, n_{parcels})$ or a 3D array with shape
        $(n_{ids}, n_{time}, n_{parcels})$.
        """
        
        if percentiles is None:
            percentiles = np.arange(0, 100, 5)
        # input validation
        if parcel_labels is None:
            if isinstance(timeseries_data, pd.DataFrame):
                parcel_labels = timeseries_data.columns.to_list()
            elif isinstance(timeseries_data[0], pd.DataFrame):
                parcel_labels = timeseries_data[0].columns.to_list()
        if isinstance(timeseries_data, (np.ndarray, pd.DataFrame, xr.DataArray)):
            if timeseries_data.ndim == 1 or timeseries_data.ndim > 3:
                raise ValueError("timeseries_data must be a 2D or 3D array")
            elif timeseries_data.ndim == 2:
                timeseries_data = [np.array(timeseries_data)]
            elif timeseries_data.ndim == 3:
                timeseries_data = [np.array(timeseries_data[i,:,:]) for i in range(timeseries_data.shape[0])]
        
        # z standardize
        if zscore:
            timeseries_data = [(timeseries_data[i] - np.mean(timeseries_data[i])) / np.std(timeseries_data[i]) 
                               for i in range(len(timeseries_data))]
                
        # estimate correlation matrices
        if isinstance(connectivity_estimator, str):
            if connectivity_estimator.lower() in ["pearson", "correlation", "corr"]:
                connectivity_estimator = {"method": "empiricalcovariance", "kind": "covariance", "normalize": True}
            elif connectivity_estimator.lower() in ["partial_pearson", "partial_correlation", "partial_corr"]:
                connectivity_estimator = {"method": "empiricalcovariance", "kind": "precision", "normalize": True}
        else:
            if not isinstance(connectivity_estimator, dict):
                raise ValueError("connectivity_estimator must be a dictionary or a predefined string")
            else:
                if any(k not in connectivity_estimator.keys() for k in ["method", "kind", "normalize"]):
                    raise ValueError("connectivity_estimator dictionary must contain 'method', 'kind', and 'normalize'")
        estimator = _get_matrix_estimator(**connectivity_estimator, dtype=dtype)
        conn_data = Parallel(n_jobs=n_jobs)(
            delayed(estimator)(timeseries_data[i]) 
            for i in tqdm(range(len(timeseries_data)), disable=not verbose, desc="Calculating connectivity matrices")
        )
        
        # run MAPCONN.from_matrix()
        return cls.from_matrix(connectivity_matrices=conn_data, 
                               matrix_ids=timeseries_ids,
                               r_to_z=True if connectivity_estimator["normalize"] else False,
                               map_data=map_data,
                               map_data_is_pct=map_data_is_pct,
                               flat_mappercentile_masks=flat_mappercentile_masks,
                               conn_aggregation=conn_aggregation,
                               percentiles=percentiles,
                               mappercentile_threshold=mappercentile_threshold,
                               parcel_labels=parcel_labels,
                               n_jobs=n_jobs,
                               verbose=verbose,
                               dtype=dtype)
        

class MapConnInv:
    """
    Class for the mapconn inverted test.
    """
    
    def __init__(self, 
                 mapconn_instance: Optional["MapConn"] = None,
                 mapconn_inverted_instance: Optional["MapConn"] = None,
                 mapconn_pvalues_perm: Optional[Dict[str, pd.DataFrame]] = None,
                 mapconn_pvalues_perm_norm: Optional[Dict[str, pd.DataFrame]] = None,
                 mapconn_pvalues_ttest: Optional[Dict[str, pd.DataFrame]] = None,
                 n_jobs: int = -1,
                 n_perm: int = 10000,
                 dtype: Union[np.dtype, type] = np.float32,
                 get_stats: bool = False,
                 get_pvalues: bool = False
                 ) -> None:
        """
        Initialize a MapConnInv instance.

        Stores original and inverted MapConn objects and optional p-value results.
        """
        self._mapconn_instance = mapconn_instance
        self._mapconn_inverted_instance = mapconn_inverted_instance
        self._mapconn_pvalues_perm = mapconn_pvalues_perm
        self._mapconn_pvalues_perm_norm = mapconn_pvalues_perm_norm
        self._mapconn_pvalues_ttest = mapconn_pvalues_ttest
        self._dtype = dtype
        self._n_jobs = n_jobs
        self._n_perm = n_perm
        
        # precompute
        if get_stats:
            self.get_stats()
        if get_pvalues:
            self.get_pvalues(permutation=True, norm=False, n_perm=n_perm)
            self.get_pvalues(permutation=True, norm=True, n_perm=n_perm)
            self.get_pvalues(permutation=False)
         
    def get_original(self) -> "MapConn":
        """ 
        Returns the mapconn instance stored in the instance.
        """
        return self._mapconn_instance
    
    def get_inverted(self) -> "MapConn":
        """
        Returns the mapconn inverted instance stored in the instance.
        """
        return self._mapconn_inverted_instance
    
    def get_map_data(self, **kwargs) -> pd.DataFrame:
        """
        Passed through to the original mapconn instance. See MapConn.get_map_data() for details.
        """
        return self._mapconn_instance.get_map_data(**kwargs)
    
    def get_inverted_map_data(self, **kwargs) -> pd.DataFrame:
        """
        Passed through to the inverted mapconn instance. See MapConn.get_map_data() for details.
        """
        return self._mapconn_inverted_instance.get_map_data(**kwargs)
    
    def get_connectivity_matrices(self, **kwargs) -> Union[pd.DataFrame, List[np.ndarray]]:
        """
        Passed through to the original mapconn instance. See MapConn.get_connectivity_matrices() for details.
        """
        return self._mapconn_instance.get_connectivity_matrices(**kwargs)
    
    def get_mappercentile_masks(self, **kwargs) -> pd.DataFrame:
        """
        Passed through to the original mapconn instance. See MapConn.get_mappercentile_masks() for details.
        """
        return self._mapconn_instance.get_mappercentile_masks(**kwargs)
    
    def get_curves(self, **kwargs) -> pd.DataFrame:
        """
        Passed through to the original mapconn instance. See MapConn.get_curves() for details.
        """
        return self._mapconn_instance.get_curves(**kwargs)
    
    def get_inverted_curves(self, **kwargs) -> pd.DataFrame:
        """
        Passed through to the inverted mapconn instance. See MapConn.get_curves() for details.
        """
        return self._mapconn_inverted_instance.get_curves(**kwargs)
    
    def get_stats(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the original mapconn instance. See MapConn.get_stats() for details.
        """
        return self._mapconn_instance.get_stats(**kwargs)
    
    def get_inverted_stats(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the inverted mapconn instance. See MapConn.get_stats() for details.
        """
        return self._mapconn_inverted_instance.get_stats(**kwargs)
    
    def get_delta_stats(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Difference between get_stats() of original and inverted mapconn instances.
        """
        original = self._mapconn_instance.get_stats(**(kwargs | {"force_dict": True}))
        inverted = self._mapconn_inverted_instance.get_stats(**(kwargs | {"force_dict": True}))
        stats = list(original.keys())
        
        delta = {stat: original[stat] - inverted[stat] for stat in stats}
        
        if not kwargs.get("force_dict", False):
            if len(stats) == 1:
                delta = delta[stats[0]]
        return delta
    
    def get_loo(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the original mapconn instance. See MapConn.get_loo() for details.
        """
        return self._mapconn_instance.get_loo(**kwargs)
    
    def get_inverted_loo(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the inverted mapconn instance. See MapConn.get_loo() for details.
        """
        return self._mapconn_inverted_instance.get_loo(**kwargs)
    
    def get_delta_loo(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        LOO calculated for delta between original and inverted mapconn stats.
        Note: this will result in the same as `get_loo() - get_inverted_loo()` but will likely be 
        faster if the other two are not needed.
        """
        kwargs["delta"] = True
        kwargs["mapconn_stats"] = self.get_delta_stats(
            stats=kwargs.get("stats", ["auc", "poly2"]),
            maps=kwargs.get("maps", None),
            percentiles=kwargs.get("percentiles", None),
            ids=kwargs.get("ids", None),
            recalculate=kwargs.get("recalculate", False),
            force_dict=True
        )
        return self._mapconn_instance.get_loo(**kwargs)
    
    def get_matrix_sac(self, distmat: np.ndarray, ids: Optional[Sequence[Any]] = None, 
                       **kwargs) -> pd.DataFrame:
        """
        Passed through to the original mapconn instance. See MapConn.get_matrix_sac() for details.
        """
        return self._mapconn_instance.get_matrix_sac(ids=ids, distmat=distmat, **kwargs)
    
    def get_pvalues(self, 
                    stats: Optional[Union[str, Sequence[str]]] = None, 
                    maps: Optional[Sequence[Any]] = None, 
                    percentiles: Optional[Sequence[float]] = None, 
                    ids: Optional[Sequence[Any]] = None, 
                    permutation: Union[bool, str] = True, 
                    norm: bool = False, 
                    n_perm: Optional[int] = None, 
                    n_jobs: int = -1, 
                    perm_strategy: str = "proportional", 
                    tail: str = "upper", 
                    recalculate: bool = False, 
                    force_dict: bool = False, 
                    seed: Optional[int] = None, 
                    verbose: bool = False
                    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Calculate p-values comparing original vs inverted curves.

        Returns DataFrame(s) indexed by statistic (rows) and maps (columns).
        """
        
        # check number of fc matrices
        if len(self._mapconn_instance._ids) < 2:
            raise ValueError("We need at least two input matrices to compute p-values, because p values "
                             "are calculated as permutation or paired t-tests between original and inverted curves.")
        
        # n_perm
        if n_perm is None:
            n_perm = self._n_perm
            
        # stats
        if stats is None:
            stats = ["auc", "poly2"]
        if stats == "all":
            stats = STATS
        elif isinstance(stats, str):
            stats = [stats]
            
        # get stored pvalues (can be None)
        if permutation:
            if norm:
                pvalues = self._mapconn_pvalues_perm_norm
            else:
                pvalues = self._mapconn_pvalues_perm
        else:
            pvalues = self._mapconn_pvalues_ttest
        
        # check if recalculate is needed
        if pvalues is None or recalculate or (any(stat not in pvalues.keys() for stat in stats)):
            recalculate = True
        elif pvalues is not None:
            # check if all stats and data are available
            if not all(stat in pvalues.keys() for stat in stats):
                recalculate = True
            # get original mapconn stats for reference
            mapconn_stats = self._mapconn_instance.get_stats(
                stats=stats, maps=maps, percentiles=percentiles, ids=ids, force_dict=True
            )
            # check if columns are the same
            if not np.array_equal(pvalues[stats[0]].columns, 
                                  mapconn_stats[stats[0]].columns):
                recalculate = True
        
        # recalculate if needed
        if recalculate:
            pvalues = {}
            
            # get original mapconn stats
            mapconn_stats = self._mapconn_instance.get_stats(
                stats=stats, maps=maps, percentiles=percentiles, ids=ids, force_dict=True
            )
            
            # get inverted mapconn stats
            mapconn_inverted_stats = self._mapconn_inverted_instance.get_stats(
                stats=stats, maps=maps, percentiles=percentiles, ids=ids, force_dict=True
            )
            
            # number of "subjects"
            n = mapconn_stats[stats[0]].shape[0]
            
            # prep permutation
            if permutation is True:
                permutation = "label"
            if permutation == "label":
                # "groups"
                groups = np.concatenate([np.zeros(n), np.ones(n)])
                subjects = mapconn_stats[stats[0]].index.to_list() + mapconn_stats[stats[0]].index.to_list()
                
                # permuted groups            
                groups_perm = permute_groups(groups, subjects=subjects, paired=True, strategy=perm_strategy,
                                             n_perm=n_perm, n_proc=n_jobs, seed=seed, verbose=verbose)
            elif permutation == "sign":
                rng = np.random.RandomState(seed)
                signs_perm = [rng.choice([-1, 1], size=n) for _ in range(n_perm)]
            
            # iterate over stats
            for stat in set(stats).intersection(set(mapconn_stats.keys())):
                
                # original
                original = mapconn_stats[stat]
                # inverted
                inverted = mapconn_inverted_stats[stat]
                # maps
                maps = mapconn_stats[stat].columns
                
                # calculate p-values
                if permutation:
                    pvalues_stat = pd.DataFrame(
                        columns=maps,
                        index=["stat", "p"]
                    )
                    # iterate over maps
                    for m in tqdm(maps, desc="Calculating p-values"):
                        
                        if permutation == "label":
                            # data
                            data = np.concatenate([original[m].values, inverted[m].values])
                            # original 
                            d_original = cohen_paired(data[groups == 0], data[groups == 1])
                            # null
                            d_null = [
                                cohen_paired(data[g == 0], data[g == 1])
                                for g in groups_perm
                            ]
                            
                        elif permutation == "sign":
                            # original
                            diff = original[m].values - inverted[m].values
                            d_original = np.mean(diff) / np.std(diff, ddof=1)
                            
                            # null
                            d_null = [
                                np.mean(diff * s) / np.std(diff * s, ddof=1)
                                for s in signs_perm
                            ]
                            
                        # p-value
                        pvalues_stat.loc["stat", m] = d_original
                        pvalues_stat.loc["p", m] = null_to_p(test_value=d_original, null_array=d_null, 
                                                            tail=tail, fit_norm=norm)
                else:
                    # ttest
                    ttest = ttest_rel(
                        original, 
                        inverted, 
                        axis=0, 
                        nan_policy="raise", 
                        alternative="greater" if tail == "upper" else "less" if tail == "lower" else "two-sided"
                    )
                    # result
                    pvalues_stat = pd.DataFrame(
                        {
                            "stat": ttest.statistic,
                            "p": ttest.pvalue
                        },
                        index=maps
                    ).T
                    
                # store
                pvalues[stat] = pvalues_stat
                    
            # store
            if permutation:
                if norm:
                    self._mapconn_pvalues_perm_norm = pvalues
                else:
                    self._mapconn_pvalues_perm = pvalues
            else:
                self._mapconn_pvalues_ttest = pvalues
            
        # return
        pvalues = {stat: pvalues[stat] for stat in stats}
        if not force_dict:
            if len(pvalues.keys()) == 1:
                pvalues = pvalues[stats[0]]
                
        return pvalues
    
    def get_summary(self, 
                    level: str = "group", 
                    stats: Optional[Union[str, Sequence[str]]] = None, 
                    maps: Optional[Sequence[Any]] = None, 
                    percentiles: Optional[Sequence[float]] = None, 
                    ids: Optional[Sequence[Any]] = None, 
                    agg_stats: Optional[Sequence[str]] = None, 
                    reduce_index: bool = True
                    ) -> pd.DataFrame:
        """
        Returns concatenated dataframes of all available summary data (no curves).
        """
        if level not in ["group", "individual"]:
            raise ValueError(f"level must be one of 'group' | 'individual', got {level}")
        
        # get stats and inverted stats
        if agg_stats is None:
            agg_stats = ["mean", "std", "min", "max"]
        kwargs = {"stats": stats, "maps": maps, "percentiles": percentiles, "ids": ids, "force_dict": True}
        stats_by_metric = {
            "original": self.get_stats(**kwargs),
            "inverted": self.get_inverted_stats(**kwargs),
            "delta": self.get_delta_stats(**kwargs)
        }
        
        # concatenate
        df_list = []
        for stat in stats_by_metric["original"]:
            for metric in stats_by_metric:
                df_list.append(
                    stats_by_metric[metric][stat]
                    .reset_index(names="id")
                    .assign(curve_stat=stat, metric=metric, variable="val")
                )
        df = pd.concat(df_list, axis=0).set_index(["curve_stat", "metric", "variable", "id"])
        
        # group by stat and metric
        if level == "group":
            df = (
                df.groupby(["curve_stat", "metric"], sort=False)
                .agg(agg_stats)
                .stack(future_stack=True)
            )
            # TODO: add inverted vs original ttest/permutation stats and p value?
            df.index.names = df.index.names[:-1] + ["variable"]
            
        # return
        if reduce_index:
            df = reduce_df_index(df)
        return df
    
    def _ensure_results(self) -> None:
        """
        Ensure standard p-value results are computed and cached.

        Runs permutation (raw and normalized) and paired t-test p-values
        so that `.save()` can persist a fully populated instance, even if source data were dropped.
        """
        self.get_pvalues(permutation=True, norm=False)
        self.get_pvalues(permutation=True, norm=True)
        self.get_pvalues(permutation=False)
    
    def save(self, path: Union[str, Path], ensure_results: bool = True) -> None:
        """
        Pickle the mapconn instance to a file.
        """
        
        if ensure_results:
            self._ensure_results()
        
        path = Path(path)
        save_gzip = path.suffix == ".gz"
        if save_gzip:
            with gzip.open(path, "wb", compresslevel=9) as f:
                pickle.dump(self, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)
                
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MapConnInv":
        """
        Load the mapconn instance from a pickled file.
        """
        path = Path(path)
        save_gzip = path.suffix == ".gz"
        if save_gzip:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
        
    @classmethod
    def from_mapconn(cls, 
                     mapconn_instance: "MapConn", 
                     map_data_inverted: Optional[pd.DataFrame] = None,
                     verbose: bool = True, 
                     dtype: Optional[Union[np.dtype, type]] = None, 
                     n_jobs: Optional[int] = None, 
                     get_stats: bool = True, 
                     get_pvalues: bool = False, 
                     n_perm: int = 10000
                     ) -> "MapConnInv":
        """
        Create a MapConnInv instance from an "original" MapConn instance.
        """
        
        # checks
        if not isinstance(mapconn_instance, MapConn):
            raise ValueError("mapconn_instance must be a MapConn instance")
        if not hasattr(mapconn_instance, "_map_data"):
            raise ValueError("mapconn_instance must have original map data stored in ._map_data")
        
        # get map data
        map_data = mapconn_instance.get_map_data(pct=False).copy()
        
        # dtype
        if dtype is None:
            dtype = mapconn_instance._dtype
            
        # n_jobs
        if n_jobs is None:
            n_jobs = mapconn_instance._n_jobs
            
        # inverted map data
        if map_data_inverted is None:
            map_data = map_data.T
            map_data_mean = np.nanmean(map_data)
            map_data_inverted = (map_data - map_data_mean) * (-1) + map_data_mean
            map_data_inverted = map_data_inverted.T      
            
        # run
        mapconn_inverted = MapConn.from_flat_matrix(
            flat_connectivity_matrices=mapconn_instance.get_connectivity_matrices(flat=True),
            map_data=map_data_inverted,
            map_data_is_pct=False,
            matrix_ids=mapconn_instance._ids,
            parcel_labels=mapconn_instance._parcel_labels,
            r_to_z = mapconn_instance._r_to_z,
            conn_aggregation=mapconn_instance._conn_agg,
            percentiles=mapconn_instance._percentiles,
            mappercentile_threshold=mapconn_instance._mappct_thresh,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype
        )
        
        # return
        return cls(mapconn_instance=mapconn_instance,
                   mapconn_inverted_instance=mapconn_inverted,
                   n_jobs=n_jobs,
                   dtype=dtype,
                   n_perm=n_perm,
                   get_stats=get_stats,
                   get_pvalues=get_pvalues)
        
    @classmethod
    def from_flat_matrix(cls, flat_connectivity_matrices: Union[np.ndarray, pd.DataFrame], 
                         map_data: Optional[Union[np.ndarray, pd.DataFrame, xr.DataArray]] = None, 
                         map_data_is_pct: bool = False, 
                         flat_mappercentile_masks: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                         matrix_ids: Optional[Sequence[Any]] = None, 
                         parcel_labels: Optional[Sequence[Any]] = None, 
                         r_to_z: bool = False, 
                         percentiles: Optional[Sequence[float]] = None,
                         mappercentile_threshold: MapPctThreshold = "overequal", 
                         conn_aggregation: ConnAggregation = "mean", 
                         map_data_inverted: Optional[pd.DataFrame] = None,
                         get_stats: bool = True,
                         get_pvalues: bool = False,
                         n_perm: int = 10000,
                         n_jobs: int = -1, 
                         verbose: bool = True, 
                         dtype: Union[np.dtype, type] = np.float32
                         ) -> "MapConnInv":
        """
        Create an instance of MapConnInv from flattened connectivity matrices.
        """
       
        if percentiles is None:
            percentiles = np.arange(0, 100, 5)
        # get original mapconn instance
        mapconn_instance = MapConn.from_flat_matrix(
            flat_connectivity_matrices=flat_connectivity_matrices,
            map_data=map_data,
            map_data_is_pct=map_data_is_pct,
            flat_mappercentile_masks=flat_mappercentile_masks,
            matrix_ids=matrix_ids,
            parcel_labels=parcel_labels,
            r_to_z=r_to_z,
            percentiles=percentiles,
            mappercentile_threshold=mappercentile_threshold,
            conn_aggregation=conn_aggregation,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype
        )
        
        # return MapConnInv instance
        return cls.from_mapconn(
            mapconn_instance=mapconn_instance,
            map_data_inverted=map_data_inverted,
            get_stats=get_stats,
            get_pvalues=get_pvalues,
            n_perm=n_perm,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype
        )
        
    @classmethod
    def from_matrix(cls, connectivity_matrices: Union[np.ndarray, pd.DataFrame, List[np.ndarray]], 
                    map_data: Optional[Union[np.ndarray, pd.DataFrame, xr.DataArray]] = None, 
                    map_data_is_pct: bool = False, 
                    flat_mappercentile_masks: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                    matrix_ids: Optional[Sequence[Any]] = None, 
                    parcel_labels: Optional[Sequence[Any]] = None, 
                    r_to_z: bool = False, 
                    percentiles: Optional[Sequence[float]] = None,
                    mappercentile_threshold: MapPctThreshold = "overequal", 
                    conn_aggregation: ConnAggregation = "mean", 
                    map_data_inverted: Optional[pd.DataFrame] = None,
                    get_stats: bool = True, 
                    get_pvalues: bool = False, 
                    n_perm: int = 10000,
                    n_jobs: int = -1, 
                    verbose: bool = True, 
                    dtype: Union[np.dtype, type] = np.float32
                    ) -> "MapConnInv":
        """
        Create an instance of MapConnInv from connectivity matrices.
        """
        
        if percentiles is None:
            percentiles = np.arange(0, 100, 5)
        # get original mapconn instance
        mapconn_instance = MapConn.from_matrix(
            connectivity_matrices=connectivity_matrices,
            map_data=map_data,
            map_data_is_pct=map_data_is_pct,
            flat_mappercentile_masks=flat_mappercentile_masks,
            matrix_ids=matrix_ids,
            parcel_labels=parcel_labels,
            r_to_z=r_to_z,
            percentiles=percentiles,
            mappercentile_threshold=mappercentile_threshold,
            conn_aggregation=conn_aggregation,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype
        )
        
        # return MapConnInv instance
        return cls.from_mapconn(
            mapconn_instance=mapconn_instance,
            map_data_inverted=map_data_inverted,
            get_stats=get_stats,
            get_pvalues=get_pvalues,
            n_perm=n_perm,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype
        )
          
    @classmethod
    def from_timeseries(cls, timeseries_data: Union[np.ndarray, pd.DataFrame, xr.DataArray, List[Any]], 
                        map_data: Optional[Union[np.ndarray, pd.DataFrame, xr.DataArray]] = None, 
                        map_data_is_pct: bool = False, 
                        flat_mappercentile_masks: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                        connectivity_estimator: Union[str, Dict[str, Any]] = 'correlation', 
                        zscore: bool = True, 
                        timeseries_ids: Optional[Sequence[Any]] = None, 
                        parcel_labels: Optional[Sequence[Any]] = None, 
                        percentiles: Optional[Sequence[float]] = None,
                        mappercentile_threshold: MapPctThreshold = "overequal", 
                        conn_aggregation: ConnAggregation = "mean", 
                        map_data_inverted: Optional[pd.DataFrame] = None,
                        get_stats: bool = True, 
                        get_pvalues: bool = False, 
                        n_perm: int = 10000,
                        n_jobs: int = -1, 
                        verbose: bool = True, 
                        dtype: Union[np.dtype, type] = np.float32
                        ) -> "MapConnInv":
        """
        Create an instance of MapConnInv from time series data.
        """
        
        if percentiles is None:
            percentiles = np.arange(0, 100, 5)
        # get original mapconn instance
        mapconn_instance = MapConn.from_timeseries(
            timeseries_data=timeseries_data,
            map_data=map_data,
            map_data_is_pct=map_data_is_pct,
            flat_mappercentile_masks=flat_mappercentile_masks,
            connectivity_estimator=connectivity_estimator,
            zscore=zscore,
            timeseries_ids=timeseries_ids,
            parcel_labels=parcel_labels,
            percentiles=percentiles,
            mappercentile_threshold=mappercentile_threshold,
            conn_aggregation=conn_aggregation,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype
        )
        
        # return MapConnInv instance
        return cls.from_mapconn(
            mapconn_instance=mapconn_instance,
            map_data_inverted=map_data_inverted,
            get_stats=get_stats,
            get_pvalues=get_pvalues,
            n_perm=n_perm,
            n_jobs=n_jobs,
            verbose=verbose,
            dtype=dtype
        )
        
        
class MapConnNull:
    """
    Class for generating null distributions of map connectivity data.
    """
    
    def __init__(self, 
                 mapconn_instance: Optional[Union["MapConn", "MapConnInv"]] = None,
                 map_data_null: Optional[List[np.ndarray]] = None,
                 mapconn_null_curves: Optional[List[np.ndarray]] = None, 
                 mapconn_null_curves_dist: Optional[pd.DataFrame] = None,
                 mapconn_null_stats: Optional[Dict[str, List[pd.DataFrame]]] = None,
                 mapconn_null_stats_dist_group: Optional[Dict[str, pd.DataFrame]] = None,
                 mapconn_null_stats_dist_indiv: Optional[Dict[str, pd.DataFrame]] = None,
                 mapconn_pvalues: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
                 n_nulls: Optional[int] = None,
                 distmat: Optional[np.ndarray] = None,
                 n_jobs: int = -1,
                 dtype: Union[np.dtype, type] = np.float32,
                 get_results: bool = False
                 ) -> None:
        """
        Initialize a MapConnNull instance.

        Stores null maps/curves and derived null distributions.
        """
        self._mapconn_instance = mapconn_instance
        self._map_data_null = map_data_null
        self._mapconn_null_curves = mapconn_null_curves
        self._mapconn_null_curves_dist = mapconn_null_curves_dist
        self._mapconn_null_stats = mapconn_null_stats
        self._mapconn_null_stats_dist_group = mapconn_null_stats_dist_group
        self._mapconn_null_stats_dist_indiv = mapconn_null_stats_dist_indiv
        self._mapconn_pvalues = mapconn_pvalues if mapconn_pvalues is not None else {}
        self._distmat = distmat
        self._dtype = dtype
        self._n_jobs = n_jobs
        
        # delta stats-related
        self._mapconn_inverted_null_curves = None
        self._mapconn_inverted_null_stats = None
        self._mapconn_pvalues_delta = None
        self._mapconn_delta_null_stats_dist_group = None
        self._mapconn_delta_null_stats_dist_indiv = None
        
        # n nulls
        if n_nulls is None:
            n_nulls = len(mapconn_null_curves)
        self._n_nulls = n_nulls
        
        # precompute
        if get_results:
            self._ensure_results(include_delta=isinstance(self._mapconn_instance, MapConnInv))
         
    def get_original(self) -> Union["MapConn", "MapConnInv"]:
        """ 
        Returns the mapconn instance stored in the instance.
        """
        return self._mapconn_instance
    
    def get_map_data(self, **kwargs) -> pd.DataFrame:
        """
        Passed through to the original mapconn instance. See MapConn.get_map_data() for details.
        """
        return self._mapconn_instance.get_map_data(**kwargs)
    
    def get_null_map_data(self, **kwargs) -> pd.DataFrame:
        """
        Return null map data for the instance.

        This accessor is not implemented for `MapConnNull` and will raise
        `NotImplementedError` if called.
        """
        raise NotImplementedError
    
    def get_connectivity_matrices(self, **kwargs) -> Union[pd.DataFrame, List[np.ndarray]]:
        """
        Passed through to the original mapconn instance. See MapConn.get_connectivity_matrices() for details.
        """
        return self._mapconn_instance.get_connectivity_matrices(**kwargs)
    
    def get_mappercentile_masks(self, **kwargs) -> pd.DataFrame:
        """
        Passed through to the original mapconn instance. See MapConn.get_mappercentile_masks() for details.
        """
        return self._mapconn_instance.get_mappercentile_masks(**kwargs)
    
    def get_curves(self, **kwargs) -> pd.DataFrame:
        """
        Passed through to the original mapconn instance. See MapConn.get_curves() for details.
        """
        return self._mapconn_instance.get_curves(**kwargs)
    
    def get_inverted_curves(self, **kwargs) -> pd.DataFrame:
        """
        Passed through to the original mapconn instance. See MapConnInv.get_inverted_curves() for details.
        """
        if not isinstance(self._mapconn_instance, MapConnInv):
            raise ValueError("mapconn_instance must be a MapConnInv instance")
        return self._mapconn_instance.get_inverted_curves(**kwargs)
    
    def get_stats(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the original mapconn instance. See MapConn.get_stats() for details.
        """
        return self._mapconn_instance.get_stats(**kwargs)
    
    def get_inverted_stats(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the original mapconn instance. See MapConnInv.get_inverted_stats() for details.
        """
        if not isinstance(self._mapconn_instance, MapConnInv):
            raise ValueError("mapconn_instance must be a MapConnInv instance")
        return self._mapconn_instance.get_inverted_stats(**kwargs)
    
    def get_delta_stats(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the original mapconn instance. See MapConnInv.get_delta_stats() for details.
        """
        if not isinstance(self._mapconn_instance, MapConnInv):
            raise ValueError("mapconn_instance must be a MapConnInv instance")
        return self._mapconn_instance.get_delta_stats(**kwargs)
    
    def get_loo(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the original mapconn instance. See MapConn.get_loo() for details.
        """
        return self._mapconn_instance.get_loo(**kwargs)
    
    def get_inverted_loo(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the inverted mapconn instance. See MapConn.get_loo() for details.
        """
        if not isinstance(self._mapconn_instance, MapConnInv):
            raise ValueError("mapconn_instance must be a MapConnInv instance")
        return self._mapconn_instance.get_inverted_loo(**kwargs)
    
    def get_delta_loo(self, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Passed through to the inverted mapconn instance. See MapConn.get_delta_loo() for details.
        """
        if not isinstance(self._mapconn_instance, MapConnInv):
            raise ValueError("mapconn_instance must be a MapConnInv instance")
        return self._mapconn_instance.get_delta_loo(**kwargs)
    
    def get_null_curves(self, 
                        maps: Optional[Sequence[Any]] = None, 
                        percentiles: Optional[Sequence[float]] = None, 
                        ids: Optional[Sequence[Any]] = None, 
                        return_df: bool = True, 
                        remove_global: bool = True, 
                        inverted: bool = False
                        ) -> Union[List[np.ndarray], List[pd.DataFrame]]:
        """
        Return null curves, optionally as DataFrames aligned to observed curves.

        Each element corresponds to one null sample.
        """
        if not inverted:
            null_curves = self._mapconn_null_curves
        else:
            null_curves = self._mapconn_inverted_null_curves
        if null_curves is None:
            raise AttributeError(
                "No null curves found! Have they not been calculated or dropped from the instance?\n"
                "Try `.get_null_curves_dist()` to obtain distribution statistics for null curves.\n"
                "Or try `.get_null_stats_dist()` to obtain distribution statistics for null stats.")
            
        obs_full = self._mapconn_instance.get_curves(remove_global=False)
        obs_sel = self._mapconn_instance.get_curves(maps=maps, percentiles=percentiles, ids=ids, remove_global=False)
        row_idc_full = obs_full.index.to_list()
        row_idc_sel = [row_idc_full.index(l) for l in obs_sel.index]
        col_idc_full = obs_full.columns.to_list()
        col_idc_sel = [col_idc_full.index(l) for l in obs_sel.columns]
        null = [arr[row_idc_sel, :][:, col_idc_sel] for arr in null_curves]
        if return_df:
            if remove_global:
                null = [
                    _remove_global(pd.DataFrame(arr, index=obs_sel.index, columns=obs_sel.columns)) 
                    for arr in null
                ]
            else:
                null = [
                    pd.DataFrame(arr, index=obs_sel.index, columns=obs_sel.columns) 
                    for arr in null
                ]
        return null
    
    def get_null_curves_dist(self, maps: Optional[Sequence[Any]] = None, 
                             percentiles: Optional[Sequence[float]] = None, 
                             ids: Optional[Sequence[Any]] = None, remove_global: bool = True, 
                             recalculate: bool = False, dist_stats_from_mean: bool = True, 
                             dist_stats_quantiles: Optional[Sequence[float]] = None
                             ) -> pd.DataFrame:
        """
        Get distribution statistics for null mapconn curves.

        If `dist_stats_from_mean=True`, statistics are computed on the
        mean across ids for each null. Returns a DataFrame indexed by
        distribution metrics with columns `(map, pct)`.
        """
        
        if dist_stats_quantiles is None:
            dist_stats_quantiles = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
        if self._mapconn_null_curves_dist is None or recalculate:
                    
            # get null stats
            mapconn_null_curves = self.get_null_curves(
                maps=maps, percentiles=percentiles, ids=ids, remove_global=remove_global
            )
            
            # TODO: implement stats for each id/subject
            if not dist_stats_from_mean:
                raise NotImplementedError("dist_stats_from_mean must be True, distribution stats can "
                                          "currently only be calculated for mean across subjects")
            
            # calculate null distribution of curves: output dfs (maps * percentiles, dist_stats)
            # null across ids
            null_dist = (
                pd.DataFrame(
                    np.stack(mapconn_null_curves).mean(axis=1),
                    columns=mapconn_null_curves[0].columns,
                )
                .describe(percentiles=dist_stats_quantiles)
            )
            
            # store
            self._mapconn_null_curves_dist = null_dist

        # return
        if maps is None:
            maps = self._mapconn_null_curves_dist.columns.get_level_values("map").unique()
        if percentiles is None:
            percentiles = self._mapconn_null_curves_dist.columns.get_level_values("pct").unique()
        dist_stats = ["count", "mean", "std", "min"] + [f"{q*100}%".replace(".0", "") for q in dist_stats_quantiles] + ["max"]
        return self._mapconn_null_curves_dist.loc[dist_stats, (maps, percentiles)]
    
    
    def get_null_stats(self, 
                       stats: Optional[Union[str, Sequence[str]]] = None, 
                       maps: Optional[Sequence[Any]] = None, 
                       percentiles: Optional[Sequence[float]] = None, 
                       ids: Optional[Sequence[Any]] = None, 
                       recalculate: bool = False, 
                       force_dict: bool = False, 
                       inverted: bool = False, 
                       multilevel_index: bool = False, 
                       remove_global: bool = True
                       ) -> Union[List[pd.DataFrame], Dict[str, List[pd.DataFrame]], pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Compute or return null statistics for each null sample."""
        if stats is None:
            stats = ["auc", "poly2"]
        if isinstance(stats, str):
            stats = [stats]
        if not inverted:
            mapconn_null_stats = self._mapconn_null_stats
        else:
            mapconn_null_stats = self._mapconn_inverted_null_stats
        
        if mapconn_null_stats is None or recalculate or (any(stat not in mapconn_null_stats.keys() for stat in stats)):
            recalculate = True
        elif mapconn_null_stats is not None:
            # get mapconn curves for reference
            mapconn_curves = self._mapconn_instance.get_curves(maps=maps, percentiles=percentiles, ids=ids, remove_global=True)    
            # check if all stats and data are available
            if not all(stat in mapconn_null_stats.keys() for stat in stats):
                recalculate = True
            if not np.array_equal(mapconn_null_stats[stats[0]][0].index, 
                                  mapconn_curves.index):
                recalculate = True
            if not np.array_equal(mapconn_null_stats[stats[0]][0].columns.unique(), 
                                  mapconn_curves.columns.get_level_values("map").unique()):
                recalculate = True
                
        if recalculate:
            mapconn_null_curves = self.get_null_curves(maps=maps, percentiles=percentiles, ids=ids, remove_global=False,
                                                       inverted=inverted)
            mapconn_null_stats = []
            for mapconn_null_curves_i in mapconn_null_curves:
                mapconn_null_stats.append(
                    _calc_mapconn_stats(mapconn_null_curves_i, stats=stats, force_dict=True, remove_global=remove_global)
                )
            mapconn_null_stats = {stat: [null[stat] for null in mapconn_null_stats] 
                                  for stat in mapconn_null_stats[0].keys()}
            
            if not inverted:
                self._mapconn_null_stats = mapconn_null_stats
            else:
                self._mapconn_inverted_null_stats = mapconn_null_stats
            
        mapconn_null_stats = {stat: mapconn_null_stats[stat] for stat in stats}
        if multilevel_index:
            mapconn_null_stats = {
                stat: pd.concat(mapconn_null_stats[stat], axis=0, 
                                keys=range(len(mapconn_null_stats[stat])), names=["null", "id"]) 
                for stat in stats
            }
        if not force_dict:
            if len(mapconn_null_stats.keys()) == 1:
                mapconn_null_stats = mapconn_null_stats[stats[0]]
                
        return mapconn_null_stats
    
    def get_delta_null_stats(self, 
                             stats: Optional[Union[str, Sequence[str]]] = None, 
                             maps: Optional[Sequence[Any]] = None, 
                             percentiles: Optional[Sequence[float]] = None, 
                             ids: Optional[Sequence[Any]] = None, 
                             recalculate: bool = False, 
                             force_dict: bool = False
                             ) -> Union[List[pd.DataFrame], Dict[str, List[pd.DataFrame]]]:
        """
        Compute delta null stats (original minus inverted) for each null sample.

        Returns list or dict of DataFrames aligned to observed maps.
        """
        if stats is None:
            stats = ["auc", "poly2"]
        # get null stats
        kwargs = dict(stats=stats, maps=maps, percentiles=percentiles, ids=ids, recalculate=recalculate, force_dict=True)
        null_stats_original = self.get_null_stats(**kwargs)
        null_stats_inverted = self.get_null_stats(**kwargs, inverted=True)
        stats = list(null_stats_original.keys())
        n_nulls = len(null_stats_original[stats[0]])
        
        # calculate delta stats
        null_stats_delta = {}
        for stat in stats:
            null_stats_delta[stat] = [
                null_stats_original[stat][i] - null_stats_inverted[stat][i]
                for i in range(n_nulls)
            ]
        
        # quick enough so storage not necessary
        
        # return
        if not force_dict:
            if len(null_stats_delta.keys()) == 1:
                null_stats_delta = null_stats_delta[stats[0]]
        return null_stats_delta
            
    def get_null_stats_dist(self, 
                            stats: Optional[Union[str, Sequence[str]]] = None, 
                            maps: Optional[Sequence[Any]] = None, 
                            percentiles: Optional[Sequence[float]] = None, 
                            ids: Optional[Sequence[Any]] = None, 
                            recalculate: bool = False, 
                            dist_stats_from_mean: bool = True,
                            dist_stats_quantiles: Optional[Sequence[float]] = None,
                            force_dict: bool = False, 
                            null_stats_dict: Optional[Dict[str, List[pd.DataFrame]]] = None
                            ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get distribution statistics for null curve statistics.

        If `dist_stats_from_mean=True`, statistics are computed on the
        mean across ids for each null. Returns DataFrame(s) with maps as
        columns and distribution metrics as the index (and optionally id).
        """
        if stats is None:
            stats = ["auc", "poly2"]
        if dist_stats_quantiles is None:
            dist_stats_quantiles = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
        if isinstance(stats, str):
            stats = [stats]
        
        if dist_stats_from_mean:
            mapconn_null_stats_dist = self._mapconn_null_stats_dist_group
        else:
            mapconn_null_stats_dist = self._mapconn_null_stats_dist_indiv
            
        # check stat
        if mapconn_null_stats_dist is not None:
            if not all(stat in mapconn_null_stats_dist.keys() for stat in stats):
                recalculate = True
        
        # run
        if mapconn_null_stats_dist is None or recalculate or null_stats_dict is not None:
            
            # get null stats
            if null_stats_dict is None:
                mapconn_null_stats = self.get_null_stats(
                    stats=stats, maps=maps, percentiles=percentiles, ids=ids,
                    recalculate=recalculate == "all", force_dict=True,
                    
                )
            else:
                mapconn_null_stats = null_stats_dict
            
            # indices
            stats = list(mapconn_null_stats.keys())
            maps = mapconn_null_stats[stats[0]][0].columns
            ids = mapconn_null_stats[stats[0]][0].index
            
            # calculate null distribution of stats: output is dict per stat with dfs (maps, i)
            null_dist = {}
            for stat in stats:
                
                # null data stacked as a numpy array with shape (n_nulls, n_subs, n_maps)
                null = np.stack(mapconn_null_stats[stat])

                if dist_stats_from_mean:
                    # null is averaged across ids
                    null_mean = null.mean(axis=1)
                    null_dist[stat] = (
                        pd.DataFrame(null_mean, columns=maps)
                        .describe(percentiles=dist_stats_quantiles)
                        .rename_axis(index="variable")
                    )
                    null_dist[stat].loc["mad"] = median_abs_deviation(null_mean, axis=0)
                else:
                    # null for each id/subject
                    null_dist[stat] = {}
                    for i, id in enumerate(ids):
                        n = null[:, i, :]
                        null_dist[stat][id] = (
                            pd.DataFrame(n, columns=maps).describe(percentiles=dist_stats_quantiles)
                        )
                        null_dist[stat][id].loc["mad"] = median_abs_deviation(n, axis=0)
                    null_dist[stat] = pd.concat(null_dist[stat], names=["id", "variable"])
                    
            # store
            if null_stats_dict is None:
                if dist_stats_from_mean:
                    self._mapconn_null_stats_dist_group = null_dist
                else:
                    self._mapconn_null_stats_dist_indiv = null_dist
            else:
                return null_dist
            
        # return
        if dist_stats_from_mean:
            mapconn_null_stats_dist = self._mapconn_null_stats_dist_group
        else:
            mapconn_null_stats_dist = self._mapconn_null_stats_dist_indiv
        if maps is None:
            maps = mapconn_null_stats_dist[stats[0]].columns    
        dist_stats = ["count", "mean", "std", "mad", "min"] + [f"{q*100}%".replace(".0", "") for q in dist_stats_quantiles] + ["max"]
        null_dist = {
            stat: mapconn_null_stats_dist[stat].loc[dist_stats if dist_stats_from_mean else (slice(None), dist_stats), maps]
            for stat in stats
        }
        
        # return
        null_dist = {stat: null_dist[stat] for stat in stats}
        if not force_dict:
            if len(null_dist.keys()) == 1:
                null_dist = null_dist[stats[0]]
        
        return null_dist
    
    
    def get_delta_null_stats_dist(self, 
                                  stats: Optional[Union[str, Sequence[str]]] = None, 
                                  maps: Optional[Sequence[Any]] = None, 
                                  percentiles: Optional[Sequence[float]] = None, 
                                  ids: Optional[Sequence[Any]] = None, 
                                  recalculate: bool = False,
                                  dist_stats_from_mean: bool = True, 
                                  dist_stats_quantiles: Optional[Sequence[float]] = None,
                                  force_dict: bool = False
                                  ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Compute distribution statistics for delta null stats.

        Returns DataFrame(s) of summary metrics across null samples.
        """
        # TODO: ensure that subsetting works
        if stats is None:
            stats = ["auc", "poly2"]
        if dist_stats_quantiles is None:
            dist_stats_quantiles = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
        if isinstance(stats, str):
            stats = [stats]
        
        if dist_stats_from_mean:
            null_stats_delta = self._mapconn_delta_null_stats_dist_group
        else:
            null_stats_delta = self._mapconn_delta_null_stats_dist_indiv
            
        if null_stats_delta is None or recalculate:
            
            # get null stats
            null_stats_delta = self.get_delta_null_stats(
                stats=stats, maps=maps, percentiles=percentiles, ids=ids,
                recalculate=recalculate == "all", force_dict=True)
            
            # get null stats dist
            null_stats_dist = self.get_null_stats_dist(
                null_stats_dict=null_stats_delta,
                stats=stats, maps=maps, percentiles=percentiles, ids=ids,
                dist_stats_from_mean=dist_stats_from_mean, dist_stats_quantiles=dist_stats_quantiles,
                recalculate=recalculate, force_dict=True)
            
            # store
            if dist_stats_from_mean:
                self._mapconn_delta_null_stats_dist_group = null_stats_dist
            else:
                self._mapconn_delta_null_stats_dist_indiv = null_stats_dist
                
        # return
        if dist_stats_from_mean:
            null_stats_delta = self._mapconn_delta_null_stats_dist_group
        else:
            null_stats_delta = self._mapconn_delta_null_stats_dist_indiv
        if maps is None:
            maps = null_stats_delta[stats[0]].columns
        dist_stats = ["count", "mean", "std", "mad", "min"] + [f"{q*100}%".replace(".0", "") for q in dist_stats_quantiles] + ["max"]
        null_dist = {
            stat: null_stats_delta[stat].loc[dist_stats if dist_stats_from_mean else (slice(None), dist_stats), maps]
            for stat in stats
        }
        
        # return
        null_dist = {stat: null_dist[stat] for stat in stats}
        if not force_dict:
            if len(null_dist.keys()) == 1:
                null_dist = null_dist[stats[0]]
        
        return null_dist
    
    def get_matrix_sac(self, distmat: Optional[np.ndarray] = None, 
                       ids: Optional[Sequence[Any]] = None, **kwargs) -> pd.DataFrame:
        """
        Passed through to the original mapconn instance. See MapConn.get_matrix_sac() for details.
        If distmat is not provided, the distmat stored in the MapConnNull instance is used.
        """
        if distmat is None:
            distmat = self._distmat
        return self._mapconn_instance.get_matrix_sac(ids=ids, distmat=distmat, **kwargs)
    
    def get_pvalues(self, stats: Optional[Union[str, Sequence[str]]] = None, 
                    maps: Optional[Sequence[Any]] = None, percentiles: Optional[Sequence[float]] = None, 
                    ids: Optional[Sequence[Any]] = None, p_from_mean: bool = True,
                    inverted: bool = False, norm: bool = False, tail: str = "upper", 
                    recalculate: bool = False, n_jobs: Optional[int] = None, force_dict: bool = False
                    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Compute p-values comparing observed stats to null distributions.

        Returns DataFrame(s) indexed by id (or "mean") with map columns.
        """
        
        if stats is None:
            stats = ["auc", "poly2"]
        if stats == "all":
            stats = STATS
        elif isinstance(stats, str):
            stats = [stats]
        
        # TODO: decide if p values for delta stats are needed. requires special null map runs (computation time!)
        
        if tail not in ["upper", "lower", "two"]:
            raise ValueError("tail must be one of 'upper', 'lower', 'two'")
        
        # n_jobs
        if n_jobs is None:
            n_jobs = self._n_jobs
            
        # get stored pvalues (can be None)
        l = "group" if p_from_mean else "individual"
        d = "exact" if not norm else "norm"
        m = "obs" if not inverted else "inv"
        t = tail
        pvalues_key = f"map-{m}_level-{l}_dist-{d}_tail-{t}"
        pvalues = self._mapconn_pvalues.get(pvalues_key, None)
        
        # kwargs to load stats
        get_stats_kwargs = dict(stats=stats, maps=maps, percentiles=percentiles, ids=ids, force_dict=True)
        
        # check if recalculate is needed
        if pvalues is None or recalculate or (any(stat not in pvalues.keys() for stat in stats)):
            recalculate = True
        elif pvalues is not None:
            # check if all stats and data are available
            if not all(stat in pvalues.keys() for stat in stats):
                recalculate = True
            # get original mapconn stats for reference
            mapconn_stats = self._mapconn_instance.get_stats(**get_stats_kwargs)
            # check if p_from_mean is needed
            if p_from_mean:
                if not pvalues[stats[0]].index[0] == "mean":
                    recalculate = True
            else:
                if not np.array_equal(pvalues[stats[0]].index, 
                                      mapconn_stats[stats[0]].index):
                    recalculate = True
            if not np.array_equal(pvalues[stats[0]].columns, 
                                  mapconn_stats[stats[0]].columns):
                recalculate = True
        
        # recalculate if needed
        if recalculate:
            pvalues = {}
            
            # get original mapconn stats
            if not inverted:
                mapconn_stats = self._mapconn_instance.get_stats(**get_stats_kwargs)
            else:
                mapconn_stats = self._mapconn_instance.get_inverted_stats(**get_stats_kwargs)
            # elif direction == "delta":
            #     mapconn_stats = self._mapconn_instance.get_delta_stats(**get_stats_kwargs)
            
            # get null mapconn stats
            mapconn_null_stats = self.get_null_stats(**get_stats_kwargs)
            
            # iterate over stats
            for stat in set(stats).intersection(set(mapconn_stats.keys())):
                
                # original
                obs = np.array(mapconn_stats[stat])
                if p_from_mean:
                    obs = obs.mean(axis=0, keepdims=True)
                # null
                null = np.stack(mapconn_null_stats[stat], axis=0)
                if p_from_mean:
                    null = np.mean(null, axis=1, keepdims=True)
                    
                # calculate p-values in parallel
                maps = mapconn_stats[stat].columns
                ids = mapconn_stats[stat].index if not p_from_mean else ["mean"]
                pvalues_stat = Parallel(n_jobs=n_jobs)(
                    delayed(null_to_p)(
                        test_value=obs[i_idx, i_m],
                        null_array=null[:, i_idx, i_m],
                        tail=tail,
                        fit_norm=norm
                    )
                    for i_idx, idx in enumerate(ids)
                    for i_m, m in enumerate(maps)
                )
                
                # store
                pvalues[stat] = pd.DataFrame(
                    np.reshape(pvalues_stat, (len(ids), len(maps))),
                    columns=maps,
                    index=ids
                )
                
            # store
            self._mapconn_pvalues[pvalues_key] = pvalues
        
        # return
        pvalues = {stat: pvalues[stat] for stat in stats}
        if not force_dict:
            if len(pvalues.keys()) == 1:
                pvalues = pvalues[stats[0]]
                
        return pvalues
    
    def get_delta_pvalues(self, 
                          stats: Optional[Union[str, Sequence[str]]] = None, 
                          maps: Optional[Sequence[Any]] = None, 
                          percentiles: Optional[Sequence[float]] = None, 
                          ids: Optional[Sequence[Any]] = None, 
                          p_from_mean: bool = True,
                          verbose: bool = True, 
                          tail: str = "two", 
                          recalculate: bool = False,
                          n_jobs: Optional[int] = None, 
                          force_dict: bool = False
                          ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        # TODO: implement get methods
        # TODO: optimize and add parallelization
        # TODO: add p value options: norm, not norm, tails (?), individual, group
        """
        Get permuted p-values for delta statistics.
        Will re-run null analysis, so takes as much time as "fitting" of the model did.
        """
        if not isinstance(self._mapconn_instance, MapConnInv):
            raise ValueError("Requires MapConnNull instance created from MapConnInv instance")
        # TODO: implement p-values for individual matrices
        if not p_from_mean:
            raise NotImplementedError("p_from_mean must be True, p-values for delta stats can "
                                      "currently only be calculated for mean across subjects")
        # TODO: check if subsetting is implemented correctly
        if any(arg is not None for arg in [maps, percentiles, ids]):
            if verbose:
                logger.warning("Subsetting might not be implemented correctly for delta p-values")
        
        # stats
        if stats is None:
            stats = ["auc", "poly2"]
        if isinstance(stats, str):
            stats = [stats]
        
        # check if already calculated
        pvalues = self._mapconn_pvalues_delta
        if pvalues is not None and not recalculate:
            
            # check if all stats and data are available
            if not all(stat in pvalues.keys() for stat in stats):
                recalculate = True
            
            # TODO: add rest of subsetting checks
            else:
                
                # return
                pvalues = {stat: pvalues[stat] for stat in set(stats).intersection(pvalues.keys())}
                if not force_dict:
                    if len(pvalues.keys()) == 1:
                        pvalues = pvalues[stats[0]]
                return pvalues
            
        # calculate: checks
        if self._map_data_null is None:
            raise ValueError("Null maps not available. Have they been dropped?")
        if self._mapconn_null_stats is None:
            raise ValueError("Null stats not available. Have they been dropped?")
   
        # get original MapConn instance
        mapconn_instance = self._mapconn_instance._mapconn_instance
             
        # n_jobs
        if n_jobs is None:
            n_jobs = self._n_jobs
        
        # get delta
        delta_stats = self._mapconn_instance.get_delta_stats(
            stats=stats, maps=maps, percentiles=percentiles, ids=ids, force_dict=True)
        stats = list( set(stats).intersection(delta_stats.keys()) )
        maps = delta_stats[stats[0]].columns
        ids = delta_stats[stats[0]].index
        if percentiles is None:
            percentiles = mapconn_instance._percentiles
        
        # get null maps
        map_data_null = self._map_data_null
        n_nulls = len(map_data_null)
 
        # calculate inverted null maps
        if verbose:
            logger.info("Calculating inverted null maps")
        map_data_null_inverted = []
        for arr in map_data_null:
            arr_mean = np.nanmean(arr)
            arr = (arr - arr_mean) * (-1) + arr_mean
            map_data_null_inverted.append(arr)
            
        # rerun null analysis
        # get null mapconn curves
        flat_connectivity_matrices = mapconn_instance.get_connectivity_matrices(flat=True)
        r_to_z = mapconn_instance._r_to_z
        conn_agg = mapconn_instance._conn_agg
        mappct_thresh = mapconn_instance._mappct_thresh
        dtype = mapconn_instance._dtype
        null_curves_inverted = Parallel(n_jobs=n_jobs)(
            delayed(calculate_mapconn)(
                flat_connectivity_matrices, 
                map_data=map_data_null_i, 
                map_data_is_pct=False,
                percentiles=percentiles, 
                return_mappct=False,
                return_df=False,
                r_to_z=r_to_z,
                conn_agg=conn_agg,
                mappercentile_threshold=mappct_thresh,
                n_jobs=1, 
                verbose=False,
                dtype=dtype,
            )
            for map_data_null_i 
            in tqdm(map_data_null_inverted, disable=not verbose, desc="Calculating inverted null mapconn curves")
        )
        # save
        self._mapconn_inverted_null_curves = null_curves_inverted
        # get null curves for inverted curves
        null_curves_inverted = self.get_null_curves(
            maps=maps, percentiles=percentiles, ids=ids, inverted=True)

        # calculate null delta stats
        if verbose:
            logger.info("Calculating null delta stats")
        null_stats_delta = self.get_delta_null_stats(
            stats=stats, maps=maps, percentiles=percentiles, ids=ids, force_dict=True)
        
        # calculate p-values
        if verbose:
            logger.info("Calculating p-values")
        pvalues = {}
        for stat in stats:
            pvalues[stat] = [
                null_to_p(
                    test_value=delta_stats[stat].values[:, i_m].mean(),
                    null_array=[null_stats_delta[stat][i].values[:, i_m].mean() for i in range(n_nulls)],
                    tail=tail,
                    fit_norm=False
                )
                for i_m, m in enumerate(maps)
            ]
            pvalues[stat] = pd.DataFrame(
                np.array(pvalues[stat])[np.newaxis, :],
                columns=maps,
                index=["mean"]
            )
        
        # save
        self._mapconn_pvalues_delta = pvalues
        
        # return
        pvalues = {stat: pvalues[stat] for stat in stats}
        if not force_dict:
            if len(stats) == 1:
                pvalues = pvalues[stats[0]]
        return pvalues
    
    
    def get_summary(self, 
                    level: str = "group", 
                    stats: Optional[Union[str, Sequence[str]]] = None, 
                    maps: Optional[Sequence[Any]] = None, 
                    percentiles: Optional[Sequence[float]] = None, 
                    ids: Optional[Sequence[Any]] = None,
                    agg_stats: Optional[Sequence[str]] = None, 
                    reduce_index: bool = True
                    ) -> pd.DataFrame:
        """
        Returns concatenated dataframes of all available summary data (no curves).
        """
        get_kwargs = {"maps": maps, "percentiles": percentiles, "ids": ids}
        
        if agg_stats is None:
            agg_stats = ["mean", "std", "min", "max"]
        df_summary = self._mapconn_instance.get_summary(
            level=level, stats=stats, agg_stats=agg_stats, reduce_index=False, **get_kwargs
        )
        stats = df_summary.index.get_level_values("curve_stat").unique()
        metrics = df_summary.index.get_level_values("metric").unique()

        df = []
        for stat in stats:
            for metric in metrics: # TODO: delta?
                
                # individual
                if level == "individual":
                    
                    # results from summary
                    df.append(
                        df_summary.loc[(stat, metric, "val", slice(None)), :]
                        .reset_index()
                        #.assign(variable="val")
                        .set_index(["curve_stat", "metric", "variable", "id"])
                    )
                    
                    # p values
                    for p in ["p", "pz"]:
                        df.append(
                            self.get_pvalues(
                                stats=stat, 
                                p_from_mean=False, 
                                inverted=True if metric == "inverted" else False,
                                norm=True if p == "pz" else False,
                                **get_kwargs
                            )
                            .reset_index(names="id")
                            .assign(curve_stat=stat, metric=metric, variable=p)
                            .set_index(["curve_stat", "metric", "variable", "id"])
                        )    
                        
                    # null distribution
                    if metric in ["original", "delta"]:
                        if metric == "original":
                            tmp = self.get_null_stats_dist(
                                dist_stats_from_mean=False, stats=stat, **get_kwargs) 
                        else:
                            try:
                                tmp = self.get_delta_null_stats_dist(
                                    dist_stats_from_mean=False, stats=stat, **get_kwargs)
                            except Exception as e:
                                # TODO: handle this better
                                continue
                        df.append(
                            tmp
                            .reset_index() # makes columns: id and variable
                            .assign(variable=lambda x: "null_" + x.variable,
                                    curve_stat=stat,
                                    metric=metric)
                            .set_index(["curve_stat", "metric", "variable", "id"])
                        )
                    
                # group
                else:
                    # results from summary
                    df.append(
                        df_summary.loc[(stat, metric, slice(None), slice(None)), :]
                    )
                    
                    # p values
                    if not metric == "delta":
                        for p in ["p", "pz"]:
                            df.append(
                                self.get_pvalues(
                                    stats=stat, 
                                    inverted=True if metric == "inverted" else False,
                                    norm=True if p == "pz" else False,
                                    **get_kwargs
                                )
                                .assign(curve_stat=stat, metric=metric, variable=p)
                                .set_index(["curve_stat", "metric", "variable"])
                            ) 
                    else:
                        if self._mapconn_pvalues_delta is not None:
                            df.append(
                                self.get_delta_pvalues(
                                    stats=stat,
                                    **get_kwargs
                                )
                                .assign(curve_stat=stat, metric=metric, variable="p")
                                .set_index(["curve_stat", "metric", "variable"])
                            )
                    
                    # null distribution
                    if metric in ["original", "delta"]:
                        if metric == "original":
                            tmp = self.get_null_stats_dist(stats=stat, **get_kwargs) 
                        else:
                            try:
                                tmp = self.get_delta_null_stats_dist(stats=stat, **get_kwargs)
                            except Exception as e:
                                # TODO: handle this better
                                continue
                        df.append(
                            tmp
                            .assign(variable=lambda x: "null_" + x.index,
                                    curve_stat=stat,
                                    metric=metric)
                            .set_index(["curve_stat", "metric", "variable"])
                        )
                        
        df = pd.concat(df, axis=0)
        
        # adjusted stats
        var_order = df.index.get_level_values("variable").unique().to_list()
        for stat in stats:
            for metric in metrics:
                metric_null = "original" if metric != "delta" else "delta"
                if level == "group":
                    mean = df.loc[(stat, metric, "mean"), :]
                    try:
                        null_mean = df.loc[(stat, metric_null, "null_mean"), :]
                        null_std = df.loc[(stat, metric_null, "null_std"), :]
                        null_med = df.loc[(stat, metric_null, "null_50%"), :]
                        null_mad = df.loc[(stat, metric_null, "null_mad"), :]
                    except KeyError:
                        continue
                    df.loc[(stat, metric, "mean_z"), :] = (mean.values - null_mean.values) / null_std.values
                    df.loc[(stat, metric, "mean_rz"), :] = (mean.values - null_med.values) / (1.4826 * null_mad.values)
                elif level == "individual":
                    val = df.loc[(stat, metric, "val", slice(None)), :]
                    try:
                        null_mean = df.loc[(stat, metric_null, "null_mean", slice(None)), :]
                        null_std = df.loc[(stat, metric_null, "null_std", slice(None)), :]
                        null_med = df.loc[(stat, metric_null, "null_50%", slice(None)), :]
                        null_mad = df.loc[(stat, metric_null, "null_mad", slice(None)), :]
                    except KeyError:
                        continue
                    df = pd.concat([
                        df,
                        ( (val - null_mean.values) / null_std.values ).rename(index={"val": "val_z"}),
                        ( (val - null_med.values) / (1.4826 * null_mad.values) ).rename(index={"val": "val_rz"})
                    ])
        if df.index.get_level_values("variable").isin(["mean_z", "val_z"]).any():
            var_order.insert(1, "mean_z" if level == "group" else "val_z")
            var_order.insert(1, "mean_rz" if level == "group" else "val_rz")
            df = df.loc[stats, metrics, var_order]
        
        if reduce_index:
            df = reduce_df_index(df)
        return df
    
    def _ensure_results(self, stats: Optional[Union[str, Sequence[str]]] = None, 
                        include_delta: bool = False) -> None:
        """
        Ensure null stats and p-values are computed for standard outputs.

        Populates cached null curves/statistics and p-values so downstream
        accessors (and `.save()`) do not require recomputation.
        """
        # TODO: handle stats types
        # TODO: add delta stats
        if stats is None:
            stats = ["auc", "poly2"]
        self.get_stats(stats=stats)
        for p_from_mean, norm, inverted in product([True, False], [True, False], [True, False]):
            try:
                self.get_pvalues(stats=stats, p_from_mean=p_from_mean, norm=norm, inverted=inverted)
            except AttributeError:
                pass
        self.get_null_curves_dist()
        self.get_null_stats_dist(dist_stats_from_mean=True, stats=stats)
        self.get_null_stats_dist(dist_stats_from_mean=False, stats=stats)
        
        # TODO: handle delta stats better
        try:
            self.get_delta_pvalues(stats=stats)
            self.get_delta_null_stats_dist(dist_stats_from_mean=True, stats=stats)
            self.get_delta_null_stats_dist(dist_stats_from_mean=False, stats=stats)
        except Exception as e:
            pass
        
    
    def drop_nulls(self, ensure_results: bool = True, keep_null_stats: bool = False, 
                   stats: Optional[Union[str, Sequence[str]]] = None) -> None:
        """
        Drop nulls from the mapconn instance.
        """
        # TODO: add delta stats
        if stats is None:
            stats = ["auc", "poly2"]
        if ensure_results:
            self._ensure_results(include_delta=True, stats=stats)
       
        self._map_data_null = None
        self._mapconn_null_curves = None
        self._mapconn_inverted_null_curves = None
        if not keep_null_stats:
            self._mapconn_null_stats = None
            self._mapconn_inverted_null_stats = None

    def save(self, path: Union[str, Path], drop_nulls: bool = True, 
             ensure_results: bool = True) -> None:
        """
        Pickle the mapconn instance to a file.
        """
        if drop_nulls:
            self.drop_nulls(ensure_results=ensure_results)
            
        path = Path(path)
        save_gzip = path.suffix == ".gz"
        if save_gzip:
            with gzip.open(path, "wb", compresslevel=9) as f:
                pickle.dump(self, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)
                
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MapConnNull":
        """
        Load the mapconn instance from a pickled file.
        """
        path = Path(path)
        save_gzip = path.suffix == ".gz"
        if save_gzip:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
        
    @classmethod
    def from_mapconn(cls, mapconn_instance: Union["MapConn", "MapConnInv"], 
                     map_data_null: Optional[List[np.ndarray]] = None,
                     parcellation: Optional[Any] = None, 
                     parcellation_space: str = "mni152", 
                     distmat: Optional[np.ndarray] = None, 
                     n_nulls: int = 1000, 
                     n_jobs: Optional[int] = None, 
                     seed: Optional[int] = None, 
                     verbose: bool = True, 
                     null_verbose: bool = False, 
                     dtype: Optional[Union[np.dtype, type]] = None, 
                     get_results: bool = True, 
                     **kwargs
                     ) -> "MapConnNull":
        """
        Create a MapConnNull instance from an "original" MapConn instance.
        Arguments to modify null generation:
        - lr_mirror_dist_mat: mirror distance matrix across left and right hemispheres
        - lr_mirror_null_maps: mirror null maps across left and right hemispheres
        - match_interhemi_correlation: match interhemispheric correlation of null maps to original maps
        - cx_sc_minmax_scale: scale subcortical and cortical parcels independently to range in original map data
        - parc_idc_lh: indices of left hemisphere parcels, necessary for above arguments 1-3
        - parc_idc_rh: indices of right hemisphere parcels, necessary for above arguments 1-3
        - parc_idc_sc: indices of subcortical parcels, necessary for above arguments 4
        - l2rmap: left-to-right mapping for parcellation, necessary for above arguments 2-3 in case of non-symmetric parcellation
        - parc_symmetric: whether parcellation is symmetric, set to True to enable above arguments 1-3 without l2rmap
        """
        
        # checks and handle MapConnInv
        if not isinstance(mapconn_instance, (MapConn, MapConnInv)):
            raise ValueError("mapconn_instance must be a MapConn or MapConnInv instance")
        elif isinstance(mapconn_instance, MapConnInv):
            mapconn_instance_input = mapconn_instance
            mapconn_instance = mapconn_instance.get_original()
        else:
            mapconn_instance_input = mapconn_instance
        if not hasattr(mapconn_instance, "_map_data"):
            raise ValueError("mapconn_instance must have original map data stored in ._map_data")
        if parcellation is None and distmat is None and map_data_null is None:
            raise ValueError("Either parcellation or distance matrix or map_data_null must be provided")
        
        # get map data
        map_data = mapconn_instance.get_map_data(pct=False)
        
        # dtype
        if dtype is None:
            dtype = mapconn_instance._dtype
            
        # n_jobs
        if n_jobs is None:
            n_jobs = mapconn_instance._n_jobs
            
        # get flat connectivity data
        flat_connectivity_matrices = mapconn_instance.get_connectivity_matrices(flat=True)
        r_to_z = mapconn_instance._r_to_z
        
        # get percentiles
        percentiles = mapconn_instance._percentiles
        
        # aggregation method
        conn_agg = mapconn_instance._conn_agg
        
        # map percentile threshold
        mappct_thresh = mapconn_instance._mappct_thresh
        
        # settings for null generation
        null_kwargs = {
            "method": "moran",
            "lr_mirror_dist_mat": False,
            "lr_mirror_null_maps": False,
            "match_interhemi_correlation": False,
            "l2rmap": None,
            "parc_idc_lh": None,
            "parc_idc_rh": None,
            "parc_idc_sc": None,
            "cx_sc_minmax_scale": False,
        } | kwargs
    
        # get null data
        if map_data_null is None:
            map_data_null, distmat = generate_null_maps(
                data=map_data,
                parcellation=parcellation,
                parc_space=parcellation_space,
                dist_mat=distmat,
                n_nulls=n_nulls,
                seed=seed,
                n_proc=n_jobs,
                verbose=null_verbose,
                dtype=dtype,
                **null_kwargs
            )
            map_data_null = [
                np.stack([map_data_null[m][i,:] for m in map_data_null.keys()], dtype=dtype)
                for i in range(n_nulls)
            ]
            
        # get null mapconn curves
        mapconn_null_curves = Parallel(n_jobs=n_jobs)(
            delayed(calculate_mapconn)(
                flat_connectivity_matrices, 
                map_data=map_data_null_i, 
                map_data_is_pct=False,
                percentiles=percentiles, 
                return_mappct=False,
                return_df=False,
                r_to_z=r_to_z,
                conn_agg=conn_agg,
                mappercentile_threshold=mappct_thresh,
                n_jobs=1, 
                verbose=False,
                dtype=dtype,
            )
            for map_data_null_i 
            in tqdm(map_data_null, disable=not verbose, desc="Calculating null mapconn curves")
        )
    
        # return
        return cls(mapconn_instance=mapconn_instance_input,
                   map_data_null=map_data_null,
                   mapconn_null_curves=mapconn_null_curves,  
                   distmat=distmat,
                   n_nulls=n_nulls,
                   n_jobs=n_jobs,
                   dtype=dtype,
                   get_results=get_results)
        
        
# mapconn curves
def calculate_mapconn(flat_connectivity_matrices: Union[np.ndarray, pd.DataFrame], 
                      map_data: Optional[Union[np.ndarray, pd.DataFrame, xr.DataArray]] = None, 
                      map_data_is_pct: bool = False, 
                      mappct_masks_flat: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                      r_to_z: bool = False, 
                      square: bool = False, 
                      percentiles: Optional[Sequence[float]] = None, 
                      mappercentile_threshold: MapPctThreshold = "overequal", 
                      conn_agg: ConnAggregation = "mean", 
                      return_mappct: bool = False, 
                      return_df: bool = True, 
                      n_jobs: int = -1, 
                      verbose: bool = True, 
                      dtype: Union[np.dtype, type] = np.float32
                      ) -> Union[np.ndarray, pd.DataFrame, Tuple[pd.DataFrame, Any, Any]]:
    """
    Compute mapconn curves from flattened connectivity data and map data.

    Parameters:
    - `flat_connectivity_matrices`: shape $(n_{ids}, n_{edges})$.
    - `map_data`: shape $(n_{maps}, n_{parcels})$.
    - `percentiles`: values in $[0, 100]$.

    Returns:
    - DataFrame with shape $(n_{ids}, n_{maps} \times n_{percentiles})$ when `return_df=True`.
    """
    if percentiles is None:
        percentiles = np.arange(0, 100, 5)
    conn_data_flat = np.array(flat_connectivity_matrices, dtype=dtype)
    
    if mappct_masks_flat is not None:
        map_data = None
        
    # checks
    if conn_data_flat.ndim != 2:
        raise ValueError("conn_data must be a 2D array (flattened lower triangle of matrix/matrices)")
    if map_data is not None:
        if map_data.ndim != 2:
            raise ValueError("map_data must be a 2D array")
        if conn_data_flat.shape[1] != _n_sym_matrix_tri_elem_from_shape(map_data.shape[1]):
            raise ValueError("flat_connectivity_matrices and map_data must have corresponding number of columns. "
                             f"flat_connectivity_matrices.shape: {conn_data_flat.shape}, map_data.shape: is: {map_data.shape}, "
                             f"should be: (N, {_n_sym_matrix_tri_elem_from_shape(map_data.shape[1])})")
    elif mappct_masks_flat is not None:
        if mappct_masks_flat.ndim != 2:
            raise ValueError("flat_mappercentile_data must be a 2D array")
        if conn_data_flat.shape[1] != mappct_masks_flat.shape[1]:
            raise ValueError("flat_connectivity_matrices and flat_mappercentile_data must have same number of columns (=elements)")
    else:
        raise ValueError("map_data or mappercentile_data_vect must be provided") 
    
    # square
    if square:
        r_to_z = False
        conn_data_flat = conn_data_flat ** 2 * np.sign(conn_data_flat)
    
    # fisher's z transform
    if r_to_z:
        conn_data_flat = np.arctanh(conn_data_flat)
    
    # get map percentiles
    if mappct_masks_flat is None:
        mappct_data, mappct_masks_flat = _calc_mappct_masks(
            map_data, 
            map_data_is_pct=map_data_is_pct,
            percentiles=percentiles, 
            verbose=verbose, 
            pct_threshold=mappercentile_threshold,
            return_df=return_df,
            dtype=dtype
        )
    else:
        mappct_data = None
    mappct_masks_flat_arr = np.array(mappct_masks_flat)
        
    # mean/median after applying percentile thresholds
    if conn_agg == "mean":
        threshold_fun = _threshold_conn_data_mean
    elif conn_agg == "median":
        threshold_fun = _threshold_conn_data_median
    else:
        raise ValueError(f"conn_agg must be 'mean' or 'median', got {conn_agg}")
    # run thresholding/aggregation in parallel
    mapconn = Parallel(n_jobs=n_jobs)(
        delayed(threshold_fun)(conn_data_flat, mappct_masks_flat_arr[map_idx, :]) 
        for map_idx 
        in tqdm(range(mappct_masks_flat_arr.shape[0]), disable=not verbose, desc="Calculating mapConn curves")
    )
    mapconn = np.stack(mapconn, axis=1, dtype=dtype)
    
    # sort into df and return
    if return_df:
        mapconn = pd.DataFrame(
            mapconn, 
            index=flat_connectivity_matrices.index 
                if isinstance(flat_connectivity_matrices, pd.DataFrame) else None, 
            columns=mappct_masks_flat.index 
                if isinstance(mappct_masks_flat, pd.DataFrame) else None,
            dtype=dtype
        )
    return mapconn if not return_mappct else (mapconn, mappct_data, mappct_masks_flat)


def _threshold_conn_data_mean(conn_data_flat: np.ndarray, bool_vector: np.ndarray) -> np.ndarray:
    """Compute row-wise mean of thresholded connectivity data."""
    conn_data_thresh = conn_data_flat[:, bool_vector]
    # rows with not only nans
    valid_rows = ~np.all(np.isnan(conn_data_thresh), axis=1)
    # results array
    result = np.full(conn_data_thresh.shape[0], np.nan)
    # calculate mean for valid rows
    result[valid_rows] = np.nanmean(conn_data_thresh[valid_rows], axis=1)
    return result

def _threshold_conn_data_median(conn_data_flat: np.ndarray, bool_vector: np.ndarray) -> np.ndarray:
    """Compute row-wise median of thresholded connectivity data."""
    conn_data_thresh = conn_data_flat[:, bool_vector]
    # rows with not only nans
    valid_rows = ~np.all(np.isnan(conn_data_thresh), axis=1)
    # results array
    result = np.full(conn_data_thresh.shape[0], np.nan)
    # calculate median for valid rows
    result[valid_rows] = np.nanmedian(conn_data_thresh[valid_rows], axis=1)
    return result


