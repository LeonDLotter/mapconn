
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from nilearn.connectome import vec_to_sym_matrix
from nispace.nulls import generate_null_maps
from nispace.stats.misc import null_to_p

from .matrix import (_get_matrix_calculator, _vectorize_sym_matrices,
                     _n_sym_matrix_tri_elem_from_shape, _sym_matrix_shape_from_n_tri_elem)
from .percentiles import _calc_mappct_masks
from .utils import _construct_flat_label_pairs, _mappct_flat_to_parcels
from .stats import _calc_mapconn_stats, _remove_global
from .constants import STATS

class MapConn:
    """
    Class for storing and analyzing map-connectivity data.
    """
    
    def __init__(self, 
                 flat_connectivity_matrices=None, 
                 flat_mappercentile_masks=None, 
                 map_data=None,
                 mappct_data=None,
                 mapconn_curves=None,
                 parcel_labels=None,
                 n_parcels=None,
                 r_to_z=False,
                 mapconn_stats=None,
                 dtype=np.float32):
        """
        Initialize the MapConn class.
        """
        self._conn_data = flat_connectivity_matrices
        self._mappct_masks = flat_mappercentile_masks
        self._map_data = map_data
        self._mappct_data = mappct_data
        self._mapconn_curves = mapconn_curves
        self._parcel_labels = parcel_labels
        self._n_parcels = n_parcels 
        self._r_to_z = r_to_z
        self._mapconn_stats = mapconn_stats
        self._dtype = dtype
        
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
    
    
    def __getitem__(self, key):
        """
        Allows slicing of the mapconn_curves DataFrame directly via the instance.
        """
        return self._mapconn_curves.loc[key]
    
    def get_mapconn_curves(self, maps=None, percentiles=None, ids=None,
                           remove_global=False):
        """
        Returns the mapconn curves stored in the instance.
        """
        mapconn_curves = self._mapconn_curves
        maps = maps if maps is not None else self._maps
        percentiles = percentiles if percentiles is not None else self._percentiles
        ids = ids if ids is not None else self._ids
        
        if remove_global:
            mapconn_curves = _remove_global(mapconn_curves)

        return mapconn_curves.loc[ids, (maps, percentiles)]
    
    def get_parcel_labels(self, flat_pairs=False):
        """
        Returns the parcel labels. If flat_pairs is True, returns parcel pairs corresponding 
        to columns of the flat data format.
        """
        if flat_pairs:
            return _construct_flat_label_pairs(self._parcel_labels, discard_diagonal=True)
        else:
            return self._parcel_labels
    
    def get_connectivity_matrices(self, ids=None, flat=True, flat_col_names=True, fill_diagonal=np.nan):
        """
        Returns the connectivity matrices. If flat is True, returns the flattened (vectorized) form.
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

    def get_mappercentile_masks(self, maps=None, percentiles=None, flat=True, flat_col_names=True):
        """
        Returns the mappercentile data. 
        If flat is True, returns the flat form, which 
        corresponds to the flat connectivity matrices. 
        If flat is False, returns the parcel-format.
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
    
    def get_map_data(self, maps=None, pct=False):
        """
        Returns the map data.
        """
        map_data = self._map_data if not pct else self._mappct_data
        if map_data is None:
            raise ValueError("No map data available")
        maps = maps if maps is not None else self._maps
        return map_data.loc[maps]
    
    def get_mapconn_stats(self, stats="auc", maps=None, percentiles=None, ids=None,
                          recalculate=False):
        """
        Returns the mapconn stats.
        """
        if isinstance(stats, (str, int)):
            stats = [stats]
        if not all(stat in STATS for stat in stats):
            raise ValueError(f"stats must be one or more of {STATS}, got {stats}")
        # get stats
        mapconn_stats = self._mapconn_stats
        # recalculate
        if mapconn_stats is None or recalculate:
            recalculate = True
        # maybe recalculate
        elif mapconn_stats is not None:
            # get mapconn curves for reference
            mapconn_curves = self.get_mapconn_curves(maps=maps, percentiles=percentiles, ids=ids)    
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
            mapconn_curves = self.get_mapconn_curves(maps=maps, percentiles=percentiles, ids=ids)
            mapconn_stats = _calc_mapconn_stats(mapconn_curves, stats=stats, force_dict=True)
            self._mapconn_stats = mapconn_stats
        return mapconn_stats


    @classmethod
    def from_flat_matrix(cls, flat_connectivity_matrices, map_data=None, map_data_is_pct=False, flat_mappercentile_masks=None, 
                         matrix_ids=None, parcel_labels=None, r_to_z=False, percentiles=np.arange(0, 100, 5),
                         mappercentile_threshold="overequal", n_jobs=-1, verbose=True, dtype=np.float32):
        """
        Create an instance of MAPCONN from flattened connectivity matrices.
        """
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
                   r_to_z=r_to_z,
                   dtype=dtype)

    @classmethod
    def from_matrix(cls, connectivity_matrices, map_data=None, map_data_is_pct=False, flat_mappercentile_masks=None, 
                    matrix_ids=None, parcel_labels=None, r_to_z=False, percentiles=np.arange(0, 100, 5),
                    mappercentile_threshold="overequal",
                    n_jobs=-1, verbose=True, dtype=np.float32):
        """
        Estimate mapFC from connectivity matrices.
        """
        
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
                                    parcel_labels=parcel_labels,
                                    n_jobs=n_jobs,
                                    verbose=verbose,
                                    dtype=dtype)
        
        
    @classmethod
    def from_timeseries(cls, timeseries_data, map_data=None, map_data_is_pct=False, flat_mappercentile_masks=None, 
                        connectivity_type='correlation', zscore=True, timeseries_ids=None, parcel_labels=None,
                        percentiles=np.arange(0, 100, 5), mappercentile_threshold="overequal",
                        n_jobs=-1, verbose=True, dtype=np.float32):
        """
        Estimate mapconn from time series data.
        """
        
        # input validation
        if isinstance(timeseries_data, (np.ndarray, pd.DataFrame)):
            timeseries_data = [timeseries_data]
        if timeseries_data[0].ndim != 2:
            raise ValueError("timeseries_data must be a (list of) 2D array(s)")
        if parcel_labels is None:
            if isinstance(timeseries_data, pd.DataFrame):
                parcel_labels = timeseries_data.columns.to_list()
            
        # z standardize
        if zscore:
            timeseries_data = [(timeseries_data[i] - np.mean(timeseries_data[i])) / np.std(timeseries_data[i]) 
                               for i in range(len(timeseries_data))]
                
        # estimate correlation matrices
        if isinstance(connectivity_type, str):
            if connectivity_type.lower() in ["pearson", "correlation", "corr"]:
                connectivity_type = {"method": "empiricalcovariance", "kind": "covariance", "normalize": True}
            elif connectivity_type.lower() in ["partial_pearson", "partial_correlation", "partial_corr"]:
                connectivity_type = {"method": "empiricalcovariance", "kind": "precision", "normalize": True}
        else:
            if not isinstance(connectivity_type, dict):
                raise ValueError("connectivity_type must be a dictionary or a predefined string")
            else:
                if not all(k in ["method", "kind", "normalize"] for k in connectivity_type.keys()):
                    raise ValueError("connectivity_type dictionary must contain 'method', 'kind', and 'normalize'")
        calculator = _get_matrix_calculator(**connectivity_type, dtype=dtype)
        conn_data = Parallel(n_jobs=n_jobs)(
            delayed(calculator)(timeseries_data[i]) 
            for i in tqdm(range(len(timeseries_data)), disable=not verbose, desc="Calculating connectivity matrices")
        )
        
        # run MAPCONN.from_matrix()
        return cls.from_matrix(connectivity_matrices=conn_data, 
                               matrix_ids=timeseries_ids,
                               r_to_z=True if connectivity_type["normalize"] else False,
                               map_data=map_data,
                               map_data_is_pct=map_data_is_pct,
                               flat_mappercentile_masks=flat_mappercentile_masks,
                               percentiles=percentiles,
                               mappercentile_threshold=mappercentile_threshold,
                               parcel_labels=parcel_labels,
                               n_jobs=n_jobs,
                               verbose=verbose,
                               dtype=dtype)
        
class MapConnNull:
    """
    Class for generating null distributions of map connectivity data.
    """
    
    def __init__(self, 
                 mapconn_instance=None,
                 map_data_null=None,
                 mapconn_curves_null=None, 
                 mapconn_stats_null=None,
                 mapconn_p_values=None,
                 n_nulls=None,
                 dtype=np.float32):
        
        self._mapconn_instance = mapconn_instance
        self._mapconn_curves_null = mapconn_curves_null
        self._map_data_null = map_data_null
        self._mapconn_stats_null = mapconn_stats_null
        self._mapconn_p_values = mapconn_p_values
        self._dtype=dtype
        
        # n nulls
        if n_nulls is None:
            n_nulls = len(mapconn_curves_null)
        self._n_nulls = n_nulls
         
    def get_observed(self):
        """ 
        Returns the mapconn instance stored in the instance.
        """
        return self._mapconn_instance
    
    def get_mapconn_curves_null(self, maps=None, percentiles=None, ids=None, return_df=True,
                                remove_global=False):
        obs_full = self._mapconn_instance.get_mapconn_curves()
        obs_sel = self._mapconn_instance.get_mapconn_curves(maps=maps, percentiles=percentiles, ids=ids)
        row_idc_full = obs_full.index.to_list()
        row_idc_sel = [row_idc_full.index(l) for l in obs_sel.index]
        col_idc_full = obs_full.columns.to_list()
        col_idc_sel = [col_idc_full.index(l) for l in obs_sel.columns]
        null = [arr[row_idc_sel, :][:, col_idc_sel] for arr in self._mapconn_curves_null]
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
    
    def get_mapconn_stats_null(self, stats="auc", maps=None, percentiles=None, ids=None,
                               recalculate=False):
        if isinstance(stats, (str, int)):
            stats = [stats]
        mapconn_stats_null = self._mapconn_stats_null
        if mapconn_stats_null is None or recalculate:
            recalculate = True
        if mapconn_stats_null is not None:
            # get mapconn curves for reference
            mapconn_curves = self._mapconn_instance.get_mapconn_curves(maps=maps, percentiles=percentiles, ids=ids)    
            # check if all stats and data are available
            if not all(stat in mapconn_stats_null.keys() for stat in stats):
                recalculate = True
            if not np.array_equal(mapconn_stats_null[stats[0]][0].index, 
                                  mapconn_curves.index):
                recalculate = True
            if not np.array_equal(mapconn_stats_null[stats[0]][0].columns.unique(), 
                                  mapconn_curves.columns.get_level_values("map").unique()):
                recalculate = True
                
        if recalculate:
            mapconn_curves_null = self.get_mapconn_curves_null(maps=maps, percentiles=percentiles, ids=ids)
            mapconn_stats_null = []
            for mapconn_curves_null_i in mapconn_curves_null:
                mapconn_stats_null.append(
                    _calc_mapconn_stats(mapconn_curves_null_i, stats=stats, force_dict=True)
                )
            mapconn_stats_null = {stat: [null[stat] for null in mapconn_stats_null] 
                                  for stat in mapconn_stats_null[0].keys()}
            self._mapconn_stats_null = mapconn_stats_null
            
        return mapconn_stats_null
    
    def get_p_values(self, stats="auc", maps=None, percentiles=None, ids=None, p_from_mean=True,
                     norm=False, tail="upper", recalculate=False, n_jobs=-1):
        if isinstance(stats, (str, int)):
            stats = [stats]
        mapconn_p_values = self._mapconn_p_values
        if mapconn_p_values is None or recalculate:
            recalculate = True
        if mapconn_p_values is not None:
            # check if all stats and data are available
            if not all(stat in mapconn_p_values.keys() for stat in stats):
                recalculate = True
            # check is norm
            if self._mapconn_p_norm != norm:
                recalculate = True
            # get observed mapconn stats for reference
            mapconn_stats = self._mapconn_instance.get_mapconn_stats(
                stats=stats, maps=maps, percentiles=percentiles, ids=ids
            )
            if p_from_mean:
                if not mapconn_p_values[stats[0]].index[0] == f"mean_{stats[0]}":
                    recalculate = True
            else:
                if not np.array_equal(mapconn_p_values[stats[0]].index, 
                                      mapconn_stats[stats[0]].index):
                    recalculate = True
            if not np.array_equal(mapconn_p_values[stats[0]].columns, 
                                  mapconn_stats[stats[0]].columns):
                recalculate = True
        
        if recalculate:
            mapconn_p_values = {}
            # get observed mapconn stats
            mapconn_stats = self._mapconn_instance.get_mapconn_stats(
                stats=stats, maps=maps, percentiles=percentiles, ids=ids
            )
            # get null mapconn stats
            mapconn_stats_null = self.get_mapconn_stats_null(
                stats=stats, maps=maps, percentiles=percentiles, ids=ids
            )
            # iterate over stats
            if stats == "all":
                stats = STATS
            if isinstance(stats, (str, int)):
                stats = [stats]
            for stat in set(stats).intersection(set(mapconn_stats.keys())):
                # out df
                p_values = pd.DataFrame(
                    columns=mapconn_stats[stat].columns,
                    index=mapconn_stats[stat].index if not p_from_mean else [f"mean_{stat}"]
                )
                # observed
                obs = np.array(mapconn_stats[stat])
                if p_from_mean:
                    obs = obs.mean(axis=0, keepdims=True)
                # null
                null = np.stack(mapconn_stats_null[stat], axis=0)
                if p_from_mean:
                    null = np.mean(null, axis=1, keepdims=True)
                # calculate p-values
                for i_m, m in enumerate(p_values.columns):
                    for i_idx, idx in enumerate(p_values.index):
                        p_values.loc[idx, m] = null_to_p(
                            test_value=obs[i_idx, i_m],
                            null_array=null[:, i_idx, i_m],
                            tail=tail,
                            fit_norm=norm
                        )
                # store
                mapconn_p_values[stat] = p_values
                
            self._mapconn_p_values = mapconn_p_values
            self._mapconn_p_norm = norm
            
        return mapconn_p_values
    
        
    @classmethod
    def from_instance(cls, mapconn_instance, map_data_null=None, 
                      parcellation=None, parcellation_space="mni152", dist_mat=None, 
                      n_nulls=1000, n_jobs=-1, seed=None, verbose=True, dtype=None):
        """
        Create a MapConnNull instance from an "observed" MapConn instance.
        """
        
        # checks
        if not isinstance(mapconn_instance, MapConn):
            raise ValueError("mapconn_instance must be a MapConn instance")
        if not hasattr(mapconn_instance, "_map_data"):
            raise ValueError("mapconn_instance must have original map data stored in ._map_data")
        if parcellation is None and dist_mat is None and map_data_null is None:
            raise ValueError("Either parcellation or distance matrix or map_data_null must be provided")
        
        # get map data
        map_data = mapconn_instance.get_map_data(pct=False)
        
        # dtype
        if dtype is None:
            dtype = mapconn_instance._dtype
        
        # get flat connectivity data
        flat_connectivity_matrices = mapconn_instance.get_connectivity_matrices(flat=True)
        r_to_z = mapconn_instance._r_to_z
        
        # get percentiles
        percentiles = mapconn_instance._percentiles
        
        # get null data
        if map_data_null is None:
            map_data_null, dist_mat = generate_null_maps(
                method="moran",
                data=map_data,
                parcellation=parcellation,
                parc_space=parcellation_space,
                dist_mat=dist_mat,
                n_nulls=n_nulls,
                seed=seed,
                n_proc=n_jobs,
                verbose=False,
                dtype=dtype
            )
            map_data_null = [
                np.stack([map_data_null[m][i,:] for m in map_data_null.keys()], dtype=dtype)
                for i in range(n_nulls)
            ]
        
        # get null mapconn curves
        mapconn_curves_null = Parallel(n_jobs=n_jobs)(
            delayed(calculate_mapconn)(
                flat_connectivity_matrices, 
                map_data=map_data_null_i, 
                map_data_is_pct=False,
                percentiles=percentiles, 
                return_mappct=False,
                return_df=False,
                r_to_z=r_to_z,
                n_jobs=1, 
                verbose=False,
                dtype=dtype
            )
            for map_data_null_i 
            in tqdm(map_data_null, disable=not verbose, desc="Calculating null mapconn curves")
        )
        
        # return
        return cls(mapconn_instance=mapconn_instance,
                   map_data_null=map_data_null,
                   mapconn_curves_null=mapconn_curves_null,  
                   n_nulls=n_nulls,
                   dtype=dtype)
        

# mapconn curves
def calculate_mapconn(flat_connectivity_matrices, map_data=None, map_data_is_pct=False, mappct_masks_flat=None, 
                      r_to_z=False, percentiles=np.arange(0, 100, 5), mappercentile_threshold="overequal",
                      return_mappct=False, return_df=True, n_jobs=-1, verbose=True, dtype=np.float32):
    
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
        
    # mean after applying quantile thresholds
    mapconn = Parallel(n_jobs=n_jobs)(
        delayed(_threshold_conn_data)(conn_data_flat, mappct_masks_flat_arr[map_idx, :]) 
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


def _threshold_conn_data(conn_data_flat, bool_vector):
    conn_data_thresh = conn_data_flat[:, bool_vector]
    conn_data_thresh = np.nanmean(conn_data_thresh, axis=1)
    return conn_data_thresh

