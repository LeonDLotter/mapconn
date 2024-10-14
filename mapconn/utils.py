import numpy as np
import pandas as pd
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix

from .matrix import _sym_matrix_shape_from_n_tri_elem

def over(array1, array2):
    return array1 > array2

def overequal(array1, array2):
    return array1 >= array2

def below(array1, array2):
    return array1 < array2

def belowequal(array1, array2):
    return array1 <= array2

# convert flat mappercentile data to parcel-format data
def _mappct_flat_to_parcels(mappct_data_flat, parcel_labels=None):
    n_parcels = _sym_matrix_shape_from_n_tri_elem(mappct_data_flat.shape[1])
    
    diagonal = np.full(n_parcels, False)
    mappct_data = np.full((mappct_data_flat.shape[0], n_parcels), False)
    for i_v in range(mappct_data.shape[0]):
        mat = vec_to_sym_matrix(mappct_data_flat.values[i_v, :], diagonal=diagonal)
        mappct_data[i_v, :] = mat.sum(axis=0) > 0
        
    return pd.DataFrame(
        mappct_data,
        columns=parcel_labels,
        index=mappct_data_flat.index,
        dtype=bool
    )
    
# convert parcel-format mappercentile data to flat data
def _mappct_parcels_to_flat(mappct_data, parcel_pair_labels=None):
    return pd.DataFrame(
        np.stack([sym_matrix_to_vec(np.outer(mappct_data.values[i,:], mappct_data.values[i,:]), discard_diagonal=True) 
                  for i in range(mappct_data.shape[0])]),
        index=mappct_data.index,
        columns=parcel_pair_labels
    )

def _construct_flat_label_pairs(labels, discard_diagonal=True):
    idc = np.tril_indices(len(labels), -1 if discard_diagonal else 0)
    return [(labels[i], labels[j]) for i, j in zip(*idc)]

