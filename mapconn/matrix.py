import logging
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import sklearn.covariance as skcov
from nilearn.connectome import sym_matrix_to_vec
from spatiotemporal import spatial_autocorrelation

logger = logging.getLogger(__name__)

def _get_matrix_estimator(method: str = "EmpiricalCovariance", kind: str = "covariance", normalize: bool = True, 
                          dtype: Union[np.dtype, type] = float, verbose: bool = False, **kwargs: Any) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get a function to calculate a covariance matrix from standardized time series data. 
    Input to the returned function is a timeseries array with shape (n_timepoints, n_parcels).
    """
    
    # intercept if standard pearson because much faster
    if method.lower() == "empiricalcovariance" and kind == "covariance" and normalize:
        def matrix_fun(timeseries: np.ndarray) -> np.ndarray:
            """Compute Pearson correlation matrix from time series."""
            correlation_matrix = np.corrcoef(timeseries.T)
            return correlation_matrix      
        
        # verbose
        if verbose:
            logger.info("Estimating pearson correlation matrix")
            
    # other options: use sklearn
    else:
        
        # estimator
        if method.lower() == "empiricalcovariance":
            estimator = skcov.EmpiricalCovariance
        elif method.lower() == "graphicallassocv":
            estimator = skcov.GraphicalLassoCV
        elif method.lower() == "shrunkcovariance":
            estimator = skcov.ShrunkCovariance
        else:
            raise ValueError(f"Method {method} not recognized")
        
        # matrix kind
        if "cov" in kind.lower():
            matrix_attr = "covariance_"
        elif "prec" in kind.lower():
            matrix_attr = "precision_"
        else:
            raise ValueError(f"Kind {kind} not recognized")
        
        # normalization
        if normalize:
            norm_fun = _covariance_to_pearson if matrix_attr == "covariance_" else _precision_to_partial_pearson
        else:
            def norm_fun(matrix: np.ndarray) -> np.ndarray:
                """Return unmodified matrix when normalization is disabled."""
                return matrix
            
        # define function
        def matrix_fun(timeseries: np.ndarray) -> np.ndarray:
            """Compute covariance/precision matrix and normalize if requested."""
            matrix = getattr(
                estimator(**kwargs).fit(np.array(timeseries)), 
                matrix_attr
            )
            matrix_norm = norm_fun(matrix)
            return matrix_norm.astype(dtype)
        
        # verbose
        if verbose:
            if method.lower() == "empiricalcovariance" and kind == "precision" and normalize:
                logger.info("Estimating partial pearson correlation matrix")
            else:
                logger.info("Estimating %s%s matrix using sklearn - %s",
                            "normalized " if normalize else "",
                            matrix_attr[:-1],
                            method)
    
    # return function
    return matrix_fun

def _covariance_to_pearson(covariance_matrix: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to Pearson correlation matrix."""
    std = np.sqrt(np.diag(covariance_matrix))
    cor_matrix = covariance_matrix / np.outer(std, std)
    np.fill_diagonal(cor_matrix, 1)
    return cor_matrix

def _precision_to_partial_pearson(precision_matrix: np.ndarray) -> np.ndarray:
    """Convert precision matrix to partial Pearson correlation matrix."""
    std = np.sqrt(np.diag(precision_matrix))
    pcor_matrix = -precision_matrix / np.outer(std, std)
    np.fill_diagonal(pcor_matrix, 1)
    return pcor_matrix

# get number of elements in a symmetric matrix triangle
def _n_sym_matrix_tri_elem_from_shape(shape0: int, discard_diagonal: bool = True) -> int:
    """Return number of elements in a symmetric matrix triangle."""
    if discard_diagonal:
        return shape0 * (shape0 - 1) // 2
    else:
        return shape0 * (shape0 + 1) // 2
    
# get original size of a matrix given the number of elements in its triangle
def _sym_matrix_shape_from_n_tri_elem(num_elements: int, discard_diagonal: bool = True) -> int:
    """Return matrix size from number of triangular elements."""
    if discard_diagonal:
        a = 1
        b = -1
        c = -2 * num_elements
    else:
        a = 1
        b = 1
        c = -2 * num_elements

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No real solutions. The number of elements does not correspond to a valid symmetric matrix.")

    # Calculate the positive root of the equation (since matrix size cannot be negative)
    return int((-b + np.sqrt(discriminant)) / (2 * a))

# vectorize a list of symmetric matrices
def _vectorize_sym_matrices(sym_matrices: Union[np.ndarray, Sequence[np.ndarray]], discard_diagonal: bool = True) -> np.ndarray:
    """Vectorize a list/array of symmetric matrices."""
    return np.array([sym_matrix_to_vec(m, discard_diagonal=discard_diagonal) for m in sym_matrices])

# spatial autocorrelation of a connectivity matrix
def matrix_sac(matrix: np.ndarray, distmat: np.ndarray, discretization: Optional[Union[str, float]] = None) -> np.ndarray:
    """Compute spatial autocorrelation of a connectivity matrix."""
    if discretization is None:
        discretization = "fd"
    if discretization == "fd":
        x = np.asarray(distmat)
        np.fill_diagonal(x, 1)
        x = x[~np.isnan(x)].flatten()
        q25, q75 = np.percentile(x, [25, 75])
        discretization = 2 * (q75 - q25) / np.cbrt(x.size)

    return spatial_autocorrelation(matrix, distmat, discretization=discretization)