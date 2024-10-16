import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from sklearn.manifold import MDS
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def coords_from_distmat(distmat, n_dim=3):
    """
    Constructs coordinates from a distance matrix using Multidimensional Scaling (MDS).

    Parameters:
    - distance_matrix: A square matrix where element (i, j) represents the distance between points i and j.
    - n_components: The number of dimensions for the output coordinates (e.g., 2 for 2D, 3 for 3D).

    Returns:
    - A numpy array of shape (n_samples, n_components) containing the constructed coordinates.
    """
    mds = MDS(n_components=n_dim, dissimilarity="precomputed", random_state=42)
    coordinates = mds.fit_transform(distmat)
    return coordinates

def largest_distance(coords, reference_coord):
    """
    Finds the largest Euclidean distance between a list of 3D coordinates and a single reference coordinate.

    Parameters:
    - coords: A list or array of 3D coordinates, where each coordinate is a tuple or list (x, y, z).
    - reference_coord: A single 3D coordinate (x, y, z) to compare against.

    Returns:
    - The largest distance found.
    """
    coords = np.array(coords)
    reference_coord = np.array(reference_coord)
    
    # Calculate the Euclidean distance from each point to the reference point
    distances = np.linalg.norm(coords - reference_coord, axis=1)
    
    # Find the maximum distance
    max_distance = np.max(distances)
    return max_distance


def create_sphere_surface(radius, num_points=100, center=(0, 0, 0)):
    """
    Creates a surface of a sphere around a given center and returns the coordinates of the sphere surface.

    Parameters:
    - center: A tuple or list of (x, y, z) coordinates for the center of the sphere.
    - radius: The radius of the sphere.
    - num_points: The number of divisions along each spherical coordinate. 

    Returns:
    - A numpy array of shape (num_points^2, 3) containing the 3D coordinates of the sphere surface points.
    """
    # Create a grid of angles
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]

    return np.vstack((x.ravel(), y.ravel(), z.ravel())).T


def assign_parcels_to_sphere(data_obs, sphere_points):
    """
    Assigns parcels to points on a sphere.

    Parameters:
    - data_obs: A numpy array of shape (n, 3) containing the 3D coordinates of the parcels.
    - sphere_points: A numpy array of shape (m, 3) containing the 3D coordinates of the sphere points.

    Returns:
    - A numpy array of shape (n,) containing the indices of the closest points on the sphere for each parcel.
    """
    tree = KDTree(sphere_points)
    _, indices = tree.query(data_obs)
    return indices


def rotate_sphere(sphere_points, angle, axis):
    """
    Rotates a set of 3D points around a fixed axis.

    Parameters:
    - sphere_points: A numpy array of shape (n, 3) containing the 3D coordinates of the points to rotate.
    - angle: The angle of rotation in radians.
    - axis: A tuple or list of (x, y, z) coordinates for the axis of rotation.

    Returns:
    - A numpy array of the rotated 3D coordinates.
    """
    rotation = R.from_rotvec(angle * np.array(axis))
    return rotation.apply(sphere_points)


def rotate_sphere_random(sphere_coords, center, random_state=None):
    """
    Rotates a set of 3D points randomly around a fixed center.

    Parameters:
    - sphere_points: A numpy array of shape (n, 3) containing the 3D coordinates of the points to rotate.
    - center: A tuple or list of (x, y, z) coordinates for the center of the sphere.
    - random_state: An integer or numpy random state for reproducibility.
    
    Returns:
    - A numpy array of the rotated 3D coordinates.
    """
    # Translate points to origin
    translated_points = sphere_coords - center
    
    # Generate a random rotation
    random_rotation = R.random(random_state=random_state)
    
    # Apply the random rotation to the translated points
    rotated_translated_points = random_rotation.apply(translated_points)
    
    # Translate points back to the original center
    rotated_points = rotated_translated_points + center
    
    return rotated_points


def match_coord_sets(coords1, coords2):
    """
    Finds the best one-to-one matching between two lists of coordinates, minimizing the total distance.

    Parameters:
    - coords1: A list or array of 3D coordinates, where each coordinate is a tuple or list (x, y, z).
    - coords2: A list or array of 3D coordinates, where each coordinate is a tuple or list (x, y, z).

    Returns:
    - A list of tuples, where each tuple contains the indices of the matched coordinates from coords1 and coords2.
    """
    # Ensure the input lists are numpy arrays
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)

    # Calculate the distance matrix
    distance_matrix = np.linalg.norm(coords1[:, np.newaxis] - coords2, axis=2)

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Return the list of matched indices
    return np.stack([row_ind, col_ind], axis=1)


class VolSpin:

    def __init__(self, data, coords=None, distmat=None, verbose=True):
        
        # check data
        if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series, list)):
            raise ValueError("Data must be a numpy array, pandas DataFrame, pandas Series, or list")
        data = np.array(data).squeeze()
        if data.ndim != 1:
            raise ValueError("Data must be of shape (n,) or (n, 1)")
        
        # check coords and distmat
        if coords is not None and distmat is not None:
            raise ValueError("Coordinates and distance matrix provided, ignoring distance matrix")
            distmat = None
        if coords is None and distmat is None:
            raise ValueError("Either coordinates or distance matrix must be provided")
        
        # check coords
        if coords is not None:
            if not isinstance(coords, (np.ndarray, pd.DataFrame, list)):
                raise ValueError("Coordinates must be a numpy array, pandas DataFrame, or list")
            coords = np.array(coords)
            if coords.shape[0] < 1 or coords.shape[1] != 3:
                raise ValueError("Coordinates must be of shape (n, 3)")
            if len(data) != coords.shape[0]:
                raise ValueError("Data and coordinates must be the same length / dimension 0")
        
        # check distmat
        if distmat is not None:
            if not isinstance(distmat, (np.ndarray, pd.DataFrame)):
                raise ValueError("Distance matrix must be a numpy array or pandas DataFrame")
            distmat = np.array(distmat)
            if distmat.shape[0] != distmat.shape[1] != len(data):
                raise ValueError("Distance matrix must be square and match the length of the data")
        
        # set
        self._data = data
        self._coords = coords
        self._distmat = distmat
        self._verbose = verbose
    def fit(self, center=None, radius=None, num_points=100):
        
        # get coords if not provided
        if self._coords is None:
            self._coords = coords_from_distmat(self._distmat)

        # get data center
        if center is None:
            self._center = self._coords.mean(axis=0)

        # get radius
        if radius is None:
            self._radius = largest_distance(self._coords, self._center) * 1.5
        
        # create the sphere around the data center
        self._sphere_coords = create_sphere_surface(self._radius, num_points, self._center)

        # for each data point, find the closest point on the sphere
        self._data_idc_in_sphere = assign_parcels_to_sphere(self._coords, self._sphere_coords)

        # keep only sphere points with an associated parcel
        self._sphere_coords_data = self._sphere_coords[self._data_idc_in_sphere]
        
    def transform(self, n_perm=1000, seed=None, n_jobs=1):
        
        # rotation fun
        def perm_fun(data=self._data, data_coords=self._coords, 
                     sphere_coords_data=self._sphere_coords_data, center=self._center, seed=seed):
            # set random state
            rng = np.random.RandomState(seed)
            
            # rotate the sphere
            sphere_coords_data_rot = rotate_sphere_random(sphere_coords_data, center=center, random_state=rng)

            # match data coordinates to rotated sphere coordinates, matching is one-to-one!
            # first column is index in data_coords, second column is index in sphere_coords_data_rot
            coord_match = match_coord_sets(data_coords, sphere_coords_data_rot)

            # rotated data
            return data[coord_match[:,1]], sphere_coords_data_rot
            
        # run
        data_perm = Parallel(n_jobs=n_jobs)(
            delayed(perm_fun)(seed=(seed+i**2 if seed is not None else None)) 
            for i in tqdm(range(n_perm), disable=not self._verbose)
        )
        self._data_perm = np.stack([d[0] for d in data_perm], axis=0)
        self._sphere_coords_data_perm = np.stack([d[1] for d in data_perm], axis=0)
        
        # return
        return self._data_perm
    
    def fit_transform(self, center=None, radius=None, num_points=100, 
                       n_perm=10, seed=None, n_jobs=1):
        self.fit(center, radius, num_points)
        return self.transform(n_perm, seed, n_jobs)


def plot_3d_coordinates(coords, title="", colors=None, cmap="viridis", elev=20, azim=30, fig=None, ax=None):
    """
    Plots a set of 3D coordinates in 3D space using Matplotlib.

    Parameters:
    - coords: A list or array of 3D coordinates, where each coordinate is a tuple or list (x, y, z).
    - title: The title of the plot (optional).
    """
    
    if ax is None:
        # Create a new figure for the 3D plot
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    # Unpack the coordinates
    x, y, z = zip(*coords)
    # Plot the points
    ax.scatter(x, y, z, c=colors, marker='o', alpha=0.7, cmap=cmap)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set the camera position
    ax.view_init(elev=elev, azim=azim)

    # return
    return fig, ax
