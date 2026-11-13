import numpy as np
import pandas as pd
import pytest

N_PARCELS = 60
N_SUBJECTS = 4
N_NULLS = 40


@pytest.fixture(scope="session")
def toy_data():
    """
    Toy connectomes, two skewed maps, and circular-shift null maps on a ring of parcels.

    Returns (conn (n_subjects, n_parcels, n_parcels), map_data DataFrame (n_maps, n_parcels),
    nulls dict {map: array (n_nulls, n_parcels)}).
    """
    rng = np.random.default_rng(0)
    parcels = [f"p{i}" for i in range(N_PARCELS)]

    # parcels on a ring; FC decays with ring distance, plus a shared non-stationary component
    # and subject-specific noise
    pos = np.arange(N_PARCELS)
    d = np.abs(pos[:, None] - pos[None, :])
    d = np.minimum(d, N_PARCELS - d)
    shared = rng.normal(0, 0.3, (N_PARCELS, N_PARCELS))
    shared = (shared + shared.T) / 2
    conn = []
    for _ in range(N_SUBJECTS):
        noise = rng.normal(0, 0.1, (N_PARCELS, N_PARCELS))
        c = np.exp(-d / 4) * 0.8 + shared + (noise + noise.T) / 2
        np.fill_diagonal(c, 1)
        conn.append(np.clip(c, -0.99, 0.99))
    conn = np.stack(conn)

    # skewed maps: compact upper tail (one bump), diffuse lower tail
    def bump(center, width):
        dc = np.minimum(np.abs(pos - center), N_PARCELS - np.abs(pos - center))
        return np.exp(-(dc**2) / (2 * width**2))

    maps = {
        "mapA": np.exp(3 * bump(15, 3.5)) + rng.normal(0, 0.05, N_PARCELS),
        "mapB": np.exp(2 * bump(45, 6)) + rng.normal(0, 0.05, N_PARCELS),
    }
    map_data = pd.DataFrame(maps, index=parcels).T  # (n_maps, n_parcels)

    # nulls: circular shifts (ring analogue of spins)
    shifts = rng.choice(np.arange(1, N_PARCELS), N_NULLS, replace=False)
    nulls = {m: np.stack([np.roll(v, s) for s in shifts]) for m, v in maps.items()}

    return conn, map_data, nulls
