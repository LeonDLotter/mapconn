"""
Regression tests: inverted (AUC-) statistics must be tested against inverted null maps.

Inversion commutes with the null-generating transformation (spins / here: circular shifts),
so p-values for the inverted map must equal those obtained by treating the negated map, with
the negated null maps, as a new reference map.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from mapconn import MapConn, MapConnInv, MapConnNull

N_PARCELS = 60
N_SUBJECTS = 4
N_NULLS = 40
STAT = "auc"


@pytest.fixture(scope="module")
def toy_data():
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


def _kwargs():
    return dict(r_to_z=True, n_jobs=1, verbose=False, dtype=np.float64)


@pytest.fixture(scope="module")
def models(toy_data):
    conn, map_data, nulls = toy_data

    # MapConnInv model with nulls for the original maps
    mc_inv = MapConnInv.from_matrix(connectivity_matrices=conn, map_data=map_data, **_kwargs())
    null_inv = MapConnNull.from_mapconn(
        mc_inv,
        map_data_null=nulls,
        n_nulls=N_NULLS,
        n_jobs=1,
        verbose=False,
        dtype=np.float64,
        compute_pvalues=True,
        compute_delta=True,
    )

    # reference: negated maps with negated nulls, treated as new reference maps
    mc_neg = MapConn.from_matrix(connectivity_matrices=conn, map_data=-map_data, **_kwargs())
    null_neg = MapConnNull.from_mapconn(
        mc_neg,
        map_data_null={m: -v for m, v in nulls.items()},
        n_nulls=N_NULLS,
        n_jobs=1,
        verbose=False,
        dtype=np.float64,
        compute_pvalues=True,
    )
    return null_inv, null_neg


def test_inverted_observed_stats_match_negated(models):
    null_inv, null_neg = models
    np.testing.assert_allclose(
        null_inv.get_inverted_stats(stats=STAT).values,
        null_neg.get_stats(stats=STAT).values,
        rtol=1e-6,
        atol=1e-10,
    )


def test_inverted_null_stats_match_negated(models):
    null_inv, null_neg = models
    inv = np.stack(null_inv.get_null_stats(stats=STAT, inverted=True))
    neg = np.stack(null_neg.get_null_stats(stats=STAT))
    np.testing.assert_allclose(inv, neg, rtol=1e-6, atol=1e-10)
    # sanity: in this setup, inverted nulls differ from original nulls (the bug was visible)
    orig = np.stack(null_inv.get_null_stats(stats=STAT))
    assert not np.allclose(orig.mean(axis=(0, 1)), inv.mean(axis=(0, 1)))


@pytest.mark.parametrize("p_from_mean", [True, False])
@pytest.mark.parametrize("tail", ["upper", "two"])
def test_inverted_pvalues_match_negated(models, p_from_mean, tail):
    null_inv, null_neg = models
    p_inv = null_inv.get_pvalues(stats=STAT, inverted=True, p_from_mean=p_from_mean, tail=tail)
    p_neg = null_neg.get_pvalues(stats=STAT, p_from_mean=p_from_mean, tail=tail)
    np.testing.assert_allclose(p_inv.values, p_neg.values)


def test_inverted_summary_z_matches_negated(models):
    null_inv, null_neg = models
    s_inv = null_inv.get_summary(level="group", reduce_index=False, verbose=False)
    s_neg = null_neg.get_summary(level="group", reduce_index=False, verbose=False)
    for var in ["mean", "null_mean", "null_std", "mean_z", "mean_rz", "p"]:
        np.testing.assert_allclose(
            s_inv.loc[(STAT, "inverted", var)].values.ravel(),
            s_neg.loc[(STAT, "original", var)].values.ravel(),
            rtol=1e-6,
            err_msg=var,
        )


def test_inverted_results_survive_drop_nulls(models, tmp_path):
    null_inv = copy.deepcopy(models[0])  # save() drops nulls in place
    s_before = null_inv.get_summary(level="group", reduce_index=False, verbose=False)
    path = tmp_path / "null.pkl"
    null_inv.save(path)
    loaded = MapConnNull.load(path)
    s_after = loaded.get_summary(level="group", reduce_index=False, verbose=False)
    pd.testing.assert_frame_equal(s_before, s_after)


def test_delta_pvalues_use_shared_inverted_nulls(models, toy_data):
    null_inv, null_neg = models
    _, map_data, _ = toy_data
    p_delta = null_inv.get_delta_pvalues(stats=STAT, verbose=False, recalculate=True)
    assert p_delta.shape == (1, map_data.shape[0])
    assert np.all((p_delta.values > 0) & (p_delta.values <= 1))
    # delta nulls = original nulls - inverted nulls, where inverted nulls equal negated-map nulls
    null_delta = np.stack(null_inv.get_delta_null_stats(stats=STAT))
    expected = np.stack(null_inv.get_null_stats(stats=STAT)) - np.stack(
        null_neg.get_null_stats(stats=STAT)
    )
    np.testing.assert_allclose(null_delta, expected, rtol=1e-6, atol=1e-10)
