"""
Map data and null map inputs in different formats (single maps as 1D arrays/Series, numpy
arrays, transposed data, null arrays instead of dicts) must give the same results as the
canonical DataFrame (n_maps, n_parcels) + dict {map: (n_nulls, n_parcels)} input.
"""

import numpy as np
import pytest
import xarray as xr

from mapconn import MapConnInv, MapConnNull

KW = dict(r_to_z=True, n_jobs=1, verbose=False, dtype=np.float64)


def _fit(conn, map_data, map_data_null, **kwargs):
    mc = MapConnInv.from_matrix(connectivity_matrices=conn, map_data=map_data, **KW)
    return MapConnNull.from_mapconn(
        mc,
        map_data_null=map_data_null,
        n_jobs=1,
        verbose=False,
        dtype=np.float64,
        compute_pvalues=True,
        compute_delta=True,
        **kwargs,
    )


@pytest.fixture(scope="module")
def reference(toy_data):
    conn, map_data, nulls = toy_data
    m = map_data.index[0]
    one = _fit(conn, map_data.iloc[[0]], {m: nulls[m]})
    two = _fit(conn, map_data, nulls)
    return one, two


def _summaries(null_model):
    return (
        null_model.get_summary(level="group", verbose=False, reduce_index=False),
        null_model.get_summary(level="individual", verbose=False, reduce_index=False),
    )


def _assert_same_values(a, b):
    for sa, sb in zip(_summaries(a), _summaries(b)):
        assert sa.shape == sb.shape
        np.testing.assert_allclose(sa.values, sb.values, rtol=1e-10, equal_nan=True)


@pytest.mark.parametrize(
    "variant",
    [
        "series",
        "ndarray_1d",
        "list_1d",
        "ndarray_2d",
        "column_vector",
        "xarray_1d",
        "null_dict_key",
    ],
)
def test_single_map_formats(toy_data, reference, variant):
    conn, map_data, nulls = toy_data
    m = map_data.index[0]
    row, null = map_data.iloc[0], nulls[m]
    md, nl = {
        "series": (row, {m: null}),
        "ndarray_1d": (row.values, null),
        "list_1d": (list(row.values), null),
        "ndarray_2d": (row.values[np.newaxis, :], null),
        "column_vector": (row.values[:, np.newaxis], null.T),
        "xarray_1d": (xr.DataArray(row.values), null),
        "null_dict_key": (row.values, {"other_name": null}),
    }[variant]
    _assert_same_values(_fit(conn, md, nl), reference[0])


def test_multi_map_transposed_and_3d_nulls(toy_data, reference):
    conn, map_data, nulls = toy_data
    nulls_3d = np.stack([nulls[m] for m in map_data.index])
    _assert_same_values(_fit(conn, map_data.T, nulls_3d), reference[1])


def test_null_dict_order_follows_map_names(toy_data, reference):
    conn, map_data, nulls = toy_data
    reversed_nulls = {m: nulls[m] for m in reversed(map_data.index)}
    _assert_same_values(_fit(conn, map_data, reversed_nulls), reference[1])


def test_n_nulls_capped_by_provided_nulls(toy_data):
    conn, map_data, nulls = toy_data
    n_available = next(iter(nulls.values())).shape[0]
    assert _fit(conn, map_data, nulls)._n_nulls == n_available  # default n_nulls=1000
    assert _fit(conn, map_data, nulls, n_nulls=10)._n_nulls == 10


def test_invalid_inputs_raise(toy_data):
    conn, map_data, nulls = toy_data
    with pytest.raises(ValueError, match="1D .* or 2D"):
        MapConnInv.from_matrix(conn, map_data=np.zeros((2, 3, 4)), **KW)
    with pytest.raises(ValueError, match="missing null maps"):
        _fit(conn, map_data, {"mapA": nulls["mapA"]})


def test_tuple_matrix_ids(toy_data):
    """Tuple ids (e.g., (subject, session)) must not be split into multiple index levels."""
    conn, map_data, nulls = toy_data
    m = map_data.index[0]
    tuple_ids = [(f"sub-{i // 2}", f"ses-{i % 2}") for i in range(conn.shape[0])]
    str_ids = [f"{s}_{t}" for s, t in tuple_ids]

    def fit(ids):
        mc = MapConnInv.from_matrix(
            connectivity_matrices=list(conn), map_data=map_data.iloc[[0]], matrix_ids=ids, **KW
        )
        return MapConnNull.from_mapconn(
            mc, map_data_null={m: nulls[m]}, n_jobs=1, verbose=False, dtype=np.float64
        )

    null_tuple, null_str = fit(tuple_ids), fit(str_ids)
    np.testing.assert_allclose(
        null_tuple.get_pvalues(p_from_mean=False, inverted=True).values,
        null_str.get_pvalues(p_from_mean=False, inverted=True).values,
    )
    s_tuple = null_tuple.get_summary(level="individual", verbose=False, reduce_index=False)
    s_str = null_str.get_summary(level="individual", verbose=False, reduce_index=False)
    assert s_tuple.index.names == s_str.index.names
    assert set(s_tuple.index.get_level_values("id")) == set(tuple_ids)
    np.testing.assert_allclose(s_tuple.values, s_str.values, rtol=1e-10, equal_nan=True)
