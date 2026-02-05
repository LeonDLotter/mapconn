def test_imports():
    from mapconn import MapConn, MapConnInv, MapConnNull

    assert MapConn is not None
    assert MapConnInv is not None
    assert MapConnNull is not None
