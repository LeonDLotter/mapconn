STATS = [
    "auc",
    "auc2",
    "poly2",
    "peak_conn",
    "peak_pct",
    *[f"pct=={i}" for i in range(0, 100, 5)],
    *[f"auc<={i}" for i in range(5, 100, 5)],
    *[f"auc2<={i}" for i in range(5, 100, 5)],
    *[f"poly2<={i}" for i in range(5, 100, 5)],
]
