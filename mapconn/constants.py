from typing import Literal

MapPctThreshold = Literal["over", "overequal", "below", "belowequal"]

STATS = [
    "auc",
    "auc2",
    "auc_abs",
    "auc_pos",
    "auc_neg",
    "poly2",
    "peak_conn",
    "peak_pct",
    *[f"pct=={i}" for i in range(0, 100, 5)],
    *[f"auc<={i}" for i in range(5, 100, 5)],
    *[f"auc2<={i}" for i in range(5, 100, 5)],
    *[f"auc_abs<={i}" for i in range(5, 100, 5)],
    *[f"auc_pos<={i}" for i in range(5, 100, 5)],
    *[f"auc_neg<={i}" for i in range(5, 100, 5)],
    *[f"poly2<={i}" for i in range(5, 100, 5)],
]
