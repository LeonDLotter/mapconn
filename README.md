# mapconn

[![Preprint](https://img.shields.io/badge/Preprint-bioRxiv-blue)](https://doi.org/...)
[![DOI: 10.5281/zenodo.19859026](https://zenodo.org/badge/DOI/10.5281/zenodo.19859026.svg)](https://doi.org/10.5281/zenodo.19859026)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](https://choosealicense.com/licenses/mit/)

**mapconn** is a Python toolbox for computing the *Neurobiologically Enriched Organization of Functional Connectivity* (NEOFC) — a framework that quantifies how functional connectivity varies as a function of regional neurobiological properties (e.g., neurotransmitter receptor and transporter density from PET). It takes parcellated connectomes and spatial reference maps as input and returns subject-level indices of map-dependent connectivity organization, along with tools for null testing and regional influence estimation.

For the full methodological description, validation, and applications, see the preprint and the associated analysis repository below.

## Cite

> Lotter LD, Shafiei G, Larabi D, Koushik A, Dipasquale O, Mehta M, Cercignani M, Sethi A, Harrison N, Holiga Š, Umbricht D, Yakushev I, Muthukumaraswamy S, Forsyth A, Hipp JF, Misic B, Caspers S, Koenig J, Patil KR, Paquola C, Eickhoff SB & Dukart J (2026). *Linking human brain functional connectivity to underlying neurotransmission*. bioRxiv.

> Lotter LD (2026). *mapconn* (v0.0.1-dev). Zenodo. https://doi.org/10.5281/zenodo.19859026

## Install

```bash
pip install git+https://github.com/leondlotter/mapconn
```

Requires Python ≥ 3.10.

## Quick start

```python
import numpy as np
import pandas as pd
from mapconn import MapConn, MapConnInv, MapConnNull

# Inputs:
#   conn        — np.ndarray (n_subjects, n_parcels, n_parcels), parcellated FC matrices
#   subject_ids — list of subject identifiers, length n_subjects
#   map_data    — pd.DataFrame (n_maps, n_parcels), reference maps (e.g. PET atlases)
#   dist_mat    — np.ndarray (n_parcels, n_parcels), parcel centroid distance matrix

# 1. Compute map-connectivity curves for the original reference maps
mc = MapConn.from_matrix(
    connectivity_matrices=conn,
    matrix_ids=subject_ids,
    map_data=map_data,
    r_to_z=False,
    n_jobs=-1,
    dtype=np.float32,
)

# 2. Compute curves for spatially inverted maps
mc_inv = MapConnInv.from_mapconn(mapconn_instance=mc)

# 3. Generate autocorrelation-preserving spatial null maps and compute null distributions
mc_null = MapConnNull.from_mapconn(
    mapconn_instance=mc_inv,
    distmat=dist_mat,
    n_nulls=1000,
    n_jobs=-1,
    seed=42,
)

# 4. Get summary table: observed AUC stats, null distribution, and p-values
summary = mc_null.get_summary()
```

Working examples in Jupyter Notebooks will be added to this repository.  
Full analysis code from the paper is available in the [associated repository](https://github.com/leondlotter/neofc).

## NiSpace

mapconn is a standalone implementation developed alongside the manuscript. Its functionality will be integrated into [NiSpace](https://github.com/leondlotter/nispace) — a broader neuroimaging spatial colocalization toolbox — in the future. For new projects, we recommend checking NiSpace once the integration is complete.

## Contact

Leon D. Lotter — leondlotter@gmail.com  
Bug reports and questions via [GitHub Issues](https://github.com/leondlotter/mapconn/issues).
