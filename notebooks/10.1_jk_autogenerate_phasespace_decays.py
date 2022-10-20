# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %matplotlib notebook

from pathlib import Path
from baumbauen import datasets
# -

# ## Generate a set of sample decays

# Set location to output dataset
root_dir = Path('/path/to/save/data')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

datasets.generate_phasespace(
    root=root_dir,
    masses=[100, 90, 80, 70, 50, 20, 25, 10],
    fsp_masses=[1, 2, 3, 5, 12],
    topologies=10,
    max_depth=8,
    max_children=6,
    min_children=2,
    #p_skip=0.3,
    train_events_per_top=1000,
    val_events_per_top=1000,
    test_events_per_top=1000,
    #seed=42,
)

# ## Load some of the generated data
#
# For inspection

import matplotlib.pyplot as plt

dset = datasets.PhasespaceSet(root=root_dir)

for i in range(len(dset.x)):
    print(dset.x[i].shape, dset.y[i].shape)

# Plot the energy spectra for one topology

# +
plt.figure()

plt.hist(
    dset.x[0][:, :, -1].reshape(-1),
    bins=100,
)

plt.show()
