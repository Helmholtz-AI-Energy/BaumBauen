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

# # Explore datasets for first experiment
#
# This just checks the topology coverage of the given datasets

# +
# %matplotlib notebook
# %config Completer.use_jedi = False
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from pathlib import Path

import numpy as np
# -

inputs = [
    ("Training", 'train', Path('/path/to/known/')),
    ("Validation unknown", 'val', Path('/path/to/unknown/')),
    ("Testing unknown", 'test', Path('/path/to/unknown')),
]

depth_leaves = {}
for (label, tag, path) in inputs: 
    for file in path.glob(f'**/lcas_{tag}*'):
        arr = np.load(file, mmap_mode='r')
        depth_leaves.setdefault(label, []).append((int(arr[0].max()) + 1, arr.shape[-1]))

# ## Plot the distribution of shapes

# First establish the max ranges for all
# [letter for word in sentence for letter in word]

max_depth = max([y[0] for x in depth_leaves.values() for y in x])
max_leaves = max([y[1] for x in depth_leaves.values() for y in x])
max_depth, max_leaves

illegal = np.ones((len(depths_bins) - 1, len(leaves_bins) - 1))
illegal = np.flip(np.tril(illegal, k=-1), 0)
illegal[illegal == 0] = np.nan
illegal[-1, 4:] = 1


def single_figure(ax, label, depth_leaves_list):
    
    depths = [x[0] for x in depth_leaves_list]
    leaves = [x[1] for x in depth_leaves_list]
    depths_bins = list(range(2, max_depth + 1))
    leaves_bins = list(range(2, max_leaves + 1))
    
    illegal = np.ones((len(depths_bins) - 1, len(leaves_bins) - 1))
    illegal = np.flip(np.tril(illegal, k=-1), 0)
    illegal[illegal == 0] = np.nan
    illegal[-1, 4:] = 1
    
    ax.imshow(
        illegal,
        extent=(min(leaves_bins), max(leaves_bins), min(depths_bins),max(depths_bins)),
        cmap='Reds_r',
    )

    (hist, xedges, yedges, im) = ax.hist2d(
        leaves,
        depths,
        bins=[
            list(range(min(leaves), max_leaves + 1)),
            list(range(min(depths), max_depth + 1))
        ],
        cmin=0.5,
        cmap='cividis',
        range=((2, max_leaves), (2, max_depth)),
    )
    print(label)
    print(hist.T[::-1])

    ax.set_xticks(xedges[:-1] + 0.5)
    ax.set_xticklabels(xedges[:-1])
    
    # Hide every second label for clarity
    for xlabel in ax.xaxis.get_ticklabels()[1::2]:
        xlabel.set_visible(False)
    
    ax.set_yticks(yedges[:-1] + 0.5)
    ax.set_yticklabels(yedges[:-1])

    ax.set_ylabel('Depth')
    
    text_box = AnchoredText(
        label, frameon=True, loc='lower right', pad=0.4,
        borderpad=0.1,
    )
    plt.setp(text_box.patch, facecolor='white', alpha=0.8)
    ax.add_artist(text_box)
    
    return im, hist


save_path = Path('/path/to/save/figures')
save_path = False

# +
fig, axes = plt.subplots(nrows=len(depth_leaves), ncols=1, figsize=(10,5), sharex='col')

for idx, label in enumerate(depth_leaves):
    ax = axes.flat[idx]
    
    im, hist = single_figure(ax, label, depth_leaves[label])

# Set the x-axis label on bottom plot only
ax.set_xlabel('Leaves')

# Make room for the colourbar
cbar_ax = fig.add_axes([0.69, 0.15, 0.015, 0.7])

cbar = fig.colorbar(
    im, ax=axes, cax=cbar_ax,
    ticks=list(np.arange(int(np.nanmin(hist)), int(np.nanmax(hist)), 2) + 0.5),
)
cbar.set_ticklabels(list(np.arange(int(np.nanmin(hist)), int(np.nanmax(hist)), 2)))
cbar.set_label('No. topologies')

if save_path:
    plt.savefig(
        save_path / 'simulated_dataset_600x5000.pdf',
        transparent=True,
        bbox_inches='tight',
    )

plt.show()
