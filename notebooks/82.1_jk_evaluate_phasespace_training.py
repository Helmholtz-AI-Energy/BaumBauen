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

# # Evaluate trained model performance
#
# This notebook loads the trained models and their corresponding configurations, and evaluates their performance on a test dataset.

# +
# %matplotlib inline
# %config Completer.use_jedi = False
import torch
import ignite as ig
import numpy as np
from pathlib import Path
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on device:\t{device}')
# -

import baumbauen as bb
import yaml

# ## Set the models you want to evaluate

top_dir = Path('/path/to/models/')
model_dict = {
    'NRI': {
        # Path to directory containing saved model
        'path': top_dir / 'my_NRI_model',
        'timestamp': '2000.01.01_00.01',
    },
    'Transformer': {
        # Path to directory containing saved model
        'path': top_dir / 'my_Transformer_model',
        'timestamp': '2000.01.01_00.01',
    },
}
# Where to save figures and results stats
save_path = Path('/path/to/save/figures/')


# ## Method to build a saved model

def build_saved_model(model_path, timestamp, device):
    
    # Glob produces a list, regardless of how many files are found
    saved_model = list(model_path.glob(f'{timestamp}*.pt'))
    assert len(saved_model) > 0, f'No saved models for timestamp {timestamp} found in {model_path}'
    saved_model = saved_model[0]
    
    saved_configs = list(model_path.glob(f'{timestamp}*.yaml'))
    assert len(saved_configs) > 0, f'No saved configs for timestamp {timestamp} found in {model_path}'
    saved_configs = saved_configs[0]


    # Load the configs
    with open(saved_configs) as file:
            configs = yaml.safe_load(file)

    print(
        f'Loaded saved model:\t {saved_model.name}\n'
        f'Loaded saved configs:\t {saved_configs.name}'
         )
    
    # Grab some useful info
    selected_model = configs['train']['model']
    infeatures = configs['dataset']['num_features']
    num_classes = configs['dataset']['num_classes']
    print(
        f'Selected model:\t{selected_model}\n'
        f'Input features:\t{infeatures}\n'
        f'Number of classes:\t{num_classes}'
    )

    # Construct the model
    if selected_model == 'transformer_model':

        model = bb.models.BBTransformerBaseline(
            infeatures=infeatures,
            num_classes=num_classes,
            **configs[selected_model],
        )
    elif selected_model == 'nri_model':

        model = bb.models.NRIModel(
            infeatures=infeatures,
            num_classes=num_classes,
            **configs[selected_model],
        )

    # Set the loss
    if configs[selected_model]['loss'] == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    elif configs[selected_model]['loss'] == 'focal':
        loss_fn = bb.losses.FocalLoss(gamma=2.5, ignore_index=-1)

    # Load in the trained weight
    model.load_state_dict(torch.load(saved_model, map_location=device)['model'])
    model.eval()
    model.to(device)
    
    return model, configs, loss_fn


# ## Create the dataloader
#
# Need to load single topology at a time to make our depth-vs-leaves plot later

def load_single_topology(configs, file_id, mode='test', samples=None):
    
    source = configs['dataset']['source']
    
    if samples is not None:
        configs['dataset'][source]['config']['samples'] = samples
    
    print(f'Loading dataset for {source}')
    dataset = bb.data.PhasespaceSet(
        root=Path(configs['dataset'][source]['known_path']),
        mode=mode,
        file_ids=file_id,
        **configs['dataset'][source]['config']
    )
    # Note we recycle some of the val configs here, feel free to overwrite
    print(f'Creating dataloader for {source}')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs['val']['batch_size'],
        drop_last=False,
        shuffle=False,
        collate_fn=bb.utils.pad_collate_fn,
        num_workers=configs['val']['num_workers'],
    )
    return dataloader, dataset


# ## Create loop for Ignite evaluator

def evaluate_saved_model(model, loss_fn, device, configs, mode='test', samples=None, debug=None):
    ''' For each file in the (un)known path, loads corresponding saved model and evaluates performance
    
    Note that this method only loads one model per dataset (and the first it finds).
    You need to be modify this if you did many training runs for any dataset and the saved models are in the same directory.
    This method also only deals with known decays, requires modification to handle unknown.
    
    Args:
        model: PyTorch model whose weights will be set by loaded models
        loss_fn: Loss function (needed to calculate metrics)
        configs: Configs dict
        mode: train/val/test flag for which dataset files to load
        samples: Number of samples per topology to load
    '''
    source = configs['dataset']['source']
    
    # Create blank results dict
    dataset_results = {}

    # Now load each dataset and evaluate
    counter = 0
    for file in tqdm(list(Path(configs['dataset'][source]['known_path']).glob(f'lcas_{mode}*'))):
    
        # I added this just to test a few models
        counter += 1
        if debug is not None and counter > debug:
            break
                
        file_id = file.suffixes[0][1:]

        # Load the dataset
        dataloader, dataset = load_single_topology(configs, file_id, mode=mode, samples=samples)
        # Record the leaves and depth
        # We add +1 to depth since it's zero indexed
        # Using getitem to make sure we only get one sample
        leaves = dataset.__getitem__(0)[1].size()[-1]
        depth = (dataset.__getitem__(0)[1].max()).item() + 1

        # Create ignite evaluator
        metrics = {
            "loss": ig.metrics.Loss(loss_fn),
            "pad_accuracy": bb.metrics.Pad_Accuracy(ignored_class=[-1, 0], device=device),
            "perfect": bb.metrics.PerfectLCA(ignore_index=[-1], device=device),
        }
        evaluator = ig.engine.create_supervised_evaluator(
            model,
            metrics=metrics,
            device=device
        )
        # An run over the test dataset
        evaluator.run(dataloader)
        metrics = evaluator.state.metrics
        
        # Save metrics
        dataset_results[file_id] = {
            'depth': depth,
            'leaves': leaves,
            'metrics': metrics,
        }

    return dataset_results, metrics


# ## Create the average scores depths vs. leaves array for a given model

def create_results_array(results_dict, metric):
    ''' Given the training results and a metric, plot the averages for every depth/leaves
    
    Args:
        results_dict: Dictionary of results output by `evaluate_saved_model`
        metric: Which metric from Ignite evaluator state dict to plot
    
    Returns:
        Results for the given metric for each depth(axis=0) vs leaves(aixs=1), averaged across topologies if there are multiple at that depth/leaves
        Count of number of topologies at each depth/leaves
    '''
    # Find how big our results array needs to be
    # Recall that depth starts at 1 for a single-leveled tree (containing leaves only)
    max_depth = max([x['depth'] for x in results_dict.values()])
    max_leaves = max([x['leaves'] for x in results_dict.values()])
    
    # The +1 here is to make the array the right size, e.g. for one total depth then max_depth=0
    results = np.empty((max_depth, max_leaves))
    results[:] = np.nan
    # Keep track of the numbers to take the average
    count = np.empty(results.shape)
    count[:] = np.nan

    for val in results_dict.values():
        depth_idx = val['depth'] - 1
        leaves_idx = val['leaves'] - 1
        results[depth_idx, leaves_idx] = val['metrics'][metric] if np.isnan(results[depth_idx, leaves_idx]) else (results[depth_idx, leaves_idx] + val['metrics'][metric])
        count[depth_idx, leaves_idx] = 1 if np.isnan(count[depth_idx, leaves_idx]) else (count[depth_idx, leaves_idx] + 1)
    
    return results / count, count


# ## Actually execute all of the above
#
# This is where we actually create the test results

# Select the mode to perform inference on
mode = 'test'

# ### As an initial run, calculate the total scores

# +
avg_test_results = {}

for arch, arch_locs in model_dict.items():
    
    print(f'Loading model {arch}')
    model, configs, loss_fn = build_saved_model(arch_locs['path'], arch_locs['timestamp'], device)
    
    # Load this dataset
    dataloader, dataset = load_single_topology(configs, file_id=None, mode=mode, samples=2000*200)
    
    # Create ignite evaluator
    metrics = {
        "loss": ig.metrics.Loss(loss_fn),
        "pad_accuracy": bb.metrics.Pad_Accuracy(ignored_class=[-1, 0], device=device),
        "perfect": bb.metrics.PerfectLCA(ignore_index=[-1], device=device),
    }
    evaluator = ig.engine.create_supervised_evaluator(
        model,
        metrics=metrics,
        device=device
    )
    # An run over the test dataset
    evaluator.run(dataloader)
    metrics = evaluator.state.metrics
    
    print(metrics)
    avg_test_results[arch] = metrics
# -

# ### Then calculate on individual topologies

# +
test_results = {}

for arch, arch_locs in model_dict.items():
    
    print(f'Loading model {arch}')
    model, configs, loss_fn = build_saved_model(arch_locs['path'], arch_locs['timestamp'], device)

    test_results[arch] = {}
    
    print(f'Evaluating model {arch}')
    model_results, metrics = evaluate_saved_model(
        model,
        loss_fn,
        device,
        configs,
        mode=mode,
        samples=2000,
#         debug=2,
    )

    # Save the model results
    save_path.mkdir(exist_ok=True)
    with open(save_path / f'{configs["output"]["run_name"]}_{arch_locs["timestamp"]}_results.yaml', 'w') as outfile:
        yaml.dump(model_results, outfile, default_flow_style=False)
    
    # Go through each metric and make the test results array
    for metric in metrics:
        
        # Count here is number of topologies at each depth/leaves
        results_arr, count_arr = create_results_array(model_results, metric)

        # This saves the number of topologies at each depth/leaves for the dataset being evaluated
        # (only needed once for all architectures)
        if 'dataset' not in test_results:
            test_results['dataset'] = count_arr
    
        test_results[arch][metric] = results_arr
# -
test_results['NRI']['loss'].shape, test_results['NRI']['perfect'].shape, test_results['dataset'].shape

test_results

model_results

# ## Plot the results

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np


def single_figure(ax, label, results, min_depth=2, min_leaves=2, vmin=None, vmax=None):
    ''' Plot a single heatplot on the given axis
    
    Args:
        ax (plt.Axes): Axis to plot on (TODO: check type here)
        label (string): Label (e.g. architecture) to add to the plot
        results (np..ndarray): Numpy array to plot with (depth, leaves) as dimensions
        min_depth (int, optional): Where in the results array to start plotting up from for depth
        min_leaves (int, optional): Where in the results array to start plotting up from for leaves
    '''
    
    # Infer these from the results array
    max_depth = results.shape[0]
    max_leaves = results.shape[1]

    # Generate list of bin labels
    depths_bins = list(range(min_depth, (max_depth + 1)))
    leaves_bins = list(range(min_leaves, (max_leaves + 1)))
    
    # Create an array to block out which topologies are illegal
    illegal = np.ones(results.shape)
    # Tree with less than two children per parent node are illegal
    illegal = np.flip(np.tril(illegal, k=-2), 0)
    illegal[illegal == 0] = np.nan
    # The 4: here is set to the (max children - 1)
    illegal[-2, 4:] = 1
    # This is just for completeness, a depth of 1 can only be a single node (the root)
    illegal[-1, 1:] = 1
    
    # Here we take only the range requested
    # Note that e.g. depth starts at 1 (index=0), so a min_depth=2 is actually results[1:, ...] 
    im = ax.imshow(
        illegal[:-(min_depth - 1):, :-(min_leaves - 1)],
        extent=(min_leaves, max_leaves + 1, min_depth, max_depth + 1),
        cmap='Reds_r',
    )

    # Here we take only the range requested
    # Note that e.g. depth starts at 1 (index=0), so a min_depth=2 is actually results[1:, ...] 
    im = ax.imshow(
        results[(min_depth - 1):, (min_leaves - 1):],
        extent=(min_leaves, max_leaves + 1, min_depth, max_depth + 1),
        origin='lower',
        vmin=vmin,
        vmax=vmax,
    )

    # Set the xticks, dropping last one to make it not add a tick outside what has plot values
    ax.set_xticks([i + 0.5 for i in leaves_bins])
    ax.set_xticklabels(leaves_bins)
    
    # Hide every second label for clarity
    for xlabel in ax.xaxis.get_ticklabels()[1::2]:
        xlabel.set_visible(False)
    
    ax.set_yticks([i + 0.5 for i in depths_bins])
    ax.set_yticklabels(depths_bins)

    # Add the model label
    text_box = AnchoredText(
        label, frameon=True, loc='lower right', pad=0.4,
        borderpad=0.1,
    )
    plt.setp(text_box.patch, facecolor='white', alpha=0.8)
    ax.add_artist(text_box)
    
    return im


# archs = {
# #     'dataset': 'dataset',
#     'nri_model': 'NRI',
# }
save = True

# +
fig, axes = plt.subplots(nrows=len(model_dict) + 1, ncols=1, figsize=(10,8))

ims = []

# Always plot dataset on top
ax = axes.flat[0]
ims.append(single_figure(ax, 'Dataset', test_results['dataset']))
ax.set_ylabel('Depth')
cbar = fig.colorbar(ims[0], ax=ax, shrink=0.8, aspect=10)
cbar.set_label('No. topologies')


for idx, arch in enumerate(model_dict):
    
    ax = axes.flat[idx+1]
    ims.append(single_figure(ax, arch, test_results[arch]['perfect'], vmin=0, vmax=1))
    ax.set_ylabel('Depth')


plt.xlabel('Leaves')

cbar = fig.colorbar(ims[1], ax=axes.flat[1:], shrink=0.8, aspect=19)
cbar.set_label('perfectLCA score')

# Save the figure if requested
if save:
    plt.savefig(
        save_path / f"{'_'.join(list(model_dict.keys()))}_{mode}_evaluation_heatmap.pdf",
        transparent=True,
        bbox_inches='tight',
    )

plt.show()
# -

# ## Plotting without dataset

# +
fig, axes = plt.subplots(nrows=len(model_dict), ncols=1, figsize=(10,3*len(model_dict)))

ims = []

for idx, arch in enumerate(model_dict):
    
    ax = axes.flat[idx] if isinstance(axes, np.ndarray) else axes
    ims.append(single_figure(ax, arch, test_results[arch]['perfect'], vmin=0, vmax=1))
    ax.set_ylabel('Depth')


plt.xlabel('Leaves')

cbar_ax = axes.flat[1:] if isinstance(axes, np.ndarray) else axes
cbar = fig.colorbar(ims[-1], ax=cbar_ax, shrink=0.8, aspect=19)
cbar.set_label('perfectLCA score')

# Save the figure if requested
if save:
    plt.savefig(
        save_path / f"{'_'.join(list(model_dict.keys()))}_{mode}_evaluation_heatmap_no_dataset.pdf",
        transparent=True,
        bbox_inches='tight',
    )

plt.show()
