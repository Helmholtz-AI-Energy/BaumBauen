# import os
# import re
from pathlib import Path
import numpy as np
import torch

from ..utils import decay2lca
from ..utils import assign_parenthetical_weight_tuples


class TreeSet(torch.utils.data.Dataset):
    """ Dataset holding trees to feed to network"""
    def __init__(self, x, y):
        """ In our use x will be the array of leaf attributes and y the LCA matrix, i.e. the labels"""
        self.x = x
        self.y = y
        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.long))


class PhasespaceSet(TreeSet):
    def __init__(
        self,
        root,
        mode='train',
        file_ids=None,
        samples=None,
        samples_per_topology=None,
        seed=None,
        apply_scaling=False,
        scaling_dict=None,
        **kwargs,
    ):
        """ Dataset handler thingy for data generated with PhaseSpace

        Args:
            root(str or Path): the root folder containing all files belonging to the dataset in its different modes or partitions
            mode(str): 'train', 'val' or 'test' mode
            file_id(int or list(int)): Optional identifier or list of identifiers of files to load
            samples(int): number of samples to load total, will be a random subset. If larger than the samples in loaded files then this is ignored
            samples_per_topology(int): number of samples to load from each file, will be a random subset. If larger than the samples in loaded files then this is ignored
            seed(int): Random seed to use for selecting sample subset
            apply_scaling(bool): Whether to apply standard scaling to features, scaling factors will be calculated if scaling_dict is None.
            scaling_dict(dict): Scaling factors contained as {mean: float, std: float}
        """
        self.root = Path(root)

        self.modes = ['train', 'val', 'test']
        if mode not in self.modes:
            raise ValueError("unknown mode")

        self.mode = mode
        # self.known = known
        self.apply_scaling = apply_scaling
        self.scaling_dict = scaling_dict

        if seed is not None:
            np.random.seed(seed)

        x_files = sorted(self.root.glob(f'leaves_{mode}.*'))
        y_files = sorted(self.root.glob(f'lcas_{mode}.*'))

        assert len(x_files) > 0, f"No files to load found in {self.root}"

        # This deals with 0 padding dataset IDs since python will convert the input numbers to ints
        # Assumes files are save as xxxxx.<file_id>.npy
        if file_ids is not None:
            # Make sure file_ids is a list of ints
            file_ids = [int(file_ids)] if not isinstance(file_ids, list) else [int(i) for i in file_ids]

            x_files = [x for x in x_files if int(x.suffixes[0][1:]) in file_ids]
            y_files = [y for y in y_files if int(y.suffixes[0][1:]) in file_ids]

        # # In the case we're not loading train files, need to separate the known topologies from the unknown
        # # This is very hackish, really should have place the un/known files in separate directories to begin with
        # if self.mode is not 'train':
        #     train_files = sorted(self.root.glob('leaves_train.*'))
        #     train_files = [x.suffixes[0] for x in train_files]

        #     if self.known:
        #         x_files = [x for x in x_files if x.suffixes[0] in train_files]
        #         y_files = [y for y in y_files if y.suffixes[0] in train_files]
        #     else:
        #         x_files = [x for x in x_files if not x.suffixes[0] in train_files]
        #         y_files = [y for y in y_files if not y.suffixes[0] in train_files]

        if len(x_files) != len(y_files):
            raise RuntimeError(f'"leaves" and "lcas" files in {self.root} don\'t match')

        self.x = [np.load(f) for f in x_files]
        self.y = [np.load(f) for f in y_files]

        # Don't assume same number of sample per topology
        self.samples_per_topology = np.array([a.shape[0] for a in self.x])

        # Select a random subset of samples from each topology
        if samples_per_topology is not None and samples_per_topology < min(self.samples_per_topology):
            # Choose which samples to take from each file
            ids = [np.random.choice(i, samples_per_topology, replace=False) for i in self.samples_per_topology]
            self.x = [f[ids[i]] for i, f in enumerate(self.x)]
            self.y = [f[ids[i]] for i, f in enumerate(self.y)]
            # Set a fixed number of samples per topology, could be an int but makes things messier later :(
            self.samples_per_topology = np.array([samples_per_topology] * len(self.x))

        # Need this to know which files to take samples from
        self.cum_samples = np.cumsum(self.samples_per_topology)
        # And this to know where in the files
        self.loc_samples = self.cum_samples - self.samples_per_topology

        # Intentionally selecting one subset of indexes for all files so it's reproducible
        # even if only a subset of the files are loaded
        # TODO: Change this to still keep x a list of topology arrays, just with differing lengths
        if samples is not None and samples < sum(self.samples_per_topology):
            # Need this to know which files to take samples from
            cum_samples = np.cumsum(self.samples_per_topology)
            # And this to know where in the files
            loc_samples = cum_samples - self.samples_per_topology

            ids = np.random.choice(cum_samples[-1], samples, replace=False)
            file_ids = np.searchsorted(cum_samples, ids, side='left')
            # Get the true ids locations in each file
            ids = ids - loc_samples[file_ids]
            # self.x = [arr[idx] for arr in self.x]
            # self.y = [arr[idx] for arr in self.y]
            # This is a lazy way to avoid more intelligently extracting the correct item in getitem below
            # It just pretends there's one sample per topology and (samples) number of topologies
            self.x = [self.x[f][i] for f, i in zip(file_ids, ids)]
            self.y = [self.y[f][i] for f, i in zip(file_ids, ids)]

            # Selecting a subset we have a list of individual topolgies
            self.samples_per_topology = 1

        # If scaling is requested, check the scaling factors exist and calculate them if not
        if apply_scaling and self.scaling_dict is None:
            # Calculate values from 10% of the data
            self.scaling_dict = self._calculate_scaling_dict(int(self.__len__() * 0.1))

    def _calculate_scaling_dict(self, n_samples=1000):
        ''' Calculate scalings to bring features around the [-1, 1] range.

        This calculates a standard normalisation, i.e.:
            (x - mean)/std

        Args:
            n_samples (int, optional): Number of samples to use when calculating scalings
        Returns:
            Scaling dictionary containing {mean, std}  arrays of values for each feature
        '''
        # Select a random subset to calculate scalings from
        # In this case treating all samples as equal, so flatten them as if it's one long list of detected particles
        # Alternative approach would be to calculate the mean/std of each sample individually, then calculate their means
        x_sample = np.concatenate([self.__getitem__(i)[0] for i in np.random.choice(self.__len__(), size=n_samples, replace=False)])  # (n_samples*l, d)
        mean = np.mean(x_sample, axis=0)  # (d,)
        std = np.std(x_sample, axis=0)  # (d,)

        return {'mean': mean, 'std': std}

    def __getitem__(self, idx):
        ''' This currently has two modes:
            1. When self.x is still a list of one array per file (the if clause below)
            2. When self.x is a list of individual samples (the else clause below)
        '''
        idx = int(idx)
        if isinstance(self.samples_per_topology, np.ndarray):
            # Find file and location of this sample
            file_idx = np.searchsorted(self.cum_samples, idx, side='right')
            idx = idx - self.loc_samples[file_idx]
            item = [self.x[file_idx][idx], self.y[file_idx][idx]]
        else:
            item = [self.x[idx], self.y[idx]]

        # Apply scaling dict if requested
        if self.apply_scaling and self.scaling_dict is not None:
            item[0] = (item[0] - self.scaling_dict['mean']) / self.scaling_dict['std']

        # Set diagonal to -1, our padding must be -1 as well so we can tell the Loss to ignore it
        np.fill_diagonal(item[1], -1)

        return (
            torch.tensor(item[0], dtype=torch.float),  # (l, d)
            torch.tensor(item[1], dtype=torch.long)  # (l, l)  NOTE: confirm this
        )

    def __len__(self):
        # Handle case that we have selected samples randomly
        return self.samples_per_topology.sum() if isinstance(self.samples_per_topology, np.ndarray) else len(self.x)


def generate_phasespace(
        root,
        masses,
        fsp_masses,
        topologies=10,
        max_depth=5,
        max_children=6,
        min_children=2,
        isp_weight=1.,
        train_events_per_top=5,
        val_events_per_top=10,
        test_events_per_top=10,
        seed=None,
        iso_retries=0,
        generate_unknown=True,
):
    """ Generate a PhaseSpace dataset

    Args:
        root (str or Path): root folder
        masses (list): intermediate particle masses, root mass is masses[0]
        fsp_masses (list): list of final state particles
        topologies (int): number of decay tree topologies to generate, twice this number
                          for validation and thrice this number for testing
        max_depth (int): maximum allowed depth of the decay trees
        max_children (int): maximum allowed number of children for intermediate particles
        min_children (int): minumum required number of children for intermediate particles
                            (can fall short if kinematically impossible)
        isp_weight (float): relative weight of intermediate state particle probability (higher is more likely than fsp)
        train_events_per_top(int): number of training samples generated per decay tree
        val_events_per_top(int): number of validation samples generated per decay tree
        test_events_per_top(int): number of test samples generated per decay tree
        seed(int): RNG seed
        iso_retries(int):  if this is <= 0, does not perform isomorphism checks between generated topologies.
                           if > 0 gives the number of retries to ensure non-isomorphic topologies before raising
    """

    from phasespace import GenParticle
    # import tensorflow as tf

    if seed is not None:
        np.random.seed(seed)
        # This is supposed to be supported as a global seed for Phasespace but doesn't work
        # Instead we set the seed below in the calls to generate()
        # tf.random.set_seed(np.random.randint(np.iinfo(np.int32).max))

    if int(max_depth) <= 1:
        raise ValueError("Tree needs to have at least two levels")

    if int(min_children) < 2:
        raise ValueError("min_children must be two or more")

    masses = sorted(masses, reverse=True)
    fsp_masses = sorted(fsp_masses, reverse=True)
    if not set(masses).isdisjoint(set(fsp_masses)):
        raise ValueError("Particles are only identified by their masses. Final state particle masses can not occur in intermediate particle masses.")

    events_per_mode = {'train': train_events_per_top, 'val': val_events_per_top, 'test': test_events_per_top}

    topology_isomorphism_invariates = []

    total_topologies = topologies
    if generate_unknown:
        total_topologies = 3 * topologies

    for i in range(total_topologies):
        # NOTE generate tree for a topology
        for j in range(max(1, iso_retries)):
            queue = []
            root_node = GenParticle('root', masses[0])
            queue.append((root_node, 1))
            name = 1
            next_level = 1
            num_leaves = 0
            while len(queue) > 0:
                node, level = queue.pop(0)
                if next_level <= level:
                    next_level = level + 1
                num_children = np.random.randint(min_children, max_children + 1)

                total_child_mass = 0
                children = []

                # Mass we have to play with
                avail_mass = node._mass_val
                # Add an insurance to make sure it's actually possible to generate two children
                if avail_mass <= (2 * min(fsp_masses)):
                    raise ValueError("Any ISP mass given has to be larger than two times the smallest FSP mass.")

                for k in range(num_children):
                    # Only want to select children from mass/energy available
                    avail_mass -= total_child_mass

                    # Check we have enough mass left to generate another child
                    if avail_mass <= min(fsp_masses):
                        break

                    # use fsps if last generation or at random determined by number of possible masses and isp weight
                    if (
                        next_level == max_depth
                        or avail_mass <= min(masses)
                        or np.random.random() < (1. * len(fsp_masses)) / ((1. * len(fsp_masses)) + (isp_weight * len(masses)))
                    ):
                        child_mass = np.random.choice([n for n in fsp_masses if (n < avail_mass)])
                    else:
                        child_mass = np.random.choice([n for n in masses if (n < avail_mass)])
                    total_child_mass += child_mass

                    if total_child_mass > node._mass_val:
                        break

                    child = GenParticle(str(name), child_mass)
                    children.append(child)
                    name += 1
                    if child_mass in masses:
                        queue.append((child, next_level))
                    else:
                        num_leaves += 1

                node.set_children(*children)

            # NOTE if iso_retries given, check if topology already represented in dataset
            top_iso_invar = assign_parenthetical_weight_tuples(root_node)
            if iso_retries <= 0 or top_iso_invar not in topology_isomorphism_invariates:
                lca, names = decay2lca(root_node)
                topology_isomorphism_invariates.append(top_iso_invar)
                break
            if j == (iso_retries - 1):
                raise RuntimeError("Could not find sufficient number of non-isomorphic topologies.")
                # print("Could not find sufficient number of non-isomorphic topologies.")
                # continue

        # NOTE generate leaves and labels for training, validation, and testing
        modes = []
        # For topologies not in the training set, save them to a different subdir
        save_dir = Path(root, 'unknown')
        if i < topologies or not generate_unknown:
            modes = ['train', 'val', 'test']
            save_dir = Path(root, 'known')
        elif i < (2 * topologies):
            modes = ['val', 'test']
        else:
            modes = ['test']
        save_dir.mkdir(parents=True, exist_ok=True)

        for mode in modes:
            num_events = events_per_mode[mode]
            weights, particles = root_node.generate(
                num_events,
                seed=np.random.randint(np.iinfo(np.int32).max),
            )
            leaves = np.asarray([particles[name] for name in names])
            leaves = leaves.swapaxes(0, 1)
            assert leaves.shape == (num_events, num_leaves, 4)

            # NOTE shuffle leaves for each sample
            leaves, lcas = shuffle_leaves(leaves, lca)

            pad_len = len(str(total_topologies))
            leaves_file = save_dir / f'leaves_{mode}.{i:0{pad_len}}.npy'
            lcas_file = save_dir / f'lcas_{mode}.{i:0{pad_len}}.npy'
            np.save(leaves_file, leaves)
            np.save(lcas_file, lcas)

        del lca
        del names


def shuffle_leaves(leaves, lca):
    """
    leaves (torch.Tensor): tensor containing leaves of shape (num_samples. num_leaves, num_features)
    lca torch.Tensor): tensor containing lca matrix of simulated decay of shape (num_leaves, num_leaves)
    """
    assert leaves.shape[1] == lca.shape[0]
    assert leaves.shape[1] == lca.shape[1]
    d = lca.shape[1]

    shuff_leaves = np.zeros(leaves.shape)
    shuff_lca = np.zeros((leaves.shape[0], *(lca.shape)))

    for idx in np.arange(leaves.shape[0]):
        perms = np.random.permutation(d)
        shuff_leaves[idx] = leaves[idx, perms]
        shuff_lca[idx] = lca[perms][:, perms]

    return shuff_leaves, shuff_lca
