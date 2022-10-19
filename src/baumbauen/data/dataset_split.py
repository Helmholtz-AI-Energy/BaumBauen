import torch
from .datasets import PhasespaceSet
from ..utils import pad_collate_fn, rel_pad_collate_fn


def create_dataloader_mode_tags(configs, tags):
    '''
    Convenience function to create the dataset/dataloader for each mode tag (train/val/val unknown) and return them.

    *Important*: The scaling used on the PhasespaceSet will calculate the scaling factors according to the first dataset in tags if it
        doesn't exist in configs already and then pass this to subsequent datasets. Make sure the Training tag is first if that's what you
        want to use (you should).

    Args:
        configs (dict): Training configuration
        tags (list): Train/val mode tags containing dataset paths

    Returns:
        dict: Mode tag dictionary containing tuples of (mode, dataset, dataloader)
    '''

    mode_tags = {}
    selected_model = configs['train']['model']

    for tag, path, mode in tags:
        if configs['dataset']['source'] == 'phasespace':
            dataset = PhasespaceSet(
                root=path,
                mode=mode,
                file_ids=configs['dataset']['datasets'] if 'datasets' in configs['dataset']['phasespace'] else None,
                **configs['dataset']['phasespace']['config'],
            )

            # If scaling is requested and there's no scaling factors in the configs, extract them from the dataset
            # They will be calculated by the first dataset created without a scaling_dict given
            if configs['dataset']['phasespace']['config']['apply_scaling'] and configs['dataset']['phasespace']['config']['scaling_dict'] is None:
                configs['dataset']['phasespace']['config']['scaling_dict'] = dataset.scaling_dict

        else:
            raise NotImplementedError(f"{configs['dataset']['source']} data source not implemented.")

        print(f"{type(dataset).__name__} created for {mode} with {dataset.__len__()} samples")

        if selected_model == 'nri_model':
            def nri_collate_fn(batch):
                return rel_pad_collate_fn(batch, configs['nri_model']['self_interaction'])

            collate_fn = nri_collate_fn
        else:
            collate_fn = pad_collate_fn

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=configs[mode]['batch_size'],
            drop_last=False,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=configs[mode]['num_workers'],
            pin_memory=False,
            prefetch_factor=configs['train']['batch_size'],
        )
        mode_tags[tag] = (mode, dataset, dataloader)

    return mode_tags
