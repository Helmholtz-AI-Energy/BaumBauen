#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
from pathlib import Path

import click
from IPython.core import ultratb  # noqa

import torch
import ignite as ig
import baumbauen as bb

# fallback to debugger on error
# sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-c', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='path to config file')
@click.option('-r', '--run', 'run_name', required=False,
              type=str, help='Name of run')
@click.option('-s', '--samples', 'samples', required=False,
              type=int, help='Number of samples to train on')
@click.option('-m', '--model', 'model', required=False,
              type=click.Choice(['transformer_model', 'nri_model'], case_sensitive=False),
              help='Model architecture to use')
@click.option('-d', '--dataset', 'dataset', required=False,
              type=int, help='Individual dataset to load')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(bb.__version__)
def main(
    cfg_path: Path,
    run_name: str,
    samples: int,
    model: str,
    dataset: int,
    log_level: int,
):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # YOUR CODE GOES HERE! Keep the main functionality in src/baumbauen

    # First figure out which device all this is running on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configs
    configs, tags = bb.config.load_config(
        Path(cfg_path).resolve(),
        model=model,
        dataset=dataset,
        run_name=run_name,
        samples=samples,
    )

    # Fetch the model name
    selected_model = configs['train']['model']

    # Load datasets
    mode_tags = bb.data.create_dataloader_mode_tags(configs, tags)

    # Extract the number of features, assuming last dim is features
    infeatures = mode_tags['Training'][1][0][0].shape[-1]
    print(f'Input features: {infeatures}')
    print(configs)

    # Build the model
    # First need to extract info about the datasets that doesn't come from config file
    # num_features = train_set.__getitem__(0)[0][-1].size(0)

    # Now build the model
    if selected_model == 'transformer_model':

        model = bb.models.BBTransformerBaseline(
            infeatures=infeatures,
            num_classes=configs['dataset']['num_classes'],
            **configs[selected_model],
        )

    elif selected_model == 'nri_model':

        model = bb.models.NRIModel(
            infeatures=infeatures,
            # infeatures=len(mode_tags["Training"][1].features),  # Extract from the dataset, allows ignoring features
            num_classes=configs['dataset']['num_classes'],
            **configs[selected_model],
        )

    # Set the loss
    # Frist generate class weights if requested
    class_weights = None
    if configs['train']['class_weights']:
        class_weights = bb.utils.calculate_class_weights(
            dataloader=mode_tags['Training'][2],
            num_classes=configs['dataset']['num_classes'],
            num_batches=100,
            amp_enabled=configs['train']['mixed_precision'],
        )
        class_weights = class_weights.to(device)

    if configs[selected_model]['loss'] == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    elif configs[selected_model]['loss'] == 'focal':
        loss_fn = bb.losses.FocalLoss(gamma=2.5, ignore_index=-1, weight=class_weights)

    # Set the optimiser
    if configs['train']['optimiser'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), configs['train']['learning_rate'], amsgrad=False)
    elif configs['train']['optimiser'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), configs['train']['learning_rate'])
    elif configs['train']['optimiser'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), configs['train']['learning_rate'])

    # Set up learning rate scheduler (ignite doesn't support this one yet as a handler)
    # See also the Events.COMPLETED call below
    # plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     factor=0.1,
    #     patience=configs['train']['lr_plateau_patience'],
    #     verbose=True,
    # )

    model.to(device)
    print(model)
    # print(dict(model.named_parameters()))

    bb_ignite_trainer = bb.ignite.BBIgniteTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        configs=configs,
        tags=list(mode_tags.keys()),
        ignore_index=-1,
        include_efficiency=configs['train']['include_efficiency'],
    )
    # TODO: Make the monitor tag part of config -- actually not sure if that's a good idea since it
    # contains the dataset/dataloader
    # Set up the actual checkpoints and save the configs if requested
    bb_ignite_trainer.setup_handlers(
        cfg_filename=Path(cfg_path).name,
    )

    # Add callback to run evaluation after each epoch
    bb_ignite_trainer.trainer.add_event_handler(
        ig.engine.Events.EPOCH_COMPLETED,
        bb_ignite_trainer.log_results,
        mode_tags,
    )

    # Event to activate the learning rate decay
    # @bb_ignite_trainer.trainer.on(ig.engine.Events.EPOCH_COMPLETED)
    # def reduct_step(trainer):
    #     # engine is evaluator
    #     # engine.metrics is a dict with metrics, e.g. {"loss": val_loss_value, "acc": val_acc_value}
    #     # scheduler.step(bb_ignite_trainer.evaluators['Validation Known'].metrics['loss'])
    #     plateau_scheduler.step(bb_ignite_trainer.evaluators['Training'].state.metrics['nll'])

    # Actually run the training, mode_tags calls the train_loader
    train_steps = configs['train']['steps'] if 'steps' in configs['train'] else None
    bb_ignite_trainer.trainer.run(mode_tags['Training'][2], max_epochs=configs['train']['epochs'], epoch_length=train_steps)


if __name__ == '__main__':
    main()
