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
import optuna

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
              type=click.Choice(['transformer', 'nri'], case_sensitive=False),
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
    # est = baumbauen.models.Estimator()

    # First figure out which device all this is running on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configs
    configs, tags = bb.config.load_config(
        Path(cfg_path).resolve(),
        model=model,
        dataset=dataset,
        run_name=run_name,
    )

    # Fetch the model name
    selected_model = configs['train']['model']

    # Load datasets
    mode_tags = bb.data.create_dataloader_mode_tags(configs, tags)

    # Extract the number of features, assuming last dim is features
    infeatures = mode_tags['Training'][1][0][0].shape[-1]
    print(f'Input features: {infeatures}')
    print(configs)

    def objective(trial):

        # Build the model

        # First need to extract info about the datasets that doesn't come from config file
        # num_features = train_set.__getitem__(0)[0][-1].size(0)

        # Set up this trial's training params
        # All are categorial so we can simplify this
        if 'optuna' in configs[selected_model]:
            for param in configs[selected_model]['optuna']:
                configs[selected_model][param] = trial.suggest_categorical(
                    param,
                    configs[selected_model]['optuna'][param],
                )
        if 'explore' in configs['train']['optuna']:
            for param in configs['train']['optuna']['explore']:
                configs['train'][param] = trial.suggest_categorical(
                    param,
                    configs['train']['optuna']['explore'][param],
                )

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

        model = model.to(device)
        print(model)

        # Generate class weights if requested
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

        # Set up the actual checkpoints and save the configs if requested
        bb_ignite_trainer.setup_handlers(
            cfg_filename=Path(cfg_path).name,
        )

        # Add Optuna pruning to ignite
        pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(
            trial,
            "pad_accuracy",
            bb_ignite_trainer.trainer,
        )
        # Note that we attach it to the evaluator
        bb_ignite_trainer.evaluators['Validation Known'].add_event_handler(ig.engine.Events.COMPLETED, pruning_handler)

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

        # Print this trial's params
        print("Trial  Params: ")
        for key, value in trial.params.items():
            print("    {}: {} \n".format(key, value))

        # Actually run the training, mode_tags calls the train_loader
        train_steps = configs['train']['steps'] if 'steps' in configs['train'] else None
        bb_ignite_trainer.trainer.run(mode_tags['Training'][2], max_epochs=configs['train']['epochs'], epoch_length=train_steps)

        return bb_ignite_trainer.evaluators[bb_ignite_trainer.monitor_tag].state.metrics['perfect']

    Path(configs['train']['optuna']['study_path']).mkdir(parents=True, exist_ok=True)
    study_name = configs['output']['run_name']
    study_file = f"sqlite:///{configs['train']['optuna']['study_path']}/{study_name}.db"

    # Create a study with 3 warmup epochs
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        study_name=study_name,
        storage=study_file,
        load_if_exists=True,
    )
    print(f"Optuna sampler is {study.sampler.__class__.__name__}")

    study.optimize(
        objective,
        n_trials=configs['train']['optuna']['ntrials'],
        timeout=configs['train']['optuna']['timeout'],
        gc_after_trial=True,
    )

    print("Number of finished trials: ", len(study.trials), "\n")

    trial = study.best_trial
    print("Best trial:", trial.value, "\n")

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {} \n".format(key, value))

    # %%
    print(study.best_trial)


if __name__ == '__main__':
    main()
