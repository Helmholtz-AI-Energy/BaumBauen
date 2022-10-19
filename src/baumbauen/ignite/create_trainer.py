import ignite as ig
import ignite.contrib.handlers
import ignite.contrib.metrics
import torch
import numpy as np
import collections.abc
import baumbauen as bb
from datetime import datetime
from pathlib import Path
import yaml


class BBIgniteTrainer:
    ''' A class to setup the ignite trainer and hold all the things associated

    '''
    def __init__(self, model, optimizer, loss_fn, device, configs, tags, ignore_index=-1., include_efficiency=False):
        ''' These are all the inputs to ignite's create_supervised_trainer plus the yaml configs

        Args:
            model(Torch Model): The actual PyTorch model
            optimizer(Torch Optimizer): Optimizer used in training
            loss_fn(Torch Loss): Loss function
            device(Torch Device): Device to use
            configs(dict): Dictionary of run configs from loaded YAML config file
            tags(list): Various tags to sort train and validation evaluators by, e.g. "Training", "Validation Known"
            ignore_index(int): Label index to ignore when calculating metrics, e.g. padding
            include_efficiency(bool): Whether to include efficiency and purity metrics. Slows down computation significantly due to building the adjacency to do to
        '''

        self.model = model
        self.optimizer = optimizer
        self.configs = configs
        self.tags = tags
        self.ignore_index = ignore_index
        self.include_efficiency = include_efficiency

        # Run timestamp to distinguish trainings
        self.timestamp = datetime.now().strftime('%Y.%m.%d_%H.%M')
        self.source = self.configs['dataset']['source']
        # And record monitor tag to know which data to watch
        self.monitor_tag = self.configs['dataset'][self.source]['config']['monitor_tag']

        # Output directory for checkpoints
        self.run_dir = None
        # Output directory for Tensorboard logging
        self.tb_dir = None
        if self.configs['output'][self.source] is not None:
            if ('path' in self.configs['output'][self.source].keys()) and (self.configs['output'][self.source]['path'] is not None):
                self.run_dir = Path(
                    self.configs['output'][self.source]['path'],
                    self.configs['output']['run_name'],
                )

            if ('tensorboard' in self.configs['output'][self.source].keys()) and (self.configs['output'][self.source]['tensorboard'] is not None):
                # Need a timestamp to organise runs in tensorboard
                self.tb_dir = Path(
                    self.configs['output'][self.source]['tensorboard'],
                    self.configs['output']['run_name'],
                    self.timestamp
                )

        # Setup ignite trainer
        self.trainer = ig.engine.create_supervised_trainer(
            model,
            optimizer,
            loss_fn,
            device=device,
            amp_mode='amp' if configs['train']['mixed_precision'] else None,
            scaler=True if configs['train']['mixed_precision'] else None,
        )

        # Setup train and validation evaluators
        self.evaluators = {}

        for tag in self.tags:
            # Setup metrics
            metrics = {
                "loss": ig.metrics.Loss(loss_fn, device=device),
                "pad_accuracy": bb.metrics.Pad_Accuracy(ignored_class=ignore_index, device=device),
                "accuracy_primary": bb.metrics.Pad_Accuracy(ignored_class=[ignore_index, 0], device=device),
                "perfect": bb.metrics.PerfectLCA(ignore_index=ignore_index, device=device),
                "perfect_primary": bb.metrics.PerfectLCA(ignore_index=[ignore_index, 0], device=device),
            }
            if (tag != "Training") and self.include_efficiency:
                metrics["efficiency"] = bb.metrics.Efficiency(ignore_index=ignore_index, ignore_disconnected_leaves=True, device=device)

            self.evaluators[tag] = ig.engine.create_supervised_evaluator(
                model,
                metrics=metrics,
                device=device,
                amp_mode='amp' if configs['train']['mixed_precision'] else None,
            )

            # Add GPU memory info
            if self.configs['train']['record_gpu_usage']:
                ig.contrib.metrics.GpuInfo().attach(self.trainer, name='gpu')  # metric names are 'gpu:X mem(%)', 'gpu:X util(%)'
                ig.contrib.metrics.GpuInfo().attach(self.evaluators[tag], name='gpu')  # metric names are 'gpu:X mem(%)', 'gpu:X util(%)'
                # metrics['gpu'] = ig.contrib.metrics.GpuInfo()
                # metrics['gpu:0 mem(%)'] = ig.contrib.metrics.GpuInfo()

            # train_evaluator.logger = ig.utils.setup_logger("Train Evaluator")

        # Call function to setup handlers
        # self._setup_handlers()

    def score_fn(self, engine):
        ''' Metric to use for early stoppging '''
        return engine.state.metrics["pad_accuracy"]

    def lca_score_fn(self, engine):
        ''' Metric to use for checkpoints '''
        return engine.state.metrics["perfect"]

    def _clean_config_dict(self, configs):
        ''' Clean configs to prepare them for writing to file

        This will convert any non-native types to Python natives.
        Currently just converts numpy arrays to lists.

        Args:
            configs (dict): Config dictionary

        Returns:
            dict: Cleaned config dict
        '''
        # TODO: Add torch conversions as well
        for k, v in configs.items():
            if isinstance(v, collections.abc.Mapping):
                configs[k] = self._clean_config_dict(configs[k])
            elif isinstance(v, np.ndarray):
                configs[k] = v.tolist()
            else:
                configs[k] = v
        return configs

    def setup_handlers(self, cfg_filename='config.yaml'):
        ''' Create the various ignite handlers (callbacks)

        Args:
            cfg_filename(str, optional): Name of config yaml file to use when saving configs
        '''
        # Create the output directory
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            # And save the configs, putting here to only save when setting up checkpointing
            with open(self.run_dir / f'{self.timestamp}_{cfg_filename}', 'w') as outfile:
                cleaned_configs = self._clean_config_dict(self.configs)
                yaml.dump(cleaned_configs, outfile, default_flow_style=False)

        # Attach a progress bar
        if self.configs['train']['progress_bar']:
            progress_metrics = ['gpu:0 mem(%)', 'gpu:0 util(%)'] if self.configs['train']['record_gpu_usage'] else None
            pbar = ig.contrib.handlers.ProgressBar(persist=True, bar_format="")
            # pbar.attach(self.trainer, output_transform=lambda x: {'loss': x})
            # pbar.attach(self.trainer, metric_names='all')
            pbar.attach(self.trainer, metric_names=progress_metrics, output_transform=lambda x: {'loss': x})

        # Setup early stopping
        early_handler = ig.handlers.EarlyStopping(
            patience=self.configs['train']['early_stop_patience'],
            score_function=self.score_fn,
            trainer=self.trainer,
            min_delta=1e-3,
        )
        self.evaluators[self.monitor_tag].add_event_handler(ig.engine.Events.EPOCH_COMPLETED, early_handler)

        # Configure saving the best performing model
        if self.run_dir is not None:
            to_save = {
                'model': self.model,
                'optimizer': self.optimizer,
                'trainer': self.trainer
            }
            # Note that we judge early stopping above by the validation loss, but save the best model
            # according to validation perfectLCA score. This lets training continue for perfectLCA plateaus
            # so long as the model is still changing (and hopefully improving again after some time).
            best_model_handler = ig.handlers.Checkpoint(
                to_save=to_save,
                save_handler=ig.handlers.DiskSaver(self.run_dir, create_dir=True, require_empty=False),
                filename_prefix=self.timestamp,
                score_function=self.lca_score_fn,
                score_name="validation_perfect",
                n_saved=1,
                global_step_transform=ig.handlers.global_step_from_engine(self.evaluators[self.monitor_tag]),
            )
            self.evaluators[self.monitor_tag].add_event_handler(ig.engine.Events.EPOCH_COMPLETED, best_model_handler)

        # Attach Tensorboard logging
        if self.tb_dir is not None:
            tb_logger = ig.contrib.handlers.TensorboardLogger(log_dir=self.tb_dir, max_queue=1)

            # Attach the logger to the evaluator on the training and validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            for tag in self.tags:
                tb_logger.attach(
                    self.evaluators[tag],
                    event_name=ig.engine.Events.EPOCH_COMPLETED,
                    log_handler=ig.contrib.handlers.tensorboard_logger.OutputHandler(
                        tag=tag,
                        metric_names='all',
                        global_step_transform=ig.handlers.global_step_from_engine(self.trainer),
                    ),
                )

        return

    # Set up end of epoch validation procedure
    # Tell it to print epoch results
    def log_results(self, trainer, mode_tags):
        ''' Callback to run evaluation and report the results.

        We place this here since it needs access to the evaluator engines in order to run.
        No idea why ignite even bother to pass the trainer engine in the first place, their examples all call
        things created outside of the log_results function that aren't passed to it... bad programming practice in my opinion.

        Call this function via the add_event_handler() ignite function to tell it when to fire, e.g.:
            `BBIgniteTrainer.trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, BBIgniteTrainer.log_results, mode_tags)`

        Args:
            trainer (ignite.Engine): trainer that gets passed by ignite to this method.
            mode_tags (dict): Dictionary of mode tags containing (mode, dataset, dataloader) tuples
        '''

        for tag, values in mode_tags.items():
            evaluator = self.evaluators[tag]

            eval_steps = self.configs['val']['steps'] if 'steps' in self.configs['val'] else None
            # Need to wrap this in autocast since it caculates metrics (i.e. loss) without autocast switched on
            # This is mostly fine except it fails to correctly cast the class weights tensor passed to the loss
            if self.configs['train']['mixed_precision']:
                with torch.cuda.amp.autocast():
                    evaluator.run(values[2], epoch_length=eval_steps)
            else:
                evaluator.run(values[2], epoch_length=eval_steps)

            metrics = evaluator.state.metrics
            message = [f'{tag} Results - Epoch: {trainer.state.epoch}']
            message.extend([f'Avg {m}: {metrics[m]:.4f}' for m in metrics])
            print(message)

            # Prints an example of a predicted LCA (for debugging only)
            # evaluator.state.output holds (y_pred, y)
            if 'print_sample' in self.configs['train'] and self.configs['train']['print_sample']:
                y_pred, y = evaluator.state.output
                probs = torch.softmax(y_pred, dim=1)  # (N, C, d1, d2)
                winners = probs.argmax(dim=1)  # (N, d1, d2)
                mask = y == -1
                winners[mask] = -1
                print(values[1][0][0])  # Input features
                print(winners[0])
                print(y[0])
                # print(values[2][0])
