# BaumBauen Scripts

This contains examples of scripts that call and run BaumBauen-related modules.

## Scripts organisation

```
└── training                <- Network training scripts.
```

### `training`

The main training scripts is `train_model.py`.
This has one input, the yaml config file `config.yaml`.
You should make your own copy of this (that's not part of the repo) to use as input.
Execute the training with:
```shell
python train_model.py -c config.yaml
```

There is also an example script for running an Optuna hyperparameter optimisation.
For this, all Optuna settings are set in the same `config.yaml` file.
