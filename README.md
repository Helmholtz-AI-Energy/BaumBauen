# BaumBauen

```
This is where we really bauen the Baums
```

This is the code for the paper [Learning tree structures from leaves for particle decay reconstruction](https://doi.org/10.1088/2632-2153/ac8de0).

The dataset used in the study is [available on Zenodo](https://doi.org/10.5281/zenodo.6983258).


## Installation

In order to set up the necessary environment:

1. Create a Python virtual environment for the project:
   ```
   python3 -m venv /path/to/baum-env
   ```
2. Activate the new environment with
   ```
   source /path/to/baum-env/bin/activate
   ```
3. The first time you'll need to update pip:
   ```
   pip3 install -U pip
   ```
4. Install the `baumbauen` package  with:
   ```
   cd /path/to/baumbauen/repo
   pip install .
   ```
   or if you're doing development work and want code changes to be reflected in the installed package immediately:
   ```
   pip install -e .
   ```


## Usage

There are two ways to see usage of the package: the examples in the [scripts](scripts/README.md) and [notebooks](notebooks) folders.

The scripts are for executing in a grid/batch environment, this includes `basf2` steering files (requires Belle II software) as well as PyTorch training scripts.

The notebooks are self-contained and are ordered according to workflow by their numbering (lettering indicates author).
They do require [Jupytext](https://github.com/mwouts/jupytext) to be able to read/commit notebooks (installed as a requirement for BaumBauen).
When opening notebooks select `File > Jupytext > Pair Notebook with light Script`


## Project Organization

```
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── LICENSE                 <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── baumbauen           <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## Note

This project has been set up using PyScaffold 3.2.3 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[Jupytext]: https://github.com/mwouts/jupytext
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
