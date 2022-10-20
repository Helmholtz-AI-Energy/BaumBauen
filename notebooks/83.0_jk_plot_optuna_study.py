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

# # Plot Optuna study
#
# Plot the results from an Optuna study and save the plots.

# %config Completer.use_jedi = False
import optuna
from pathlib import Path
from itertools import combinations
import plotly.graph_objs as go
import copy

# Set default plotting params

layout = {
    'title': None,
    'margin': go.layout.Margin(
        l=10, #left margin
        r=10, #right margin
        b=10, #bottom margin
        t=10  #top margin
    ),
    'font': {'size': 16}
}

# ## Load studies

# +
# Path to your study here
study_name = "my_study"  # Unique identifier of the study.

storage_name = f"sqlite:////path/to/optuna/{study_name}.db"
output_path = Path("/path/to/output/figures/", study_name)
output_path.mkdir(exist_ok=True)
# -

study = optuna.load_study(study_name=study_name, storage=storage_name)
# df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

# ## Filter studies for plotting a subset of params

# Create new study to store filtered trials
study_with_filtered_trials = optuna.create_study()

study_with_filtered_trials.add_trials(t for t in study.trials if t.values is not None and t.params['loss'] == 'cross_entropy') 

len(study_with_filtered_trials.trials)

fig = optuna.visualization.plot_slice(
    study_with_filtered_trials,
)
fig.show()

# ### Basic study info

len(study.trials)

study.trials[0].distributions

study.best_value

study.best_params

# ## Plot/save results

# +
fig = optuna.visualization.plot_contour(study)

# Custom margins
this_layout = copy.deepcopy(layout)
this_layout['margin'] = go.layout.Margin(
    l=50, #left margin
    r=10, #right margin
    b=120, #bottom margin
    t=50  #top margin
)
this_layout['font']['size'] = 12

fig.update_layout(this_layout)
fig.show()
fig.write_image(output_path / "contour.pdf")
# -

for pair in combinations(study.best_params.keys(), 2):
    fig = optuna.visualization.plot_contour(study, params=list(pair))
    fig.update_layout(layout)
    fig.show()
    fig.write_image(output_path / f"{pair[0]}-{pair[1]}_contour.pdf")

fig = optuna.visualization.plot_intermediate_values(study)
fig.update_layout(layout)
fig.show()
fig.write_image(output_path / "intermediate_values.pdf")

fig = optuna.visualization.plot_optimization_history(study)
fig.update_layout(layout)
fig.show()
fig.write_image(output_path / "history.pdf")

# +
fig = optuna.visualization.plot_parallel_coordinate(
    study,
#     params=[
# #         'nblocks',
#         'dim_feedforward',
# #         'block_additional_mlp_layers',
#         'final_mlp_layers',
#         'initial_mlp_layers',
# #         'loss',
#     ]
)
# Custom margins
this_layout = copy.deepcopy(layout)
this_layout['margin'] = go.layout.Margin(
    l=50, #left margin
    r=10, #right margin
    b=120, #bottom margin
    t=50  #top margin
)

fig.update_layout(this_layout)
fig.show()
fig.write_image(output_path / "parallel.pdf")

# +
fig = optuna.visualization.plot_slice(
    study,
#      params=[
# #         'nblocks',
#         'dim_feedforward',
# #         'block_additional_mlp_layers',
#         'final_mlp_layers',
#         'initial_mlp_layers',
# #         'loss',
#     ]
)
# Custom margins
this_layout = copy.deepcopy(layout)
# this_layout['height'] = 400
# this_layout['width'] = 900
# this_layout['margin'] = go.layout.Margin(
#     l=50, #left margin
#     r=10, #right margin
#     b=120, #bottom margin
#     t=50  #top margin
# )

fig.update_layout(this_layout)
fig.show()
fig.write_image(output_path / "slices.pdf")
