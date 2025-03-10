# VisModDL
Repo for visual deep learning modelling and ageing.

This is a project investigating deep learning modelling approaches for age-related neural damage. 
More specifically, the modelling is aimed to uncover the specific types of neural damage that contribute to
categorical dedifferentiation of visual representations apparent in healthy ageing. 

## Repo Structure

/configs

&nbsp;&nbsp;&nbsp;&nbsp;/all_params

&nbsp;&nbsp;&nbsp;&nbsp;/conv_layers

&nbsp;&nbsp;&nbsp;&nbsp;/plots

/data - contains the output data from model analyses

/plots - contains the output plots from model analyses

/stimuli - contains the stimuli used in the experiments

main.py - main script for running the model damage experiments

noise_test.py - script for determining appropriate noise levels to implement

run_plots.py - script for running the plotting functions

utils.py - utility functions for the main script


## Model naming convention:

For CORnet-RT

cornet_rt{n_timesteps}_{layers applied to}{bias or weights}{_std_noise}

layers applied to: c - only conv layers, all - both conv and normalisation layers

bias or weights: +b - bias and weights affected, [empty] - only weights affected

_std_noise: _std_noise - noise added is standardised based on the layer parameter distribution (mean & std). Normality is assumed.