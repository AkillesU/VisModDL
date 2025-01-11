from utils import categ_corr_lineplot, plot_categ_differences, plot_avg_corr_mat

# TODO: Fix this code, make it more modular 

# Categories and metrics we want to plot
categories_to_use = ["total","animal","face","object","place"]
metrics_to_use = "avg_within"

"""categ_corr_lineplot(
    layers=["IT","V4","V2","V1"],
    damage_type="noise",
    categories=["total"]
)
"""

"""plot_categ_differences(parent_dir="figures/haupt_stim_activ/damaged/cornet_rt/connections/V1/RDM",
image_dir="stimuli/",
mode="dirs",
file_prefix="damaged_",
file_suffixes=["0.1"]
)"""

plot_avg_corr_mat(layers=["V1","V2","V4","IT"],
                  damage_type="connections",
                  damage_levels=["0.8"]
                  )