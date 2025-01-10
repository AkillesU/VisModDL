import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
from utils.py import categ_corr_lineplot

main_directory = "C:/Users/arech/OneDrive - Royal Holloway University of London/Documents/code_repositories/VisModDL/figures/haupt_stim_activ/damaged/cornet_rt/noise/IT/selectivity/"

# Example: categories and metrics we want to plot
categories_to_use = ["total","animal","face","object","place"]
metrics_to_use = "avg_within"

categ_corr_lineplot(
    main_dir=main_directory,
    categories=categories_to_use,
    metric=metrics_to_use
)
