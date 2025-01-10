import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
from utils.py import categ_corr_lineplot

# TODO: Fix this code, make it more modular 

# Categories and metrics we want to plot
categories_to_use = ["total","animal","face","object","place"]
metrics_to_use = "avg_within"

categ_corr_lineplot(
    layers=["IT","V4","V2","V1"],
    damage_type="noise",
    categories=["total"]
)
