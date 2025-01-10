from utils import categ_corr_lineplot, plot_avg_corr_mat, plot_categ_differences

main_directory = "figures/haupt_stim_activ/damaged/cornet_rt/noise/V1/selectivity/"

# TODO: Fix this code, make it more modular 

# Categories and metrics we want to plot
categories_to_use = ["total","animal","face","object","place"]
metrics_to_use = "observed_difference"
image_directory = "stimuli/"

categ_corr_lineplot(
    main_dir=main_directory,
    categories=categories_to_use,
    metric=metrics_to_use
)

plot_avg_corr_mat(
    main_dir=main_directory,
    image_dir=image_directory,
    output_dir="figures/haupt_stim_activ/damaged/cornet_rt/connections/V1/average_RDMs",
    subdir_regex=r"damaged_([\d\.]+)$",  # e.g. "damaged_0.01"
    vmax=1.0
)


plot_categ_differences(parent_dir="figures/haupt_stim_activ/damaged/cornet_rt/connections/V1/RDM/",
                       image_dir=image_directory,
                       mode="dirs",
                       file_prefix="damaged_",
                       file_suffixes=["0.05","0.15","0.5","0.8"])