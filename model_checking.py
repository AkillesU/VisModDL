from utils import load_model

model_info = {}

model_info["source"] = "cornet"
model_info["repo"] = "" #"pytorch/vision"  
model_info["name"] = "cornet_s"
model_info["weights"] = "" # "DEFAULT"


model = load_model(model_info= model_info,
                   pretrained=True,
                   layer_name="block5_pool",
                   layer_path="",
                   )


print(model)