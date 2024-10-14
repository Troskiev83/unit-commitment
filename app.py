#########  THIS FILE WILL BE REPLACED  #########

input_file_name = "input.json"

################################################
#########    DO NOT TOUCH FROM HERE    #########
################################################

# Input data loader. Container will get data from here

import json
import warnings
warnings.filterwarnings('ignore')
#warnings.filterwarnings('ignore', category=DeprecationWarning)
with open(input_file_name) as f:
  dic = json.load(f)

# Optional extra parameters

if "extra_arguments" in dic:
    extra_arguments = dic['extra_arguments']
else:
    extra_arguments = {}

if "solver_params" in dic:
    solver_params = dic['solver_params']
else:
    solver_params = {}


import main
result = main.run(dic['data'], solver_params, extra_arguments)
print(result)