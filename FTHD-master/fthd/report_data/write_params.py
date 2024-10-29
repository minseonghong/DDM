import yaml
import os

data = [5.680247783660889,1.1782020330429075,0.19252443313598633,-0.12342524528503418,6.002655506134033,1.109113097190857,0.17878597974777222,-0.2115284204483032,0.28695258498191833,0.054567769169807434,0.05173772573471069,0.0003405254101380706,0.00002785873948596418,-0.0015499349683523178,0.000655989395454526,-0.003982823342084885,0.0011168550699949265]

coeffs = ['Bf','Cf','Df','Ef','Br','Cr','Dr','Er','Cm1','Cm2','Cr0','Cr2','Iz','Shf','Svf','Shr','Svr']

data_dict = dict()
data_dict["PARAMETERS"] = []

for i,(value,coeff) in enumerate(zip(data,coeffs)):
    coeff_map = dict()
    coeff_map[coeff] = value
    data_dict["PARAMETERS"].append(coeff_map)

print(data_dict)

training_percent = "20"

# file_path = "cfg_fullset{}/Pinn_coeff_original.yaml".format(training_percent)
file_path = "cfg_fullset{}/Pinn_coeff_HybridAdam.yaml".format(training_percent)
# file_path = "cfg_fullset{}/Pinn_coeff_HybridDDP.yaml".format(training_percent)

report_file_path = os.path.join('benchmark',file_path)
dir_name = os.path.dirname(__file__)
file_path = os.path.join(dir_name,file_path)
report_file_path = os.path.join(dir_name,report_file_path)

with open(file_path, 'w') as file:
    yaml.dump(data_dict, file)

with open(report_file_path, 'w') as file:
    yaml.dump(data_dict, file)
