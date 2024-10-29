import matplotlib.pyplot as plt
import yaml
import numpy as np
import os
from matplotlib.ticker import LogLocator

# Example data
dir_name = os.path.dirname(__file__)
print(dir_name)
ground_truth_config_file = os.path.join(dir_name,"cfg/deep_dynamics_ddp_GT.yaml")
# trained_minVal_Adamconfig_file = os.path.join(dir_name,"cfg/deep_dynamics_adam_coeff.yaml")
# trained_minVal_DualDimerconfig_file = os.path.join(dir_name,"cfg/dual_dimerPinn_coeff.yaml")
# trained_minVal_Originalconfig_file = os.path.join(dir_name,"cfg/deep_dynamics_original_coeff.yaml")

training_percent = "80"
fontsize = 25
tick_label_fontsize = 25
legend_fontsize = 25

trained_minVal_Originalconfig_file = "cfg_fullset{}/Pinn_coeff_original.yaml".format(training_percent)
trained_minVal_Adamconfig_file = "cfg_fullset{}/Pinn_coeff_HybridAdam.yaml".format(training_percent)
# trained_minVal_DualDimerconfig_file = "cfg_fullset{}/Pinn_coeff_HybridDDP.yaml".format(training_percent)

trained_minVal_Adamconfig_file = os.path.join(dir_name,trained_minVal_Adamconfig_file)
# trained_minVal_DualDimerconfig_file = os.path.join(dir_name,trained_minVal_DualDimerconfig_file)
trained_minVal_Originalconfig_file = os.path.join(dir_name,trained_minVal_Originalconfig_file)


with open(ground_truth_config_file, 'rb') as f:
    GT_param_dict = yaml.load(f, Loader=yaml.SafeLoader)

with open(trained_minVal_Adamconfig_file, 'rb') as f:
    trained_ADAMparam_dict = yaml.load(f, Loader=yaml.SafeLoader)

# with open(trained_minVal_DualDimerconfig_file, 'rb') as f:
#     trained_DUALDIMERparam_dict = yaml.load(f, Loader=yaml.SafeLoader)

with open(trained_minVal_Originalconfig_file, 'rb') as f:
    trained_ORIGINALparam_dict = yaml.load(f, Loader=yaml.SafeLoader)

groundTruth_value = []
trainedADAMCoef_vale = []
# trainedDUALDIMERCoef_vale = []
trainedORIGINALCoef_vale = []


for i in range(len(GT_param_dict["PARAMETERS"])):
    groundTruth_value.append(list(GT_param_dict["PARAMETERS"][i].values())[0])
    trainedADAMCoef_vale.append(list(trained_ADAMparam_dict["PARAMETERS"][i].values())[0])
    # trainedDUALDIMERCoef_vale.append(list(trained_DUALDIMERparam_dict["PARAMETERS"][i].values())[0])
    trainedORIGINALCoef_vale.append(list(trained_ORIGINALparam_dict["PARAMETERS"][i].values())[0])

print(groundTruth_value)

coeffs = ['$B_f$','$C_f$','$D_f$','$E_f$','$B_r$','$C_r$','$D_r$','$E_r$','$C_{m1}$','$C_{m2}$','$C_{r0}$','$C_{r2}$','$I_z$','$S_{hf}$','$S_{vf}$','$S_{hr}$','$S_{vr}$']

data_set = ['FTHD','DDM']
# Create figure and axis
colors = ['black','red']
# markers = ['o','^','x']
fig, ax = plt.subplots(figsize=(9, 8.0))

# Set x-axis to log scale
ax.set_xscale('log')
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='all', numticks=10))

# Plot a vertical line for the ground truth
ax.axvline(x=0, color='k', linestyle='--')  # x=0 because we'll plot relative to the ground truth

y_positions = np.arange(len(groundTruth_value))
pos = 0
label_pos = 0

# Plot data points relative to the ground truth
for set_idx, (ground_truth, trained_adamdata_points,trained_originaldata_points) in enumerate(zip(groundTruth_value, trainedADAMCoef_vale, trainedORIGINALCoef_vale)):
    # Adjust data points relative to ground truth

    label = coeffs[set_idx]
    if label == "$C_{r2}$":
        label = "$C_d$"
    # if coeffs[set_idx] == "Cm1" or coeffs[set_idx] == "Cm2" or coeffs[set_idx] == "Cr0" or coeffs[set_idx] == "Cr2":
    #     continue
    # sign_points = np.array([trained_adamdata_points * ground_truth,
    #                         trained_dualdimerdata_points * ground_truth,
    #                         trained_originaldata_points * ground_truth])
    
    adjusted_points = np.abs([trained_adamdata_points - ground_truth,
                       trained_originaldata_points - ground_truth])
    
    # adjusted_points = np.exp(adjusted_points)
    
    # average_distance = np.mean(adjusted_points)
    
    # adjusted_points[sign_points<0] = np.abs(adjusted_points[sign_points<0])+1
    # print(adjusted_points)
    # adjusted_points = np.log10(adjusted_points)
    # adjusted_points[sign_points<0] = -adjusted_points[sign_points<0]
    label_pos += 1
    
    for idx,point in enumerate(adjusted_points):
        # label = label+'_'+data_set[idx]
        # Plot to the left for values less than the ground truth
        # ax.plot((0.0,point),(y_pos+idx*0.1, y_pos+idx*0.1), color=colors[idx], linewidth=3, label=data_set[idx] if set_idx == 0 else "")
        ax.barh(-(idx+label_pos + pos), point, color=colors[idx],height = 1, label=data_set[idx] if set_idx ==0 else "", zorder=3)
    
        # if set_idx == 0:
        #     ax.legend(data_set[idx])
        # Place parameter label next to the axis
    # ax.text(-0.04, y_pos-0.1, label, verticalalignment='center',color='black', transform=ax.get_yaxis_transform(), ha='left',fontsize=20)
    ax.text(-0.09, -(5*label_pos-3), label, verticalalignment='center',color='black', transform=ax.get_yaxis_transform(), ha='left',fontsize=fontsize)
    pos = pos + 4

max_val = range(-1, -46)



# Customize the plot
# ax.set_title('Pacejka Coefficients Difference: {}% Trainset'.format(training_percent),fontsize=fontsize)
# ax.set_ylabel('Data Points')
ax.set_xlabel('Absolute Difference from Ground Truth',fontsize=fontsize)
ax.set_xlim(0,1)
# plt.yticks(y_positions, coeffs)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, zorder=0)
ax.legend(fontsize=legend_fontsize,borderpad=0.1,labelspacing=0.2, handlelength=1.4, handletextpad=0.37,loc='lower right')

plt.yticks(max_val, [''] * len(max_val))
plt.tick_params(axis='x',which='major', labelsize=tick_label_fontsize)
plt.tick_params(axis='x',which='minor', labelsize=tick_label_fontsize)

# ax.get_yaxis().set_visible(False)  # Hide the y-axis as it's not used

# plt.tight_layout()
# plt.legend(loc='lower right',fontsize=legend_fontsize)
plt.tight_layout()
plt.subplots_adjust(left=0.1)

# ax.set_xlim(right=20)
# Show plot
if not os.path.exists(os.path.join(dir_name,"output")):
    os.mkdir(os.path.join(dir_name,"output"))
    
saved_file = 'output/simulated_data_analysis_{}.svg'.format(training_percent)
plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='svg')

saved_file = 'output/simulated_data_analysis_{}.pdf'.format(training_percent)
plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='pdf')

saved_file = 'output/simulated_data_analysis_{}.png'.format(training_percent)
plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='png')
plt.show()

