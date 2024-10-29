import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

dir_name = os.path.dirname(__file__)

data_file = os.path.join(dir_name,"data/DYN-PP-ETHZ_1RNN_val.npz")
training_percent = "80"
fontsize = 26
tick_label_fontsize = 26
legend_fontsize = 22

plot_linewidth = 3

original_config_name = "cfg_fullset{}/Pinn_coeff_original.yaml".format(training_percent)
Hybrid_config_name = "cfg_fullset{}/Pinn_coeff_HybridAdam.yaml".format(training_percent)
DDP_config_name = "cfg_fullset{}/Pinn_coeff_HybridDDP.yaml".format(training_percent)


GT_config_file = os.path.join(dir_name,"cfg/deep_dynamics_ddp_GT.yaml")
DDM_minVal_config_file = os.path.join(dir_name,original_config_name)
Hybrid_method_config_file = os.path.join(dir_name,Hybrid_config_name)
HybridDDP_method_config_file = os.path.join(dir_name,DDP_config_name)





# trained_minVal_config_file = os.path.join(dir_name,"cfg_fullset08/Pinn_coeff_HybridAdam.yaml")




if not os.path.exists(os.path.join(dir_name,"output")):
    os.mkdir(os.path.join(dir_name,"output"))

data_npy = np.load(data_file)

features = data_npy["features"] # vx, vy, yawRate, throttle_fb, steering_fb, throttle_cmd, steering_cmd
features = features.squeeze(1)
print("feature shape:",features.shape)

# pose_features = data_npy["pose_features"] # X, Y, Phi, time
# print("pose feature shape:",pose_features.shape)

coeffs = ['Bf','Cf','Df','Ef','Br','Cr','Dr','Er','Cm1','Cm2','Cr0','Cr2','Iz','Shf','Svf','Shr','Svr']
GT_coeffs = ['Bf','Cf','Df','Ef','Br','Cr','Dr','Er','Cm1','Cm2','Cr0','Cr2','Iz','Shf','Svf','Shr','Svr']

lf = 0.029
lr = 0.033
mass = 0.041

with open(DDM_minVal_config_file, 'rb') as f:
    DDM_param_dict = yaml.load(f, Loader=yaml.SafeLoader)

with open(GT_config_file, 'rb') as f:
    GT_param_dict = yaml.load(f, Loader=yaml.SafeLoader)

with open(Hybrid_method_config_file, 'rb') as f:
    Hybrid_param_dict = yaml.load(f, Loader=yaml.SafeLoader)

with open(HybridDDP_method_config_file, 'rb') as f:
    HybridDDP_param_dict = yaml.load(f, Loader=yaml.SafeLoader)

DDMCoef_vale = []
GTCoef_vale = []
Hybrid_vale = []
DDP_vale = []

for i in range(len(DDM_param_dict["PARAMETERS"])):
    DDMCoef_vale.append(list(DDM_param_dict["PARAMETERS"][i].values())[0])

for i in range(len(GT_param_dict["PARAMETERS"])):
    GTCoef_vale.append(list(GT_param_dict["PARAMETERS"][i].values())[0])

for i in range(len(Hybrid_param_dict["PARAMETERS"])):
    Hybrid_vale.append(list(Hybrid_param_dict["PARAMETERS"][i].values())[0])

for i in range(len(HybridDDP_param_dict["PARAMETERS"])):
    DDP_vale.append(list(HybridDDP_param_dict["PARAMETERS"][i].values())[0])

# Xstart, Ystart, Phistart = pose_features[0,0], pose_features[0,1], pose_features[0,2]

# Vx, Vy, YawRate, Throttle_fb, Steering_fb, Throttle_cmd, Steering_cmd = features[0,0],features[0,1],features[0,2],features[0,3],features[0,4],features[0,5],features[0,6]

DDMpara_dict = dict()
for i in range(len(DDM_param_dict["PARAMETERS"])):
    DDMpara_dict[coeffs[i]] = DDMCoef_vale[i]

GTpara_dict = dict()
for i in range(len(DDM_param_dict["PARAMETERS"])):
    GTpara_dict[coeffs[i]] = GTCoef_vale[i]

Hybridpara_dict = dict()
for i in range(len(DDM_param_dict["PARAMETERS"])):
    Hybridpara_dict[coeffs[i]] = Hybrid_vale[i]

HybridDDPpara_dict = dict()
for i in range(len(DDM_param_dict["PARAMETERS"])):
    HybridDDPpara_dict[coeffs[i]] = DDP_vale[i]

def DDMforce_equations(vx, vy, yawRate, throttle_fb, steering_fb, throttle_cmd, steering_cmd):
    steering = steering_fb + steering_cmd
    throttle = throttle_fb + throttle_cmd
    alphaf = steering - np.arctan2(lf*yawRate + vy, np.abs(vx)) + DDMpara_dict["Shf"]
    alphar = np.arctan2((lr*yawRate - vy), np.abs(vx)) + DDMpara_dict["Shr"]
    Frx = (DDMpara_dict["Cm1"]-DDMpara_dict["Cm2"]*vx)*throttle - DDMpara_dict["Cr0"] - DDMpara_dict["Cr2"]*(vx**2)
    Ffy = DDMpara_dict["Svf"] + DDMpara_dict["Df"] * np.sin(DDMpara_dict["Cf"] * 
                                                      np.arctan(DDMpara_dict["Bf"] * alphaf - DDMpara_dict["Ef"] * 
                                                                 (DDMpara_dict["Bf"] * alphaf - np.arctan(DDMpara_dict["Bf"] * alphaf))))
    Fry = DDMpara_dict["Svr"] + DDMpara_dict["Dr"] * np.sin(DDMpara_dict["Cr"] * 
                                                      np.arctan(DDMpara_dict["Br"] * alphar - DDMpara_dict["Er"] * 
                                                                (DDMpara_dict["Br"] * alphar - np.arctan(DDMpara_dict["Br"] * alphar))))
    # # dxdt = np.zeros(1, 3)
    # vx1 = vx + (1/mass * (Frx - Ffy*np.sin(steering)) + vy*yawRate) * Ts
    # vy1 = vy + (1/mass * (Fry + Ffy*np.cos(steering)) - vx*yawRate) * Ts
    # yaw1 = yawRate + (1/para_dict["Iz"] * (Ffy*lf*np.cos(steering) - Fry*lr)) * Ts
    
    return tuple((alphaf, Ffy, alphar, Frx, Fry))

def GTforce_equations(vx, vy, yawRate, throttle_fb, steering_fb, throttle_cmd, steering_cmd):
    steering = steering_fb + steering_cmd
    throttle = throttle_fb + throttle_cmd
    alphaf = steering - np.arctan2(lf*yawRate + vy, np.abs(vx)) + GTpara_dict["Shf"]
    alphar = np.arctan2((lr*yawRate - vy), np.abs(vx)) + GTpara_dict["Shr"]
    Frx = (GTpara_dict["Cm1"]-GTpara_dict["Cm2"]*vx)*throttle - GTpara_dict["Cr0"] - GTpara_dict["Cr2"]*(vx**2)
    Ffy = GTpara_dict["Svf"] + GTpara_dict["Df"] * np.sin(GTpara_dict["Cf"] * 
                                                      np.arctan(GTpara_dict["Bf"] * alphaf - GTpara_dict["Ef"] * 
                                                                 (GTpara_dict["Bf"] * alphaf - np.arctan(GTpara_dict["Bf"] * alphaf))))
    Fry = GTpara_dict["Svr"] + GTpara_dict["Dr"] * np.sin(GTpara_dict["Cr"] * 
                                                      np.arctan(GTpara_dict["Br"] * alphar - GTpara_dict["Er"] * 
                                                                (GTpara_dict["Br"] * alphar - np.arctan(GTpara_dict["Br"] * alphar))))
    # # dxdt = np.zeros(1, 3)
    # vx1 = vx + (1/mass * (Frx - Ffy*np.sin(steering)) + vy*yawRate) * Ts
    # vy1 = vy + (1/mass * (Fry + Ffy*np.cos(steering)) - vx*yawRate) * Ts
    # yaw1 = yawRate + (1/para_dict["Iz"] * (Ffy*lf*np.cos(steering) - Fry*lr)) * Ts
    
    return tuple((alphaf, Ffy, alphar, Frx, Fry))

def DDPforce_equations(vx, vy, yawRate, throttle_fb, steering_fb, throttle_cmd, steering_cmd):
    steering = steering_fb + steering_cmd
    throttle = throttle_fb + throttle_cmd
    alphaf = steering - np.arctan2(lf*yawRate + vy, np.abs(vx)) + HybridDDPpara_dict["Shf"]
    alphar = np.arctan2((lr*yawRate - vy), np.abs(vx)) + HybridDDPpara_dict["Shr"]
    Frx = (HybridDDPpara_dict["Cm1"]-HybridDDPpara_dict["Cm2"]*vx)*throttle - HybridDDPpara_dict["Cr0"] - HybridDDPpara_dict["Cr2"]*(vx**2)
    Ffy = HybridDDPpara_dict["Svf"] + HybridDDPpara_dict["Df"] * np.sin(HybridDDPpara_dict["Cf"] * 
                                                      np.arctan(HybridDDPpara_dict["Bf"] * alphaf - HybridDDPpara_dict["Ef"] * 
                                                                 (HybridDDPpara_dict["Bf"] * alphaf - np.arctan(GTpara_dict["Bf"] * alphaf))))
    Fry = HybridDDPpara_dict["Svr"] + HybridDDPpara_dict["Dr"] * np.sin(HybridDDPpara_dict["Cr"] * 
                                                      np.arctan(HybridDDPpara_dict["Br"] * alphar - HybridDDPpara_dict["Er"] * 
                                                                (HybridDDPpara_dict["Br"] * alphar - np.arctan(HybridDDPpara_dict["Br"] * alphar))))
    # # dxdt = np.zeros(1, 3)
    # vx1 = vx + (1/mass * (Frx - Ffy*np.sin(steering)) + vy*yawRate) * Ts
    # vy1 = vy + (1/mass * (Fry + Ffy*np.cos(steering)) - vx*yawRate) * Ts
    # yaw1 = yawRate + (1/para_dict["Iz"] * (Ffy*lf*np.cos(steering) - Fry*lr)) * Ts
    
    return tuple((alphaf, Ffy, alphar, Frx, Fry))

def Hybridforce_equations(vx, vy, yawRate, throttle_fb, steering_fb, throttle_cmd, steering_cmd):
    steering = steering_fb + steering_cmd
    throttle = throttle_fb + throttle_cmd
    alphaf = steering - np.arctan2(lf*yawRate + vy, np.abs(vx)) + Hybridpara_dict["Shf"]
    alphar = np.arctan2((lr*yawRate - vy), np.abs(vx)) + Hybridpara_dict["Shr"]
    Frx = (Hybridpara_dict["Cm1"]-Hybridpara_dict["Cm2"]*vx)*throttle - Hybridpara_dict["Cr0"] - Hybridpara_dict["Cr2"]*(vx**2)
    Ffy = Hybridpara_dict["Svf"] + Hybridpara_dict["Df"] * np.sin(Hybridpara_dict["Cf"] * 
                                                      np.arctan(Hybridpara_dict["Bf"] * alphaf - Hybridpara_dict["Ef"] * 
                                                                 (Hybridpara_dict["Bf"] * alphaf - np.arctan(Hybridpara_dict["Bf"] * alphaf))))
    Fry = Hybridpara_dict["Svr"] + Hybridpara_dict["Dr"] * np.sin(Hybridpara_dict["Cr"] * 
                                                      np.arctan(Hybridpara_dict["Br"] * alphar - Hybridpara_dict["Er"] * 
                                                                (Hybridpara_dict["Br"] * alphar - np.arctan(Hybridpara_dict["Br"] * alphar))))
    # # dxdt = np.zeros(1, 3)
    # vx1 = vx + (1/mass * (Frx - Ffy*np.sin(steering)) + vy*yawRate) * Ts
    # vy1 = vy + (1/mass * (Fry + Ffy*np.cos(steering)) - vx*yawRate) * Ts
    # yaw1 = yawRate + (1/para_dict["Iz"] * (Ffy*lf*np.cos(steering) - Fry*lr)) * Ts
    
    return tuple((alphaf, Ffy, alphar, Frx, Fry))


DDMforce_state = np.zeros((features.shape[0],5))

DDMfront_force = np.zeros((features.shape[0],2))
DDMrear_force = np.zeros((features.shape[0],3))

GTforce_state = np.zeros((features.shape[0],5))

GTfront_force = np.zeros((features.shape[0],2))
GTrear_force = np.zeros((features.shape[0],3))

Hybridforce_state = np.zeros((features.shape[0],5))

Hybridfront_force = np.zeros((features.shape[0],2))
Hybridrear_force = np.zeros((features.shape[0],3))

DDPforce_state = np.zeros((features.shape[0],5))

DDPfront_force = np.zeros((features.shape[0],2))
DDPrear_force = np.zeros((features.shape[0],3))


for i in range(features.shape[0]-1):
    Vx, Vy, YawRate, Throttle_fb = features[i,0],features[i,1],features[i,2],features[i,3]
    Steering_fb, Throttle_cmd, Steering_cmd = features[i,4], features[i,5], features[i,6]

    DDMforce_state[i,:] = DDMforce_equations(Vx, Vy, YawRate, Throttle_fb, Steering_fb, Throttle_cmd, Steering_cmd)
    GTforce_state[i,:] = GTforce_equations(Vx, Vy, YawRate, Throttle_fb, Steering_fb, Throttle_cmd, Steering_cmd)
    Hybridforce_state[i,:] = Hybridforce_equations(Vx, Vy, YawRate, Throttle_fb, Steering_fb, Throttle_cmd, Steering_cmd)
    DDPforce_state[i,:] = DDPforce_equations(Vx, Vy, YawRate, Throttle_fb, Steering_fb, Throttle_cmd, Steering_cmd)


    DDMfront_force[i,:2] = DDMforce_state[i,:2]
    DDMrear_force[i,:3] = DDMforce_state[i,2:]

    GTfront_force[i,:2] = GTforce_state[i,:2]
    GTrear_force[i,:3] = GTforce_state[i,2:]

    Hybridfront_force[i,:2] = Hybridforce_state[i,:2]
    Hybridrear_force[i,:3] = Hybridforce_state[i,2:]

    DDPfront_force[i,:2] = DDPforce_state[i,:2]
    DDPrear_force[i,:3] = DDPforce_state[i,2:]
    

DDMfront_force = DDMfront_force[np.argsort(DDMfront_force[:,0])]
DDMrear_force = DDMrear_force[np.argsort(DDMrear_force[:,0])]

GTfront_force = GTfront_force[np.argsort(GTfront_force[:,0])]
GTrear_force = GTrear_force[np.argsort(GTrear_force[:,0])]

Hybridfront_force = Hybridfront_force[np.argsort(Hybridfront_force[:,0])]
Hybridrear_force = Hybridrear_force[np.argsort(Hybridrear_force[:,0])]

DDPfront_force = DDPfront_force[np.argsort(DDPfront_force[:,0])]
DDPrear_force = DDPrear_force[np.argsort(DDPrear_force[:,0])]



fig, axs = plt.subplots(2, 1, figsize=(10, 7.5))
axs[0].plot(GTfront_force[:,0],GTfront_force[:,1],color='red',label='GT $F_{fy}$', linewidth=plot_linewidth)
axs[0].plot(Hybridfront_force[:,0],Hybridfront_force[:,1],color='blue',linestyle='--',label='FTHD $F_{fy}$', linewidth=plot_linewidth)
axs[0].plot(DDMfront_force[:,0],DDMfront_force[:,1],color='black',linestyle='-.',label='DDM $F_{fy}$', linewidth=plot_linewidth)
# axs[0].plot(DDPfront_force[:,0],DDPfront_force[:,1],color='brown',linestyle=':',label='Hybrid dualdimer Ffy', linewidth=plot_linewidth)

axs[0].set_xlim(-0.505,0.64)
# axs[0].set_ylim(0.1875,0.225)
# axs[0].set_title('$F_{{fy}}$ Force with {}% Trainset'.format(training_percent),fontsize=fontsize-3)
# axs[0].legend(loc='upper left')


axs[1].plot(GTrear_force[:,0],GTrear_force[:,2],color='red',label='GT $F_{ry}$', linewidth=plot_linewidth)
axs[1].plot(Hybridrear_force[:,0],Hybridrear_force[:,2],color='blue',linestyle='--',label='FTHD $F_{ry}$', linewidth=plot_linewidth)
axs[1].plot(DDMrear_force[:,0],DDMrear_force[:,2],color='black',linestyle='-.',label='DDM $F_{ry}$', linewidth=plot_linewidth)
# axs[1].plot(DDPrear_force[:,0],DDPrear_force[:,2],color='brown',linestyle=':',label='Hybrid dualdimer Fry', linewidth=plot_linewidth)

axs[1].set_xlim(-0.228,0.5787)
# axs[1].set_ylim(0.1875,0.225)
# axs[1].set_title('$F_{{ry}}$ Force with {}% Trainset'.format(training_percent),fontsize=fontsize-3)
# axs[1].legend(loc='upper left')

# axs[2].plot(GTfront_force[:,0],GTfront_force[:,1],color='red',label='GT Ffy')
# axs[2].plot(Hybridfront_force[:,0],Hybridfront_force[:,1],color='blue',linestyle='-.',label='Hybrid Adam Ffy')
# axs[2].plot(DDMfront_force[:,0],DDMfront_force[:,1],color='black',linestyle='-.',label='DDM Ffy')
# axs[2].plot(DDPfront_force[:,0],DDPfront_force[:,1],color='brown',linestyle='-.',label='Hybrid dualdimer Fry')

# axs[2].set_xlim(0.55,0.64)
# axs[2].set_ylim(0.190,0.23)
# axs[2].set_title('Zoomed Ffy force with {} percent trainset'.format(training_percent),fontsize=fontsize)
# axs[2].legend(loc='upper left')

# axs[3].plot(GTrear_force[:,0],GTrear_force[:,2],color='red',label='GT Fry')
# axs[3].plot(Hybridrear_force[:,0],Hybridrear_force[:,2],color='blue',linestyle='-.',label='Hybrid Adam Fry')
# axs[3].plot(DDMrear_force[:,0],DDMrear_force[:,2],color='black',linestyle='-.',label='DDM Fry')
# axs[3].plot(DDPrear_force[:,0],DDPrear_force[:,2],color='brown',linestyle='-.',label='Hybrid dualdimer Fry')

# axs[3].set_xlim(0.4,0.5787)
# axs[3].set_ylim(0.12,0.19)
# axs[3].set_title('Zoomed Fry force with {} percent trainset'.format(training_percent),fontsize=fontsize)
# axs[3].legend(loc='lower right')
axs[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axs[0].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

axs[1].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
axs[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

# axs[2].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
# axs[2].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

# axs[3].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
# axs[3].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

for i in range(2):
    axs[i].set_xlabel('Slip Angle (rad)',fontsize=fontsize)
    axs[i].set_ylabel('Force (N)',fontsize=fontsize)
    axs[i].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    axs[i].legend(fontsize=legend_fontsize,borderpad=0.1,labelspacing=0.2, handlelength=1.4, handletextpad=0.37,loc='upper left')
plt.tight_layout()
saved_file = 'output/poseter_trained_force_benchmark_{}.svg'.format(training_percent)
plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='svg')

saved_file = 'output/poseter_trained_force_benchmark_{}.pdf'.format(training_percent)
plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='pdf')

# saved_file = 'output/poseter_trained_force_benchmark_{}.png'.format(training_percent)
# plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='png')

plt.show()