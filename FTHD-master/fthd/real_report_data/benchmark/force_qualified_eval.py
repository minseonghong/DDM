import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import torch
from models_supervised import string_to_dataset, string_to_model
# from train_supervised import train
from train_finetuning import train

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
    

dir_name = os.path.dirname(__file__)

fontsize = 26
tick_label_fontsize = 26
legend_fontsize = 21

plot_linewidth = 3

plot_original = True
color = ["red","black","blue","brown"]
color = {
    "5":"green",
    "15":"red",
    "30":"black",
    "60":"blue",
    "90":"brown"
}

linestyle = {
    "5":"--",
    "15":"--",
    "30":"--",
    "60":"-.",
    "90":":"
}
training_percent = ["90","60","30","15","5"]

# /home/hong/catkin_ws/src/FTHD-master/fthd/output/fthd_iac_ekf/supervised_test/2024-10-15_17_27_11

# denoised_features = denoised_features.squeeze(1)
# print("denoised_feature shape:",denoised_features.shape)
if plot_original:
    data_file = os.path.join(dir_name,"/home/hong/catkin_ws/src/FTHD-master/fthd/data/IAC_EKF_DATA/Putnam_park2023_run4_2_19RNN_hong_Val.npz")

    model_folder = os.path.join(dir_name,"/home/hong/catkin_ws/src/FTHD-master/fthd/output/fthd_iac_ekf/supervised_test/2024-10-15_17_27_11")
# else:
#     data_file = os.path.join(dir_name,"/home/hong/catkin_ws/src/FTHD-master/fthd/output/fthd_iac_ekf/supervised_test/2024-10-15_17_27_11/denoised_csv.npz")
# # else:
#     data_file = os.path.join(dir_name,"data/denoised_csv_1RNN_val.npz")
#     # data_file = os.path.join(dir_name,"data/Putnam_park2023_run4_2_1RNN_Val.npz")

#     model_folder = os.path.join(dir_name,"data/ekf_fthd_model")
model_files = os.listdir(model_folder)

pth_files = [f for f in model_files if f.endswith('.pth')]

patterns = {
    'layers': r'(\d+)layers',
    'neurons': r'(\d+)neurons',
    'batch': r'(\d+)batch',
    'lr': r'(\d+\.\d+)lr',
    'horizon': r'(\d+)horizon',
    'gru': r'(\d+)gru',
    'p': r'(\d+)p'
}

fig, axs = plt.subplots(2, 1, figsize=(10, 7.5))

# Define the function to convert the list of lists of arrays
def convert_list_to_array_correct(list_of_arrays):
    # Transpose the list of lists to get the desired order
    transposed_list = list(map(list, zip(*list_of_arrays)))
    # Concatenate the arrays within each sublist along the first axis
    concatenated_sublist = [np.concatenate(sublist, axis=0) for sublist in transposed_list]
    # Stack the resulting arrays along the first axis to get a single array
    final_array = np.vstack(concatenated_sublist)
    return final_array

for j,filename in enumerate(pth_files):
    filepath = os.path.join(model_folder, filename)
    # Load the file (assuming you are using torch to load .pth files)
    # model = torch.load(filepath) # Uncomment and customize this line based on your specific use case
    print(f"Loaded file: {filename}")
    


    # Extracting the numbers
    extracted_values = {key: re.search(pattern, filename).group(1) for key, pattern in patterns.items()}

    print(extracted_values['p'])

    model_cfg = "/home/hong/catkin_ws/src/FTHD-master/fthd/cfgs/fthd_iac_ekf.yaml"
        

    model_cfg = os.path.join(dir_name,"model_cfg",model_cfg)

    with open(model_cfg, 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
        
    param_dict["MODEL"]["LAYERS"] = []

    layer = dict()
    
    if int(extracted_values['gru']) != 0:
        layer["GRU"] = None
        layer["OUT_FEATURES"] = int(extracted_values['horizon']) ** 2
        layer["LAYERS"] = int(extracted_values['gru'])
        param_dict["MODEL"]["LAYERS"].append(layer)
    for i in range(int(extracted_values['layers'])):
        layer = dict()
        layer["DENSE"] = None
        layer["OUT_FEATURES"] = int(extracted_values['neurons'])
        layer["ACTIVATION"] = "Mish"
        param_dict["MODEL"]["LAYERS"].append(layer)
    param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = int(extracted_values['batch'])
    param_dict["MODEL"]["OPTIMIZATION"]["LR"] = float(extracted_values['lr'])
    param_dict["MODEL"]["HORIZON"] = int(extracted_values['horizon'])
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)
    
 
    if plot_original:
        data_file = os.path.join(dir_name,"/home/hong/catkin_ws/src/FTHD-master/fthd/data/IAC_EKF_DATA/Putnam_park2023_run4_2_{}RNN_hong_Val.npz".format(extracted_values['horizon']))
        
    #     data_file = os.path.join(dir_name,"data/ekf_model_data/denoised_csv_{}RNN_val.npz".format(extracted_values['horizon']))
    print("data file name:",data_file)
    
    data_npy = np.load(data_file)
    # val_data_npy = np.load(val_dataset_file)
    # train_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](train_data_npy["features"], train_data_npy["labels"],
    #                                                                train_data_npy["times"],train_data_npy["pose_features"])
    
    # dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](val_data_npy["features"], val_data_npy["labels"],
    #                                                             val_data_npy["times"],val_data_npy["pose_features"])
    
    # train_dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](train_data_npy["features"], train_data_npy["labels"],
    #                                                                train_data_npy["times"],train_data_npy["pose_features"])
    # torch.seed(0)

    dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"],
                                                            data_npy["times_features"],data_npy["times"])
    
    val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=int(extracted_values['batch']), shuffle=False, drop_last=True)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    if model.is_rnn:
        val_h = model.init_hidden(model.batch_size)
    
    f_qualify = []
    
    
    for val_f, val_la, val_timef, val_times, val_normin in val_data_loader:

        val_f, val_la, val_times = val_f.to(device), val_la.to(device), val_times.to(device)
        val_timef = val_timef.to(device)
        val_normin = val_normin.to(device)
        if model.is_rnn:
            val_h = val_h.data
            out, val_h, _, force_qualify = model(val_f, val_normin, val_h,val_timef, val_times)
        else:
            out, _, _, force_qualify  = model(val_f, val_normin,None, val_timef, val_times)
        
        # arrays = [[tensor.detach().cpu().numpy() for tensor in inner_list] for inner_list in force_qualify]
        # concatenated_arrays = [np.concatenate(inner_list, axis=1) for inner_list in arrays]
        # final_array = np.vstack(concatenated_arrays)
        force_qualify = [ele.detach().cpu().numpy() for ele in force_qualify]
        f_qualify.append(force_qualify)
        
        
            
        
        # alpha_f.append(force_qualify[0])
        # alpha_r.append(force_qualify[1])
        # Ffy.append(force_qualify[2])
        # Fry.append(force_qualify[3])
    # print(f_qualify)
    # concatenated_arrays = [np.concatenate(inner_list, axis=0) for inner_list in f_qualify]
    force_qualify = convert_list_to_array_correct(f_qualify)
    
    # force_qualify = np.vstack(f_qualify)
    force_qualify = force_qualify.transpose()
    
    print(force_qualify.shape)
    
    # alpha_f = force_qualify[:,0]
    # alpha_r = force_qualify[:,1]
    Ffy = force_qualify[:,0:2]
    Fry = force_qualify[:,2:]
    
    # print(Ffy[:,1])
    
#     # a_f = np.argsort(alpha_f)
#     # a_r = np.argsort(alpha_r)
    
#     # alpha_f = alpha_f[a_f]
#     # alpha_r = alpha_f[a_r]
    
    Ffy = Ffy[np.argsort(Ffy[:,0])]
    Fry = Fry[np.argsort(Fry[:,0])]
    
    if plot_original:
        axs[0].plot(Ffy[:,0],Ffy[:,1],color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label=r'DDM $F_{{fy}}$ {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        axs[1].plot(Fry[:,0],Fry[:,1],color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label=r'DDM $F_{{ry}}$ {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        # axs[2].plot(Ffy[:,0],Ffy[:,1],color=color[extracted_values['p']],linestyle='-.',label='DDM Ffy {}%'.format(extracted_values['p']))
        # axs[3].plot(Fry[:,0],Fry[:,1],color=color[extracted_values['p']],linestyle='-.',label='DDM Fry {}%'.format(extracted_values['p']))
     
    else:
        axs[0].plot(Ffy[:,0],Ffy[:,1],color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label=r'FTHD $F_{{fy}}$ {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        axs[1].plot(Fry[:,0],Fry[:,1],color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label=r'FTHD $F_{{ry}}$ {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        # axs[2].plot(Ffy[:,0],Ffy[:,1],color=color[extracted_values['p']],linestyle='-.',label='Hybrid Adam Ffy {}%'.format(extracted_values['p']))
        # axs[3].plot(Fry[:,0],Fry[:,1],color=color[extracted_values['p']],linestyle='-.',label='Hybrid Adam Fry {}%'.format(extracted_values['p']))
    

if not plot_original:
    # axs[0].set_xlim(-0.0958,0.062)
    # axs[1].set_xlim(-0.0526,0.026)
    axs[0].set_xlim(-0.08,0.044)
    axs[1].set_xlim(-0.08338,0.07280)
    
else:
    axs[0].set_xlim(-0.0790,0.0425)
    axs[1].set_xlim(-0.0765,0.0656)
    

# axs[0].set_ylim(0.1875,0.225)
# axs[0].set_title('Ffy force with {} percent trainset'.format(training_percent),fontsize=fontsize)
axs[0].legend(loc='lower right')


# axs[1].set_ylim(0.1875,0.225)
# axs[1].set_title('Fry force with {} percent trainset'.format(training_percent),fontsize=fontsize)
axs[1].legend(loc='lower right')

# axs[2].set_xlim(0.55,0.64)
# axs[2].set_ylim(0.180,0.201)
# axs[2].set_title('Zoomed Ffy force with {} percent trainset'.format(training_percent),fontsize=fontsize)
# axs[2].legend(loc='upper left')

# axs[3].set_xlim(0.4,0.5787)
# axs[3].set_ylim(0.16,0.185)
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
    axs[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    axs[i].legend(fontsize=legend_fontsize, handlelength=1.5, handletextpad=0.375, borderaxespad=0.2)
    
plt.tight_layout()

if plot_original:
    saved_file = 'output/model_force_benchmark_DDM.svg'
    pdf_saved_file = 'output/model_force_benchmark_DDM.pdf'

else:
    saved_file = 'output/model_force_benchmarkEKF_FTHD.svg'
    pdf_saved_file = 'output/model_force_benchmarkEKF_FTHD.pdf'
    
plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='svg')
plt.savefig(os.path.join(dir_name,pdf_saved_file),transparent=True, format='pdf')


plt.show()