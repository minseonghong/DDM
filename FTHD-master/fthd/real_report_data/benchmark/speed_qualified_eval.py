import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torch
import matplotlib.ticker as mticker
import random
from fthd.model.models_supervised_ekf import string_to_model, string_to_dataset
from fthd.train.fthd_ekf import train

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

weights = torch.tensor([0.01, 2000, 1.0]).to(device)

def convert_list_to_array_correct(list_of_arrays):
    final_array = np.vstack(list_of_arrays)
    return final_array

# Directory paths
dir_name = os.path.dirname(__file__)
data_file = os.path.join(dir_name, "/home/hong/catkin_ws/src/FTHD-master/fthd/data/SIMULATION_DATA/simulation_double_lane_change_eval_19RNN_Val.npz")
model_folder = os.path.join(dir_name, "/home/hong/catkin_ws/src/FTHD-master/fthd/output/fthd_iac_ekf/supervised_test/2024-10-29_21_53_55")
output_file_path = '/home/hong/catkin_ws/src/FTHD-master/fthd/real_report_data/benchmark/output/Best_Vehicle_Parameters.txt'

# Plot settings
fontsize = 15
tick_label_fontsize = 15
legend_fontsize = 12
plot_linewidth = 2
plot_original = True

color = {
    "5": "green", "15": "red", "30": "black", "60": "blue", "90": "brown"
}
linestyle = {
    "5": ":", "15": "--", "30": "--", "60": "-.", "90": ":"
}

# Patterns for extracting model parameters
patterns = {
    'layers': 5, 'neurons': 153, 'batch': 64, 'lr': 0.000657,
    'horizon': 19, 'gru': 2, 'p': 60
}

# Main plot
fig, axs = plt.subplots(3, 1, figsize=(16, 18))  # 3x1 grid for original plots

# Variable for mass values
mass_values = []

for j, filename in enumerate(os.listdir(model_folder)):
    if not filename.endswith('finetuned_model.pth'):
        continue
    print(j)
    filepath = os.path.join(model_folder, filename)
    print(f"Loaded file: {filename}")
    
    # Load configuration
    model_cfg_path = "/home/hong/catkin_ws/src/FTHD-master/fthd/cfgs/fthd_iac_ekf.yaml"
    with open(model_cfg_path, 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Modify model layers based on extracted values
    param_dict["MODEL"]["LAYERS"] = []
    layer = {"GRU": None, "OUT_FEATURES": int(patterns['horizon']) ** 2, "LAYERS": patterns['gru']}
    param_dict["MODEL"]["LAYERS"].append(layer)
    for i in range(patterns['layers']):
        param_dict["MODEL"]["LAYERS"].append({
            "DENSE": None, "OUT_FEATURES": patterns['neurons'], "ACTIVATION": "Mish"
        })
    param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = patterns['batch']
    param_dict["MODEL"]["OPTIMIZATION"]["LR"] = patterns['lr']
    param_dict["MODEL"]["HORIZON"] = patterns['horizon']
    
    # Load model and data
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)
    model.cuda()
    model.eval()
    
    data_npy = np.load(data_file)
    dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](
        data_npy["features"], data_npy["labels"], data_npy["times_features"], data_npy["times"]
    )
    val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=patterns['batch'], shuffle=False, drop_last=True)
    val_h = model.init_hidden(model.batch_size) if model.is_rnn else None

    # Collect prediction and label data
    s_qualify, s_label, time = [], [], []
    for features, labels, time_feature, times, norm_inputs in val_data_loader:
        features, labels, times = features.to(device), labels.to(device), times.to(device)
        time_feature = time_feature.to(device)
        norm_inputs = norm_inputs.to(device)
        
        if model.is_rnn:
            val_h = val_h.data
            x_upd, o, noise, h0_, d_o , mass, sys_param_dict  = model(features, norm_inputs, labels, val_h, time_feature, times)
        else:
            x_upd, o, noise, h0_, d_o , mass, sys_param_dict = model(features, norm_inputs, labels, None, time_feature, times)

        s_qualify.append(o.detach().cpu().numpy())
        s_label.append(labels.detach().cpu().numpy())
        time.append(time_feature.detach().cpu().numpy())
        mass_values.append(mass.detach().cpu().numpy())

        if j == 2:
            with open(output_file_path, 'w') as file:
                file.write("Best Vehicle Parameters\n")
                for key, value in sys_param_dict.items():
                    file.write(f"{key}: {value[0].item()}\n")
                print(f"Parameters saved to {output_file_path}")
                
    s_qualify, s_label, time_qualify = map(convert_list_to_array_correct, [s_qualify, s_label, time])

    # Extract data for plotting
    s_vx_qualify, s_vy_qualify, s_yaw_qualify = s_qualify[:, 0], s_qualify[:, 1], s_qualify[:, 2]
    s_vx_label, s_vy_label, s_yaw_label = s_label[:, 0], s_label[:, 1], s_label[:, 2]

    # Calculate errors and zoom bounds for each parameter
    error_vx_max = np.max(np.abs(s_vx_label - s_vx_qualify))
    error_vx_idx = np.argmax(np.abs(s_vx_label - s_vx_qualify))
    zoom_vx_time = time_qualify[error_vx_idx]
    zoom_vx_lower = s_vx_qualify[error_vx_idx]
    zoom_vx_upper = s_vx_label[error_vx_idx]

    error_vy_max = np.max(np.abs(s_vy_label - s_vy_qualify))
    error_vy_idx = np.argmax(np.abs(s_vy_label - s_vy_qualify))
    zoom_vy_time = time_qualify[error_vy_idx]
    zoom_vy_lower = s_vy_qualify[error_vy_idx]
    zoom_vy_upper = s_vy_label[error_vy_idx]

    error_yaw_max = np.max(np.abs(s_yaw_label - s_yaw_qualify))
    error_yaw_idx = np.argmax(np.abs(s_yaw_label - s_yaw_qualify))
    zoom_yaw_time = time_qualify[error_yaw_idx]
    zoom_yaw_lower = s_yaw_qualify[error_yaw_idx]
    zoom_yaw_upper = s_yaw_label[error_yaw_idx]

    print("error vx max: {}, position: {}, time: {}".format(error_vx_max, error_vx_idx, zoom_vx_time))
    print("error vy max: {}, position: {}, time: {}".format(error_vy_max, error_vy_idx, zoom_vy_time))
    print("error yaw max: {}, position: {}, time: {}".format(error_yaw_max, error_yaw_idx, zoom_yaw_time))
    
    # Plot original for Vx, Vy, and YawRate
    axs[0].plot(time_qualify, s_vx_label, color='orange', linestyle='-', label='GT', linewidth=plot_linewidth)
    axs[1].plot(time_qualify, s_vy_label, color='orange', linestyle='-', label='GT', linewidth=plot_linewidth)
    axs[2].plot(time_qualify, s_yaw_label, color='orange', linestyle='-', label='GT', linewidth=plot_linewidth)

    axs[0].plot(time_qualify, s_vx_qualify, color=color[str(patterns['p'])], linestyle=linestyle[str(patterns['p'])], label='Proposed_Model', linewidth=plot_linewidth)
    axs[1].plot(time_qualify, s_vy_qualify, color=color[str(patterns['p'])], linestyle=linestyle[str(patterns['p'])], label='Proposed_Model', linewidth=plot_linewidth)
    axs[2].plot(time_qualify, s_yaw_qualify, color=color[str(patterns['p'])], linestyle=linestyle[str(patterns['p'])], label='Proposed_Model', linewidth=plot_linewidth)

# Label original plots
for i, ylabel in enumerate(['Vx (m/s)', 'Vy (m/s)', 'YawRate (rad/s)']):
    axs[i].set_ylabel(ylabel, fontsize=fontsize)
    axs[i].set_xlabel("T (s)", fontsize=fontsize)
    axs[i].tick_params(axis='both', labelsize=tick_label_fontsize)
    axs[i].legend(fontsize=legend_fontsize)
    axs[i].set_xlim(time_qualify[0], time_qualify[-1])


original_pdf_save_file = os.path.join(dir_name, 'output/state_comparison.pdf')
plt.savefig(original_pdf_save_file, transparent=True, format='pdf')
plt.show()

# Save the zoomed plots as separate figures and PDF files
zoomed_plots = [('Vx', zoom_vx_time, zoom_vx_lower, zoom_vx_upper, s_vx_label, s_vx_qualify),
                ('Vy', zoom_vy_time, zoom_vy_lower, zoom_vy_upper, s_vy_label, s_vy_qualify),
                ('YawRate', zoom_yaw_time, zoom_yaw_lower, zoom_yaw_upper, s_yaw_label, s_yaw_qualify)]



for param, zoom_time, zoom_lower, zoom_upper, label_data, qualify_data in zoomed_plots:
    fig_zoom, ax_zoom = plt.subplots(figsize=(8, 6))
    ax_zoom.plot(time_qualify, label_data, color='orange', linestyle='-', label='GT', linewidth=plot_linewidth)
    ax_zoom.plot(time_qualify, qualify_data, color=color[str(patterns['p'])], linestyle=linestyle[str(patterns['p'])], label='Proposed_Model', linewidth=plot_linewidth)
    
    
    ax_zoom.axvline(x=zoom_time, color='red', linestyle='--', linewidth=2, alpha=0.4)
    ax_zoom.set_xlim(zoom_time - 0.3, zoom_time + 0.3)
    ax_zoom.set_ylim(zoom_lower - 0.3, zoom_upper + 0.3)
    ax_zoom.set_ylabel(f'{param} (m/s)', fontsize=fontsize)
    ax_zoom.legend(fontsize=legend_fontsize)
    ax_zoom.tick_params(axis='both', labelsize=tick_label_fontsize)
    plt.xlabel("T (s)", fontsize=fontsize)
    plt.tight_layout()
    pdf_save_file = f'output/zoomed_{param}_morai_mass_hong_double.pdf'
    plt.savefig(os.path.join(dir_name, pdf_save_file), transparent=True, format='pdf')
    plt.show()

plt.figure(figsize=(24, 20))
mass_values = np.concatenate(mass_values, axis=0)
plt.plot(time_qualify, mass_values, color='blue', linewidth=plot_linewidth)
plt.xlabel("T (s)", fontsize=fontsize)
plt.ylabel("Mass (kg)", fontsize=fontsize)
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
# plt.legend(fontsize=legend_fontsize)
plt.xlim(time_qualify[0], time_qualify[-1])

mass_saved_file = 'output/estimated_mass_plot.svg'
mass_pdf_save_file = 'output/estimated_mass_plot.pdf'
plt.savefig(os.path.join(dir_name, mass_saved_file), transparent=True, format='svg')
plt.savefig(os.path.join(dir_name, mass_pdf_save_file), transparent=True, format='pdf')
plt.show()