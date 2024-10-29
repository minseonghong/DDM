#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import csv
from tqdm import tqdm
import os

MAX_BRAKE_PRESSURE = 2757.89990234

def write_dataset(npz_path, csv_path, horizon, save=False, plot=True):
    global TOTAL_TIME
    
    dir_name = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(dir_name,"output")):
        os.mkdir(os.path.join(dir_name,"output"))
        
    denoised_data = np.load(npz_path)  # npz 파일에서 데이터 로드
    denoised_data = denoised_data["features"]
    print("denoised_data size:", denoised_data.shape)  # vx, vy, omega, throttle, steering, throttle_cmd, steering_cmd, time
 
    de_vx = denoised_data[:, 0]
    de_vy = denoised_data[:, 1]
    de_omega = denoised_data[:, 2]
    de_time = denoised_data[:, -1]
    
    with open(csv_path, encoding='ISO-8859-1') as f:
        csv_reader = csv.reader(f, delimiter=",")
        odometry = []
        throttle_cmds = []
        steering_cmds = []
        poses = []
        column_idxs = dict()
        previous_throttle = 0.0
        previous_steer = 0.0
        started = False
        for row in csv_reader:
            if len(column_idxs) == 0:
                for i in range(len(row)):
                    column_idxs[row[i].split("(")[0]] = i
                continue

            vx = float(row[column_idxs["vx"]])
            if abs(vx) < 5:
                if started:
                    break
                brake = float(row[column_idxs["brake_ped_cmd"]])
                throttle = float(row[column_idxs["throttle_ped_cmd"]])
                if brake > 0.0:
                    previous_throttle = -brake / MAX_BRAKE_PRESSURE
                else:
                    previous_throttle = throttle / 100.0
                previous_steer = float(row[column_idxs["delta"]])
                TOTAL_TIME += SAMPLING_TIME
                continue
            vy = float(row[column_idxs["vy"]])
            vtheta = float(row[column_idxs["omega"]])
            steering = float(row[column_idxs["delta"]])
            brake = float(row[column_idxs["brake_ped_cmd"]])
            if brake > 0.0:
                throttle = -brake / MAX_BRAKE_PRESSURE
            else:
                throttle = float(row[column_idxs["throttle_ped_cmd"]]) / 100.0
            steering_cmd = steering - previous_steer
            throttle_cmd = throttle - previous_throttle
            odometry.append(np.array([vx, vy, vtheta, throttle, steering]))
            poses.append([float(row[column_idxs["x"]]), float(row[column_idxs["y"]]), float(row[column_idxs["phi"]]), 
                          vx, vy, vtheta, throttle, steering, throttle_cmd, steering_cmd, 
                          TOTAL_TIME])
            TOTAL_TIME += SAMPLING_TIME
            previous_throttle += throttle_cmd
            previous_steer += steering_cmd
            if started:
                throttle_cmds.append(throttle_cmd)
                steering_cmds.append(steering_cmd)
            started = True
        odometry = np.array(odometry)
        poses_arr = np.array(poses)
        time = poses_arr[:, -1]
        vx_arr = poses_arr[:, 3]
        vy_arr = poses_arr[:, 4]
        vphi_arr = poses_arr[:, 5]
        throttle_arr = poses_arr[:, 6]
        steering_arr = poses_arr[:, 7]
        throttle_cmds = np.array(throttle_cmds)
        steering_cmds = np.array(steering_cmds)
        
        smoothed_vx = savgol_filter(vx_arr, window_length=19, polyorder=2)
        smoothed_vy = savgol_filter(vy_arr, window_length=113, polyorder=2)
        smoothed_vphi = savgol_filter(vphi_arr, window_length=113, polyorder=4)
        smoothed_steering = savgol_filter(steering_arr, window_length=27, polyorder=2)
        smoothed_throttle = savgol_filter(throttle_arr, window_length=113, polyorder=2)
        
        steering_cmds = smoothed_steering[1:] - smoothed_steering[:-1]
        throttle_cmds = smoothed_throttle[1:] - smoothed_throttle[:-1]
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        color = {
            "Original":"black",
            "Smooth":"red",
            "Denoised":"blue"
        }
        fontsize = 26
        legend_fontsize = 22
        tick_label_fontsize = 26
        plot_linewidth = 3
        print("time shape is:", time.shape)
        print("vx smoothed max:", np.argmax(smoothed_vx))
        print("vx denoised max:", np.argmax(de_vx))
        
        if plot:
            axs[0].plot(time, vx_arr, color=color["Original"], linestyle='-', label=r'$X_t$', linewidth=plot_linewidth)
            axs[0].plot(time, smoothed_vx, color=color["Smooth"], linestyle='-.', label=r'$X_s$', linewidth=plot_linewidth)
            axs[0].plot(de_time, de_vx, color=color["Denoised"], linestyle=':', label=r'$X_{EKF_t}$', linewidth=plot_linewidth)

            axs[1].plot(time, vy_arr, color=color["Original"], linestyle='-', label=r'$X_t$', linewidth=plot_linewidth)
            axs[1].plot(time, smoothed_vy, color=color["Smooth"], linestyle='-.', label=r'$X_s$', linewidth=plot_linewidth)
            axs[1].plot(de_time, de_vy, color=color["Denoised"], linestyle=':', label=r'$X_{EKF_t}$', linewidth=plot_linewidth)
            
            axs[2].plot(time, vphi_arr, color=color["Original"], linestyle='-', label=r'$X_t$', linewidth=plot_linewidth)
            axs[2].plot(time, smoothed_vphi, color=color["Smooth"], linestyle='-.', label=r'$X_s$', linewidth=plot_linewidth)
            axs[2].plot(de_time, de_omega, color=color["Denoised"], linestyle=':', label=r'$X_{EKF_t}$', linewidth=plot_linewidth)
            
            axs[0].set_ylabel('$V_{x}$ (m/s)', fontsize=fontsize)
            axs[1].set_ylabel('$V_{y}$ (m/s)', fontsize=fontsize)
            axs[2].set_ylabel(r'$\omega$ (rad/s)', fontsize=fontsize)
            
            axs[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
            axs[0].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

            axs[1].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
            axs[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
            
            axs[2].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
            axs[2].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
            
            for i in range(3):
                axs[i].set_xlim(15, 476)
                axs[i].set_xlabel("t (s)", fontsize=fontsize)
                axs[i].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
                axs[i].legend(fontsize=legend_fontsize, borderpad=0.1, labelspacing=0.2, handlelength=1.4, handletextpad=0.37, loc='upper left')
            
            plt.tight_layout()
            saved_file = 'output/smooth_data_compare.svg'
            plt.savefig(os.path.join(dir_name, saved_file), transparent=True, format='svg')

            saved_file = 'output/smooth_data_compare.pdf'
            plt.savefig(os.path.join(dir_name, saved_file), transparent=True, format='pdf')

            plt.show()


if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Convert NPZ and CSV files to dataset")
    parser.add_argument("npz_path", type=str, help="NPZ file to convert")
    parser.add_argument("csv_path", type=str, help="CSV file to process")
    parser.add_argument("horizon", type=int, help="Horizon of timestamps used")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict = vars(args)
    SAMPLING_TIME = 0.04
    TOTAL_TIME = 0.00
    write_dataset(argdict["npz_path"], argdict["csv_path"], argdict["horizon"])
