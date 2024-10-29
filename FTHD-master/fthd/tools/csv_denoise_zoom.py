#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys
import csv
from tqdm import tqdm
import os

MAX_BRAKE_PRESSURE = 2757.89990234

def write_dataset(csv_path, horizon, save=False, plot=True):
    global TOTAL_TIME
    
    dir_name = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(dir_name,"output")):
        os.mkdir(os.path.join(dir_name,"output"))
        
    denoised_folder = os.path.split(dir_name)[0]
        
    denoised_data_file = os.path.join(denoised_folder,"data","Denoised_IAC_DATA","denoised_csv.npz")
    denoised_data = np.load(denoised_data_file)
    # denoised_time = denoised_data["times"]

    
    denoised_data = denoised_data["features"]
    print("denoised_data size:",denoised_data.shape) # vx,vy,omega,throttle,steering,throttle_cmd,steering_cmd,time
    # print("denoised time size:",denoised_time.shape)

    
    de_vx = denoised_data[:,0]
    de_vy = denoised_data[:,1]
    de_omega = denoised_data[:,2]
    de_time = denoised_data[:,-1]

    with open(csv_path) as f:
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
                throttle =  float(row[column_idxs["throttle_ped_cmd"]]) / 100.0
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
        time = poses_arr[:,-1]
        vx_arr = poses_arr[:,3]
        vy_arr = poses_arr[:,4]
        vphi_arr = poses_arr[:,5]
        throttle_arr = poses_arr[:,6]
        steering_arr = poses_arr[:,7]
        throttle_cmds = np.array(throttle_cmds)
        steering_cmds = np.array(steering_cmds)
        
        smoothed_vx = savgol_filter(vx_arr, window_length=19, polyorder=2)
        smoothed_vy = savgol_filter(vy_arr, window_length=113, polyorder=2)
        smoothed_vphi = savgol_filter(vphi_arr, window_length=113, polyorder=4)
        smoothed_steering = savgol_filter(steering_arr, window_length=27, polyorder=2)
        smoothed_throttle = savgol_filter(throttle_arr, window_length=113, polyorder=2)
        
        steering_cmds = smoothed_steering[1:] - smoothed_steering[:-1]
        throttle_cmds = smoothed_throttle[1:] - smoothed_throttle[:-1]
        
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        color = {
            "Original":"black",
            "Smooth":"red",
            "Denoised":"blue"
        }
        fontsize = 26
        legend_fontsize = 22
        tick_label_fontsize = 26
        plot_linewidth = 3
        if plot:
            axs[0].plot(time,vx_arr,color=color["Original"],linestyle='-',label='Original data', linewidth=plot_linewidth)
            axs[0].plot(time,smoothed_vx,color=color["Smooth"],linestyle='-.',label='Smoothed data', linewidth=plot_linewidth)
            axs[0].plot(de_time,de_vx,color=color["Denoised"],linestyle=':',label = 'Denoised data', linewidth=plot_linewidth)

            axs[1].plot(time,vy_arr,color=color["Original"],linestyle='-',label='Original data', linewidth=plot_linewidth)
            axs[1].plot(time,smoothed_vy,color=color["Smooth"],linestyle='-.',label='Smoothed data', linewidth=plot_linewidth)
            axs[1].plot(de_time,de_vy,color=color["Denoised"],linestyle=':',label = 'Denoised data', linewidth=plot_linewidth)
            
            axs[2].plot(time,vphi_arr,color=color["Original"],linestyle='-',label='Original data', linewidth=plot_linewidth)
            axs[2].plot(time,smoothed_vphi,color=color["Smooth"],linestyle='-.',label='Smoothed data', linewidth=plot_linewidth)
            axs[2].plot(de_time,de_omega,color=color["Denoised"],linestyle=':',label = 'Denoised data', linewidth=plot_linewidth)
            

            axs[0].set_ylabel('$V_{x}$ (m/s)',fontsize=fontsize)
            axs[1].set_ylabel('$V_{y}$ (m/s)',fontsize=fontsize)
            axs[2].set_ylabel(r'$\omega$ (rad/s)',fontsize=fontsize)
            
            axs[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
            axs[0].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

            axs[1].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
            axs[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
            
            axs[2].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
            axs[2].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
            
            for i in range(3):
                axs[i].set_xlim(262.1,262.4)
                axs[i].set_xlabel("t (s)",fontsize=fontsize)
                axs[i].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
                # axs[i].legend(fontsize=legend_fontsize,borderpad=0.1,labelspacing=0.2, handlelength=1.4, handletextpad=0.37,loc='upper left')
            axs[0].set_ylim(26.5,27.1)
            axs[1].set_ylim(0.25,0.31)
            axs[2].set_ylim(-0.011,0.0035)
            
            plt.tight_layout()
            plt.subplots_adjust(left=0.4)
            saved_file = 'output/smooth_data_compare_zoom.svg'
            plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='svg')

            saved_file = 'output/smooth_data_compare_zoom.pdf'
            plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='pdf')

            
            plt.show()
            
            
            
            
        refined_range1 = [int(58/SAMPLING_TIME), int(149.5/SAMPLING_TIME)]
        refined_range2 = [int(156/SAMPLING_TIME), int(192/SAMPLING_TIME)]
        refined_range3 = [int(201/SAMPLING_TIME), int(232.5/SAMPLING_TIME)]
        
        
        poses_arr[:,3] = smoothed_vx
        poses_arr[:,4] = smoothed_vy
        poses_arr[:,5] = smoothed_vphi
        poses_arr[:,6] = smoothed_throttle
        poses_arr[:,7] = smoothed_steering
        poses_arr[1:,9] = steering_cmds
        
        
        features = np.zeros((len(throttle_cmds) - horizon -1,  horizon, 7), dtype=np.double)
        labels = np.zeros((len(throttle_cmds) - horizon -1, 3), dtype=np.double)

        time_features = np.zeros((len(throttle_cmds) - horizon -1,1),dtype=np.double)

        
        timelines = np.zeros((len(throttle_cmds) - horizon -1,1),dtype=np.double)
        speeds = np.zeros((len(throttle_cmds) - horizon -1,3),dtype=np.double)

        for i in tqdm(range(len(throttle_cmds) - horizon - 1 - 25), desc="Compiling dataset"):

            features[i] = np.array([*poses_arr[i:i+horizon]])[:,3:-1]
            
            labels[i] = poses_arr[i+horizon,3:6]

            speeds[i,:2] = poses_arr[i+horizon,3:5]
            speeds[i,2] = poses_arr[i+horizon,7]
            timelines[i] = poses_arr[i+horizon,-1]
            time_features[i] = poses_arr[i+horizon-1,-1]
        features_refine = []
        features_refine.append(features[refined_range1[0]:refined_range1[1]])
        features_refine.append(features[refined_range2[0]:refined_range2[1]])
        features_refine.append(features[refined_range3[0]:refined_range3[1]])
        features_refine = np.concatenate(features_refine,axis=0)
        
        labels_refine = []
        labels_refine.append(labels[refined_range1[0]:refined_range1[1]])
        labels_refine.append(labels[refined_range2[0]:refined_range2[1]])
        labels_refine.append(labels[refined_range3[0]:refined_range3[1]])
        labels_refine = np.concatenate(labels_refine,axis=0)
        
        speeds_refine = []
        speeds_refine.append(speeds[refined_range1[0]:refined_range1[1]])
        speeds_refine.append(speeds[refined_range2[0]:refined_range2[1]])
        speeds_refine.append(speeds[refined_range3[0]:refined_range3[1]])
        speeds_refine = np.concatenate(speeds_refine, axis=0)
        
        
        timelines_refine = []
        timelines_refine.append(timelines[refined_range1[0]:refined_range1[1]])
        timelines_refine.append(timelines[refined_range2[0]:refined_range2[1]])
        timelines_refine.append(timelines[refined_range3[0]:refined_range3[1]])
        timelines_refine = np.concatenate(timelines_refine,axis=0)
        
        time_features_refine = []
        time_features_refine.append(time_features[refined_range1[0]:refined_range1[1]])
        time_features_refine.append(time_features[refined_range2[0]:refined_range2[1]])
        time_features_refine.append(time_features[refined_range3[0]:refined_range3[1]])
        time_features_refine = np.concatenate(time_features_refine,axis=0)
        
        
        if save:

            typical_features = []
            typical_timelines = []
            typical_labels = []
            typical_timefeatures = []

            slip_angle = np.rad2deg(np.arctan2(speeds_refine[:,1],speeds_refine[:,0])-speeds_refine[:,2]).tolist()

            print("slip angle max:",max(slip_angle))


            for i in range(len(slip_angle)-1):
                if abs(slip_angle[i]) <= 1.5:
                    continue


                typical_features.append(features_refine[i])
                typical_timelines.append(timelines_refine[i])
                typical_labels.append(labels_refine[i])
                typical_timefeatures.append(time_features_refine[i])


            typical_features = np.array(typical_features)
            typical_timelines = np.array(typical_timelines)
            typical_labels = np.array(typical_labels)
            typical_timefeatures = np.array(typical_timefeatures)


            print("feature shape :",features.shape)
            print("input data shape :",typical_features.shape)
            print("label shape:",typical_labels.shape)

            if save:
                np.savez(csv_path[:csv_path.find(".csv")] + "_" + str(horizon) + "RNN.npz", features=typical_features, 
                            labels=typical_labels,times_features=typical_timefeatures, times = typical_timelines)
                np.savez(csv_path[:csv_path.find(".csv")] + "_" + str(horizon) + "RNN_Val.npz", features=features, 
                        labels=labels,times_features=time_features, times = timelines)
            return features, labels


        

            
            




if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Convert CSV file to pickled dataset")
    parser.add_argument("csv_path", type=str, help="CSV file to convert")
    parser.add_argument("horizon", type=int, help="Horizon of timestamps used")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict : dict = vars(args)
    SAMPLING_TIME = 0.04
    TOTAL_TIME = 0.00
    write_dataset(argdict["csv_path"], argdict["horizon"])
    