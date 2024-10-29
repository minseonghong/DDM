#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys
import csv
from tqdm import tqdm

MAX_BRAKE_PRESSURE = 2757.89990234

def write_dataset(csv_path, horizon, save=True, plot=False):
    global TOTAL_TIME

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
        print("horizon_step", horizon)
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
        
        if plot:

            plt.figure()
            plt.plot(time,vx_arr,label="vx")
            plt.title("Vehicle Longitudinal Velocity (vx) Over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.legend()
            # plt.plot(time,smoothed_vx,label="smoothed vx")
            plt.show()
            
            plt.figure()
            plt.plot(time,vy_arr,label="vy")
            plt.title("Vehicle Lateral Velocity (vy) Over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.legend()
            #plt.plot(time,smoothed_vy,label="smoothed vy")
            plt.show()
            
            plt.figure()
            plt.plot(time,vphi_arr,label="yaw_rate")
            plt.title("Vehicle Angular Velocity (yaw_rate) Over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("yaw_rate (rad/s)")
            plt.legend()
            #plt.plot(time,smoothed_vphi,label="smoothed vphi")
            plt.show()
            
            plt.figure()
            plt.plot(time,steering_arr,label="steering")
            plt.title("Steering Angle Over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Steering Angle (rad)")
            plt.legend()
            #plt.plot(time,smoothed_steering,label="smoothed steering")
            plt.show()
            
            plt.figure()
            plt.plot(time,throttle_arr,label="Throttle")
            plt.title("Throttle Command Over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Throttle Command (%)")
            plt.legend()
            #plt.plot(time,smoothed_throttle,label="smoothed throttle")
            plt.show()
        features = np.zeros((len(throttle_cmds) - horizon -1,  horizon, 7), dtype=np.double)
        labels = np.zeros((len(throttle_cmds) - horizon -1, 3), dtype=np.double)

        time_features = np.zeros((len(throttle_cmds) - horizon -1,1),dtype=np.double)

        
        timelines = np.zeros((len(throttle_cmds) - horizon -1,1),dtype=np.double)
        speeds = np.zeros((len(throttle_cmds) - horizon -1,3),dtype=np.double)

        for i in tqdm(range(len(throttle_cmds) - horizon - 1), desc="Compiling dataset"):

            features[i] = np.array([*poses_arr[i:i+horizon]])[:,3:-1]

            labels[i] = poses_arr[i+horizon,3:6]

            speeds[i,:2] = poses_arr[i+horizon,3:5]
            speeds[i,2] = poses_arr[i+horizon,7]
            timelines[i] = poses_arr[i+horizon,-1]
            time_features[i] = poses_arr[i+horizon-1,-1]
        
        if save:
            print("feature shape :",features.shape)
            print("label shape:",labels.shape)

            if save:
                np.savez(csv_path[:csv_path.find(".csv")] + "_" + str(horizon) + "RNN_hong_orgin_Val.npz", features=features, 
                        labels=labels,times_features=time_features, times = timelines)
                # np.savez(csv_path[:csv_path.find(".csv")] + "_" + str(horizon) + "RNN_Val.npz", features=features, 
                #         labels=labels,times_features=time_features, times = timelines)
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
    # write_dataset(argdict["csv_path"])
    