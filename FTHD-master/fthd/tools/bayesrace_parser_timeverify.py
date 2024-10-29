#!/usr/bin/env python3

import numpy as np
import sys
from tqdm import tqdm

def write_dataset(input_file, horizon, save=True):
    global TOTAL_TIME
    data = np.load(input_file)
    states =  data["dstates"]
    inputs = data["inputs"]
    poses = data["states"][:6,:].T
    # vrefs = data["vrefs"]
    odometry = states[3:,:].T # vx, vy, yaw_rate, throttle_fb, steering_fb
    pose_times = np.zeros((len(poses),4),dtype=np.double)

    for i in range(len(odometry)):
        pose_times[i,:3] = poses[i,:3]
        pose_times[i,3] = TOTAL_TIME
        TOTAL_TIME += SAMPLING_TIME

    print("poses shape :",poses.shape)
    print("poses times :",pose_times.shape)

    throttle_cmd = inputs[0,:] - [0.0, *inputs[0,:-1]]
    steering_cmd = inputs[1,:] - [0.0, *inputs[1,:-1]]
    features = np.zeros((len(odometry) - horizon - 1,  horizon, 7), dtype=np.double)
    labels = np.zeros((len(odometry) - horizon - 1, 3), dtype=np.double)
    time_features = np.zeros((len(odometry) - horizon - 1, 1), dtype=np.double)

    timelines = np.zeros((len(odometry) - horizon - 1, 1), dtype=np.double)
    speeds = np.zeros((len(odometry) - horizon - 1,3),dtype=np.double)
    for i in tqdm(range(len(throttle_cmd) - horizon - 1), desc="Compiling dataset"):
        features[i] = np.array([*odometry[i:i+horizon].T, throttle_cmd[i:i+horizon], steering_cmd[i:i+horizon]]).T
        time_features[i] = pose_times[i+horizon-1,-1]
        labels[i] = np.array([*odometry[i+horizon]])[:3]

        speeds[i,:2] = odometry[i+horizon,:2]
        speeds[i,2] = odometry[i+horizon,-1]
        timelines[i] = pose_times[i+horizon,-1]
    print("Final features shape:", features.shape)
    print("Final labels shape:", labels.shape)


    slip_angle = np.rad2deg(np.arctan2(speeds[:,1],speeds[:,0])-speeds[:,2]).tolist()
    print("slip angle max:",max(slip_angle))

    typical_features = []
    typical_timelines = []
    typical_labels = []
    typical_timefeatures = []

    for i in range(len(slip_angle)-1):
        if abs(slip_angle[i]) <= 5:
            continue
        typical_features.append(features[i])
        typical_timelines.append(timelines[i])
        typical_labels.append(labels[i])
        typical_timefeatures.append(time_features[i])


    typical_features = np.array(typical_features)
    typical_timelines = np.array(typical_timelines)
    typical_labels = np.array(typical_labels)
    typical_timefeatures = np.array(typical_timefeatures)

    print("typical features :",typical_features.shape)
    print("typical timelines :",typical_timelines.shape)

    if save:
        np.savez(input_file[:input_file.find(".npz")] + "_" + str(horizon) + "RNN.npz", features=typical_features, 
                 labels=typical_labels,times_features=typical_timefeatures, times = typical_timelines)

        np.savez(input_file[:input_file.find(".npz")] + "_" + str(horizon) + "RNN_val.npz", features=features, 
                 labels=labels,times_features=time_features, times = timelines)
    return features, labels, poses
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./bayesrace_parser_timeverify.py input_file horizon")
    else:
        TOTAL_TIME = 0.00
        SAMPLING_TIME = 0.02
        write_dataset(sys.argv[1], int(sys.argv[2]))
