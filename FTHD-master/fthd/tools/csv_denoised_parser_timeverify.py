#!/usr/bin/env python3

import numpy as np
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt

def write_dataset(input_file, horizon, save=True, plot=False):
    global TOTAL_TIME
    data = np.load(input_file)
    features = data['features'] # vx, vy, yawRate, throttle, steering ,throttle_cmd, steering_cmd, t
    times = features[:,-1]
    print("times is:",times)
    # labels = data['labels']
    # time_features = data['time_features']
    # timelines = data['times']
    time = np.zeros((features.shape[0],1),dtype=np.double)
    print(features.shape)
    dfeatures = np.zeros((features.shape[0] - horizon - 1,  horizon, 7), dtype=np.double)
    labels = np.zeros((features.shape[0] - horizon - 1, 3), dtype=np.double)
    time_features = np.zeros((features.shape[0] - horizon - 1, 1), dtype=np.double)
    timelines = np.zeros((features.shape[0] - horizon - 1, 1), dtype=np.double)
    
    for i in range(features.shape[0]):
        time[i] = TOTAL_TIME
        TOTAL_TIME += SAMPLING_TIME
    
    for i in tqdm(range(features.shape[0] - horizon - 1), desc="Compiling dataset"):
        dfeatures[i] = features[i:i+horizon,:-1]
        labels[i] = features[i+horizon,:3]
        time_features[i] = times[i+horizon-1]
        timelines[i] = times[i+horizon]
    
    print("dfeature shape:",dfeatures.shape)
    print("labels shape:",labels.shape)
    print("time features shape:",time_features.shape)
    print("timeline shape:",timelines.shape)
    
    print("last time:",time[-1])
    
    if plot:
        plt.plot(time,features[:,1],label = "vy")
        plt.legend()
        plt.show()

    if save:
        np.savez(input_file[:input_file.find(".npz")] + "_" + str(horizon) + "RNN_denoised_hong_val.npz", features=dfeatures, 
                 labels=labels,times_features=time_features, times = timelines)
    return features
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./csv_denoised_parser_timeverify.py input_file horizon")
    else:
        TOTAL_TIME = 0.00
        SAMPLING_TIME = 0.02
        write_dataset(sys.argv[1], int(sys.argv[2]))