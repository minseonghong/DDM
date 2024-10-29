#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import pandas as pd
from morai_msgs.msg import EgoVehicleStatus, CtrlCmd , GPSMessage
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion
import signal
import sys
import message_filters
import csv

class DataProcess:
    def __init__(self):
        rospy.init_node('EKF_VELOCITY_NODE', anonymous=True)

        ego_sub = message_filters.Subscriber("Ego_topic", EgoVehicleStatus)
        imu_sub = message_filters.Subscriber("/imu", Imu)
        gps_sub = message_filters.Subscriber("/gps_utm_odom", Odometry)

        # ctrl_sub = message_filters.Subscriber("/ctrl_cmd_0", CtrlCmd)


        self.ts = message_filters.ApproximateTimeSynchronizer([ego_sub, imu_sub, gps_sub], queue_size=10, slop=0.01)
        self.ts.registerCallback(self.callback)

        # self.dlc_pub = rospy.Publisher("/ctrl_cmd_0", CtrlCmd, queue_size=1)  #여기 아마 ctrl_cmd
        
        # self.ctrl_cmd_msg = CtrlCmd()
        # self.ctrl_cmd_msg.longlCmdType = 1
        # self.ctrl_cmd_msg.accel = 0
        # self.ctrl_cmd_msg.brake = 0
        # self.ctrl_cmd_msg.steering = 0
        self.is_gps = False



        # Data storage
        self.data = {
            'vx': [],
            'vy': [],
            'yaw_rate': [],
            'steering': [],
            'throttle': [],
            'brake' : [],
            'X' : [],
            'Y' : []
      
        }

        # Variables to store previous throttle and steering
        self.prev_throttle = 0.0
        self.prev_steer = 0.0
        self.total_time = 0.0
        self.sampling_time = 0.02  # 50 Hz

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def steering_to_wheel_angle(self, steering):
        """
        Convert steering value (-36.3° to 36.3°) to wheel angle (-45° to 45°)
        and then to radians.
        """
        # 선형 변환: Steering (-36.3 ~ 36.3) -> Wheel angle (-45 ~ 45)
        wheel_angle_deg = (steering / 36.3) * 45.0
        # Convert to radians
        # wheel_angle_rad = np.deg2rad(wheel_angle_deg)
        wheel_angle_rad = np.deg2rad(wheel_angle_deg)
        return wheel_angle_rad
    

    def callback(self, ego_msg, imu_msg, gps_msg):
        # Retrieve current data


        vx = round(ego_msg.velocity.x, 8)
        vy = round(ego_msg.velocity.y, 8)
        yaw_rate = round(imu_msg.angular_velocity.z, 8)
        # steering = np.deg2rad(round(ego_msg.wheel_angle, 8))
        steering = round(ego_msg.wheel_angle, 8)
        steering = self.steering_to_wheel_angle(-steering)
        throttle = round(ego_msg.accel, 8)  # Throttle as percentage
        brake = round(ego_msg.brake, 8)  # Throttle as percentage
        self.is_gps = True

        #odom_quaternion = (gps_msg.pose.pose.orientation.x, gps_msg.pose.pose.orientation.y)
        
        X = gps_msg.pose.pose.position.x
        Y = gps_msg.pose.pose.position.y
        # Calculate throttle_cmd and steering_cmd


        # Store current values
        self.data['vx'].append(vx)
        self.data['vy'].append(vy)
        self.data['yaw_rate'].append(yaw_rate)
        self.data['steering'].append(steering)
        self.data['throttle'].append(throttle)
        self.data['brake'].append(brake)
        self.data['X'].append(X)
        self.data['Y'].append(Y)


        # Update previous values for next callback


        # Print the data for each callback (optional for debugging)
        print(f"vx: {vx:.8f} | vy: {vy:.8f} | yaw_rate: {yaw_rate:.8f} | steering: {steering:.8f} | throttle: {throttle:.8f} | brake: {brake:.8f} | X: {X:.8f} | Y: {Y:.8f} ")

    def save_data(self):
        # Ensure all arrays are of the same length
        min_len = min(len(self.data['vx']), len(self.data['vy']), len(self.data['yaw_rate']),
                      len(self.data['steering']), len(self.data['throttle']),
                      len(self.data['brake']), len(self.data['X']), len(self.data['Y']),
                      )
        
        for key in self.data:
            self.data[key] = self.data[key][:min_len]

        # Save to CSV
        output_csv = 'vehicle_data_logging_double_lane_deg_train.csv'
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["vx(m/s)", "vy(m/s)", "yaw_rate(rad/s)", "steering(rad)", "throttle(%)", "brake(%)", "X" , "Y" ])
            # Write data
            for i in range(min_len):
                writer.writerow([
                    self.data['vx'][i], self.data['vy'][i], self.data['yaw_rate'][i], 
                    self.data['steering'][i], self.data['throttle'][i], self.data['brake'][i],self.data['X'][i], self.data['Y'][i],   
                  
                ])

        print(f"Data saved to {output_csv}")
    
    def signal_handler(self, sig, frame):
        print('Signal received, saving data and shutting down...')
        self.save_data()
        rospy.signal_shutdown('Signal received, shutting down...')
        sys.exit(0)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        data = DataProcess()
        data.run()
    except rospy.ROSInterruptException:
        pass
