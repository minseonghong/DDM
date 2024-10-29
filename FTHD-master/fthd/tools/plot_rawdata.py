import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# CSV 파일 불러오기
df = pd.read_csv('/home/hong/Downloads/vehicle_data_logging_double_lane_deg_xy.csv')

# 시간 계산 (각 행이 0.02초 간격이므로)
time = [i * 0.02 for i in range(len(df))]

# 플롯 설정
label_fontsize = 20  # X, Y 라벨 크기
title_fontsize = 16  # 제목 크기
tick_label_fontsize = 15  # x축과 y축 눈금 크기

# vx와 steering 값 플롯
plt.figure(figsize=(12, 6))

# vx plot
plt.subplot(2, 1, 1)
plt.plot(time, df['vx(m/s)'], color='blue')
plt.xlabel('Time (s)', fontsize=label_fontsize)
plt.ylabel('Vx (m/s)', fontsize=label_fontsize)
# plt.title('Time vs vx', fontsize=title_fontsize)
plt.tick_params(axis='x', labelsize=tick_label_fontsize)  # x축 눈금 폰트 크기 조정
plt.tick_params(axis='y', labelsize=tick_label_fontsize)  # y축 눈금 폰트 크기 조정
plt.legend()
plt.grid(True)
plt.xlim([0, max(time)])  # x축을 0부터 데이터 끝까지

# steering plot
plt.subplot(2, 1, 2)
steer = np.rad2deg(df['steering(rad)'])
plt.plot(time, steer, color='green')
plt.xlabel('Time (s)', fontsize=label_fontsize)
plt.ylabel('Steering (deg)', fontsize=label_fontsize)
plt.tick_params(axis='x', labelsize=tick_label_fontsize)  # x축 눈금 폰트 크기 조정
plt.tick_params(axis='y', labelsize=tick_label_fontsize)  # y축 눈금 폰트 크기 조정
# plt.title('Time vs Steering', fontsize=title_fontsize)
plt.legend()
plt.grid(True)
plt.xlim([0, max(time)])  # x축을 0부터 데이터 끝까지

plt.tight_layout()
# PDF로 저장
plt.savefig('/home/hong/catkin_ws/src/FTHD-master/fthd/tools/output/time_vs_vx_steering.pdf')

plt.show()

# 궤적 (X, Y 좌표) 플롯
x_coords = df['X'] - df['X'].iloc[0]  # 시작점을 (0, 0)으로 조정
y_coords = df['Y'] - df['Y'].iloc[0]

plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, linestyle='-', color='orange')  # 선으로 플롯
plt.axis('equal')
plt.xlabel('X (m)', fontsize=label_fontsize)
plt.ylabel('Y (m)', fontsize=label_fontsize)
plt.tick_params(axis='x', labelsize=tick_label_fontsize)  # x축 눈금 폰트 크기 조정
plt.tick_params(axis='y', labelsize=tick_label_fontsize)  # y축 눈금 폰트 크기 조정
plt.title('Vehicle Trajectory', fontsize=title_fontsize)
plt.legend()
plt.grid(True)

# x축과 y축을 반전
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xlim([0, min(x_coords)])  # x축을 0부터 데이터 끝까지
# PDF로 저장
plt.savefig('/home/hong/catkin_ws/src/FTHD-master/fthd/tools/output/vehicle_trajectory.pdf', format='pdf')

plt.show()
