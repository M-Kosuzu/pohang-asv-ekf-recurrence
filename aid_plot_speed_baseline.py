"""
简单脚本：基于 baseline.txt 计算船速并绘制双纵坐标图
用于辅助判断内港→外港的过渡区域
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================== 设置文件路径和分隔符 ====================
BASELINE_FILE = r"E:\PohangCanalDataset\navigation\baseline.txt"
SEP = '\t'   # baseline.txt 是制表符分隔

# ==================== 读取数据 ====================
cols = ['unix_time', 'qx', 'qy', 'qz', 'qw', 'x', 'y', 'z']
baseline = pd.read_csv(BASELINE_FILE, sep=SEP, header=None, names=cols)

# ==================== 计算相邻帧位移和时间差 ====================
dt = np.diff(baseline['unix_time'])          # 时间间隔 (秒)
dx = np.diff(baseline['x'])                  # 东向位移 (米)
dy = np.diff(baseline['y'])                  # 北向位移 (米)

# ==================== 合成速度 ====================
speed_ms = np.sqrt(dx**2 + dy**2) / dt       # 米/秒
speed_knots = speed_ms * 1.944               # 节 (1 m/s ≈ 1.944 kn)

# 速度对应的时间（用差分后的时间戳，取右端点）
speed_time = baseline['unix_time'].iloc[1:].values

# 可选：对速度做 10 点移动平均，让曲线更平滑
window = 10
if len(speed_ms) > window:
    speed_ms_smooth = np.convolve(speed_ms, np.ones(window)/window, mode='same')
    speed_knots_smooth = np.convolve(speed_knots, np.ones(window)/window, mode='same')
else:
    speed_ms_smooth = speed_ms
    speed_knots_smooth = speed_knots

# ==================== 绘图 ====================
fig, ax1 = plt.subplots(figsize=(14, 5))

# 左侧纵轴 — 节
ax1.plot(speed_time, speed_knots_smooth, 'b-', linewidth=0.8, alpha=0.8)
ax1.set_xlabel('Unix Time', fontsize=12)
ax1.set_ylabel('Speed [knots]', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, linestyle='--', alpha=0.4)

# 右侧纵轴 — 米/秒
ax2 = ax1.twinx()
ax2.plot(speed_time, speed_ms_smooth, 'r-', linewidth=0.8, alpha=0.3)
ax2.set_ylabel('Speed [m/s]', color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')

# 论文参考线
ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Inner Port (5 kn)')
ax1.axhline(y=7, color='orange', linestyle='--', alpha=0.5, label='Inner Port (7 kn)')
ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Coastal (10 kn)')

ax1.legend(loc='upper left')
plt.title('Speed from Baseline Trajectory (SLAM)')
plt.tight_layout()
plt.savefig("baseline_speed.png", dpi=200)
plt.show()
print("图像已保存为 baseline_speed.png")