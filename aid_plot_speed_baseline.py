import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "ekf_results.csv"
SEP = ','

df = pd.read_csv(CSV_FILE, sep=SEP)
df['speed_ms'] = np.sqrt(df['u']**2 + df['v']**2)
df['speed_knots'] = df['speed_ms'] * 1.944   # 1 m/s ≈ 1.944 kn

fig, ax1 = plt.subplots(figsize=(14, 5))

# 左侧纵轴 — 节
ax1.plot(df['unix_time'], df['speed_knots'], 'b-', linewidth=0.8, alpha=0.7)
ax1.set_xlabel('Unix Time')
ax1.set_ylabel('Speed [knots]', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, linestyle='--', alpha=0.4)

# 右侧纵轴 — 米/秒
ax2 = ax1.twinx()
ax2.plot(df['unix_time'], df['speed_ms'], 'r-', linewidth=0.8, alpha=0.3)  # 半透明，避免干扰
ax2.set_ylabel('Speed [m/s]', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 参考线（论文内港 5–7 节，外海 10 节）
ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Inner Port (5 kn)')
ax1.axhline(y=7, color='orange', linestyle='--', alpha=0.5, label='Inner Port (7 kn)')
ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Coastal (10 kn)')

ax1.legend(loc='upper left')
plt.title('EKF Speed (not accurate, for trend only)')
plt.tight_layout()
plt.savefig("speed_plot.png", dpi=200)
plt.show()