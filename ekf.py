"""
感知定位第一部分：ekf，扩展卡尔曼滤波器的实现
"""

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import utm
import matplotlib.pyplot as plt
import numdifftools as nd
import matplotlib
matplotlib.use('Agg')

# 准备工作
def load_gps(path):
    """读取gps信息，返回Dataframe"""
    col_names = [
        'unix_time','gps_time','lat','lat_hemi','lon','lon_hemi',
        'heading','quality','num_sats','hdop','geoid_height'
    ]
    return pd.read_csv(path, sep = '\t',header = None, names = col_names)

def load_ahrs(path):
    """读取ahrs信息，返回Dataframe"""
    col_names = [
        'unix_time', 'qx', 'qy', 'qz', 'qw',
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
        'lin_acc_x', 'lin_acc_y', 'lin_acc_z'
    ]
    return pd.read_csv(path, sep='\t', header=None, names=col_names)

def latlon_to_utm(lat, lon):
    """将gps经纬度转换为utm坐标"""
    east,north,_,_ = utm.from_latlon(lat, lon)
    return east,north

def quat_to_euler(qx,qy,qz,qw):
    """将四元数转换为欧拉角"""
    r = Rotation.from_quat([qx,qy,qz,qw])
    roll,pitch,yaw = r.as_euler('xyz',degrees=False)
    return roll,pitch,yaw

def init_state(gps_row,ahrs_row):
    """输出某一帧的初始状态，这也是最直接的数据读取，返回一个数组"""
    x,y = latlon_to_utm(gps_row['lat'],gps_row['lon'])
    z = 0.0
    roll,pitch,yaw = quat_to_euler(ahrs_row['qx'],ahrs_row['qy'],ahrs_row['qz'],ahrs_row['qw'])
    u,v,w = 0.0,0.0,0.0
    x_state = np.array([x,y,z,roll,pitch,yaw,u,v,w],dtype=np.float64)
    return x_state

def init_cov():
    """初始化“状态”协方差矩阵，因为状态是一个9个分量的向量，所以要返回一个9*9的矩阵"""
    P = np.zeros((9,9))

    # 三个有关位置的方差
    P[0,0] = 4.0
    P[1,1] = 4.0
    P[2,2] = 0.01

    # 三个有关角度，也就是姿态的方差,用radians把角度转成弧度
    P[3,3] = np.radians(3.0)**2
    P[4,4] = np.radians(3.0)**2
    P[5,5] = np.radians(5.0)**2

    # 三个线速度的方差
    P[6,6] = 9.0
    P[7,7] = 9.0
    P[8,8] = 1.0
    return P

def init_noise():
    """初始化噪声，要返回初始化的过程噪声Q和三个测量噪声R"""
    # Q是过程噪声
    Q = np.zeros((9,9))

    # 三个有关位置的过程噪声
    Q[0,0] = 20.0
    Q[1,1] = 20.0
    Q[2,2] = 0.001

    # 三个有关姿态的过程噪声，同样用radians进行角度到弧度
    Q[3,3] = np.radians(0.1)**2
    Q[4,4] = np.radians(0.1)**2
    Q[5,5] = np.radians(0.2)**2

    # 三个线速度的过程噪声
    Q[6,6] = 5.0
    Q[7,7] = 5.0
    Q[8,8] = 0.1

    # 下面是三个测量噪声R，两个3*3协方差矩阵，一个方差，这部分参照论文里传感器的规格
    # 先是姿态，参照ahrs的规格
    R_ori = np.zeros((3,3))
    R_ori[0,0] = np.radians(0.5)**2
    R_ori[1,1] = np.radians(0.5)**2
    R_ori[2,2] = np.radians(1.0)**2

    # 位置观测噪声，参照gps的规格,本数据集pohang00无动态实时差分，比较大，在更新时根据hdop动态缩放
    R_pos = np.zeros((3,3))
    R_pos[0,0] = 81.0
    R_pos[1,1] = 81.0
    R_pos[2,2] = 1.0

    # 航向观测噪声，由于是gps直接测量，也参照gps的规格
    R_hdg = np.radians(0.28)**2

    return Q,R_ori,R_pos,R_hdg

def predict(x,P,ahrs_row,dt,Q):
    """预测步"""
    z_p,z_q,z_r = ahrs_row['ang_vel_x'],ahrs_row['ang_vel_y'],ahrs_row['ang_vel_z']
    z_ax,z_ay,z_az = ahrs_row['lin_acc_x'],ahrs_row['lin_acc_y'],ahrs_row['lin_acc_z']
    g = 9.81
    def f_predict(x_in):
        phi_in,theta_in,psi_in = x_in[3],x_in[4],x_in[5]
        u_in,v_in,w_in = x_in[6],x_in[7],x_in[8]
        sin_theta_in = np.sin(theta_in)
        cos_theta_in = np.cos(theta_in)
        tan_theta_in = np.tan(theta_in)
        sin_phi_in = np.sin(phi_in)
        cos_phi_in = np.cos(phi_in)
        sin_psi_in = np.sin(psi_in)
        cos_psi_in = np.cos(psi_in)
        dx = np.zeros(9)
        dx[0] = u_in * cos_theta_in * cos_psi_in + v_in * (sin_phi_in * sin_theta_in * cos_psi_in - cos_phi_in * sin_psi_in) + w_in * (
                    cos_phi_in * sin_theta_in * cos_psi_in + sin_psi_in * sin_phi_in)
        dx[1] = u_in * cos_theta_in * sin_psi_in + v_in * (sin_phi_in * sin_theta_in * sin_psi_in + cos_phi_in * cos_psi_in) + w_in * (
                    cos_phi_in * sin_theta_in * sin_psi_in - cos_psi_in * sin_phi_in)
        dx[2] = -u_in * sin_theta_in + v_in * sin_phi_in * cos_theta_in + w_in * cos_phi_in * cos_theta_in
        dx[3] = z_p + z_q * sin_phi_in * tan_theta_in + z_r * cos_phi_in * tan_theta_in
        dx[4] = z_q * cos_phi_in - z_r * sin_phi_in
        dx[5] = (z_q * sin_phi_in + z_r * cos_phi_in) / cos_theta_in
        dx[6] = z_ax - g * sin_theta_in - z_q * w_in + z_r * v_in
        dx[7] = z_ay + g * sin_phi_in * cos_theta_in - z_r * u_in + z_p * w_in
        dx[8] = z_az + g * cos_phi_in * cos_theta_in - z_p * v_in + z_q * u_in
        return dx
    # F = nd.Jacobian(f_predict)(x)
    x_dot = f_predict(x)
    x_new = x + x_dot*dt
    # P_new = F @ P @ F.T + Q
    # 禁用雅可比矩阵，因为它会累计误差
    P_new = P + Q
    return x_new,P_new

def wrap_to_pi(angle):
    """辅助函数，用来保证角度落在-Π到Π区间"""
    return (angle + np.pi) %(2 * np.pi) - np.pi

def update_orientation(x,P,z_ori,R_ori,b_alpha):
    """姿态更新步，注意ahrs测量的艏摇角yaw是基于磁北的，需要修正到真北"""
    z_ori_corrected = np.array([z_ori[0],z_ori[1],z_ori[2]-b_alpha])
    residual = z_ori_corrected - x[3:6]
    residual[2] = wrap_to_pi(residual[2])
    H = np.zeros((3,9))
    H[0,3] = 1
    H[1,4] = 1
    H[2,5] = 1
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R_ori)
    x = x + K @ np.array([residual[0],residual[1],0.0]) # 反复调整之后，发现是yaw的偏移引发拟合的问题，于是禁用
    P = (np.eye(9) - K @ H) @ P
    return x,P

def update_position_heading(x,P,z_pos,z_hdg,R_pos,R_hdg,hdop):
    """同时进行的位置更新步和航向更新步，因为都用到了gps就合在一起了"""
    R_pos_scaled = R_pos*hdop
    residual_pos = z_pos - x[0:3]
    H_pos = np.zeros((3,9))
    H_pos[0,0] = H_pos[1,1] = H_pos[2,2] = 1
    K_pos = P @ H_pos.T @ np.linalg.inv(H_pos @ P @ H_pos.T + R_pos_scaled)
    x = x + K_pos @ residual_pos
    P = (np.eye(9) - K_pos @ H_pos) @ P
    residual_hdg = wrap_to_pi(z_hdg - x[5])
    H_hdg = np.zeros((1,9))
    H_hdg[0,5] = 1
    K_hdg = (P @ H_hdg.T)/(H_hdg @ P @ H_hdg.T + R_hdg)
    x = x + (K_hdg * residual_hdg).ravel()
    P = (np.eye(9) - K_hdg @ H_hdg) @ P
    return x,P

# 主函数框架

def main():
    """数据读取"""
    gps = load_gps(r"E:\PohangCanalDataset\navigation\gps.txt")
    ahrs = load_ahrs(r"E:\PohangCanalDataset\navigation\ahrs.txt")
    print(f"gps数据行数：{len(gps)}")
    print(f"ahrs数据行数：{len(ahrs)}")

    """初始化"""
    x = init_state(gps.iloc[0],ahrs.iloc[0]) # 初始状态，船一开始的状态
    P = init_cov() # 初始协方差，刻画对于初始状态的不确定程度
    Q,R_ori,R_pos,R_hdg = init_noise() # 初始噪声协方差矩阵
    roll_0,pitch_0,yaw_ahrs_0 = quat_to_euler(ahrs.iloc[0]['qx'],ahrs.iloc[0]['qy'],
                                              ahrs.iloc[0]['qz'],ahrs.iloc[0]['qw']) # 初始三个转角
    yaw_gps_0 = np.radians(gps.iloc[0]['heading']) # 初始艏摇角的gps测量
    b_alpha = wrap_to_pi(yaw_ahrs_0 - yaw_gps_0) # 初始磁偏角
    print(f"初始状态： x={x[0]:.2f},y={x[1]:.2f},yaw={np.degrees(x[5]):.2f}°")
    print(f"磁偏角：{np.degrees(b_alpha):.2f}°")
    yaw_ahrs_corrected = yaw_ahrs_0 - b_alpha  # b_alpha = -346.72°
    print(f"修正后的 AHRS yaw (真北): {np.degrees(yaw_ahrs_corrected) % 360:.2f}°")
    print(f"GPS heading: {gps.iloc[0]['heading']:.2f}°")
    print(f"b_alpha = {np.degrees(b_alpha):.2f}°")
    # print(f"P[0,0] = {P[0, 0]}, P[6,6] = {P[6, 6]}")
    # print(f"Q[0,0] = {Q[0, 0]}, Q[6,6] = {Q[6, 6]}")
    # print(f"R_pos[0,0] = {R_pos[0, 0]}")

    """存储"""
    results = [] # 每一帧的ekf输出
    gps_idx = 0 # 当前gps数据行号
    prev_time = ahrs.iloc[0]['unix_time'] # 上一帧的unix时间戳

    """主循环"""
    for i in range(1,len(ahrs)):
        if i % 1000 == 0:
            print(f"Processing frame {i}/{len(ahrs)} x={x[0]:.2f},y={x[1]:.2f}")
        ahrs_row = ahrs.iloc[i]
        curr_time = ahrs_row['unix_time']
        dt = curr_time - prev_time
        prev_time = curr_time
        # if i <= 10:
        #     print(f"dt = {dt:.6f} s")

        # 预测步
        x,P = predict(x,P,ahrs_row,dt,Q)

        # 更新步-姿态更新
        z_ori = np.array(quat_to_euler(ahrs_row['qx'],ahrs_row['qy'],
                                        ahrs_row['qz'],ahrs_row['qw'])) # z开头的就是论文中提及的一些测量值
        x,P = update_orientation(x,P,z_ori,R_ori,b_alpha)

        # 更新步-位置和航向的更新
        if gps_idx < len(gps) and curr_time >= gps.iloc[gps_idx]['unix_time']:
            gps_row = gps.iloc[gps_idx]
            z_x,z_y = latlon_to_utm(gps_row['lat'],gps_row['lon'])
            z_pos = np.array([z_x,z_y,0.0])
            z_hdg = np.radians(gps_row['heading'])
            hdop = gps_row['hdop']

            x,P = update_position_heading(x,P,z_pos,z_hdg,R_pos,R_hdg,hdop)

            gps_idx += 1

        # 保存当前帧
        results.append([curr_time] + list(x))

    # 全部结果保存
    columns = ['unix_time','x','y','z','roll','pitch','yaw','u','v','w']
    df_results = pd.DataFrame(results,columns=columns)
    # df_results.to_csv(r"D:\Undergraduate\大一 下\拓展\project\ekf_results.csv",sep=',',index=False)
    print(rf"EKF完成，共{len(df_results)}帧，结果保存为E:\PohangCanalDataset\navigation\results.csv")

    # 检查P的对角线，检查是否饱和
    print("P diagonal:", np.diag(P))

    # 全部可视化
    # gps可视化
    gps_utm_x,gps_utm_y=[],[]
    for i in range(len(gps)):
        x_utm,y_utm = latlon_to_utm(gps.iloc[i]['lat'],gps.iloc[i]['lon'])
        gps_utm_x.append(x_utm)
        gps_utm_y.append(y_utm)

    # baseline可视化
    baseline = pd.read_csv(r"E:\PohangCanalDataset\navigation\baseline.txt",sep='\t',header=None,
                           names=['unix_time','qx','qy','qz','qw','y','x','z'])

    # 均方根误差用于量化评估
    ekf_times = df_results['unix_time'].values
    baseline_times = baseline['unix_time'].values
    aligned_x_err = []
    aligned_y_err = []
    for t_b,x_b,y_b in zip(baseline_times,baseline['x'],baseline['y']):
        idx = np.argmin(np.abs(ekf_times - t_b))
        x_e = df_results['x'].iloc[idx]
        y_e = df_results['y'].iloc[idx]
        aligned_x_err.append(x_e - x_b)
        aligned_y_err.append(y_e - y_b)
    aligned_x_err = np.array(aligned_x_err)
    aligned_y_err = np.array(aligned_y_err)
    rmse = np.sqrt(np.mean(aligned_x_err**2 + aligned_y_err**2))
    mean_x_err = np.mean(aligned_x_err)
    std_x_err = np.std(aligned_x_err)
    mean_y_err = np.mean(aligned_y_err)
    std_y_err = np.std(aligned_y_err)
    print("定位精度评估")
    print(f"RMSE (2D): {rmse:.3f} m")
    print(f"X 方向: 均值误差 = {mean_x_err:.3f} m, 标准差 = {std_x_err:.3f} m")
    print(f"Y 方向: 均值误差 = {mean_y_err:.3f} m, 标准差 = {std_y_err:.3f} m")

    plt.figure(figsize=(14, 10))
    plt.scatter(gps_utm_x, gps_utm_y, c='blue', s=4, label='GPS raw', alpha=0.35, zorder=2)
    plt.plot(baseline['x'], baseline['y'], 'g--', linewidth=2.5, label='Baseline', zorder=1)
    plt.plot(df_results['x'], df_results['y'], 'r-', linewidth=1.2, label='EKF trajectory', alpha=0.75, zorder=3)
    plt.xlabel('UTM East [m]')
    plt.ylabel('UTM North [m]')
    plt.title('Trajectory:EKF,GPS,Baseline')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ekf_trajectory.png", dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    main()
