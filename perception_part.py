"""
总成的感知程序，按照ekf给出的位置所在的不同区域，调用不同函数进行检测
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# 各模块核心函数导入

from wall_detection_one_frame import (
    detect_walls_one_frame,
    load_extrinsics,
    build_transform,
    transform_points,
    read_lidar_bin,
    extract_timestamp,
    find_closest_file
)

from lidar_based_object_detection import (
    lidar_object_detect_one_frame
)

from radar_based_object_detection import (
    radar_object_detect_one_frame
)

# 一些辅助函数
def classify_region(lidar_frame_idx):
    # 为了方便，使用port lidar的索引，因为这个我自己确认过了
    if lidar_frame_idx < 6000:
        return 'canal'
    elif lidar_frame_idx < 15000:
        return 'inner_port'
    else:
        return 'outer_port_and_coastal'

def compute_speed_from_positions(x,y,unix_time):
    # ekf的位置差分来计算速度
    dt = np.diff(unix_time)
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.where(dt > 1e-6, dt, 1e-6) # 稳健，排除了可能除以0的情况
    vx = dx / dt
    vy = dy / dt
    speed = np.sqrt(vx**2 + vy**2)
    vx = np.insert(vx, 0, 0.0)
    vy = np.insert(vy, 0, 0.0)
    speed = np.insert(speed, 0, 0.0)
    return vx, vy, speed

def load_radar_timestamps(timestamp_path):
    # 读取radar的时间戳txt，返回两个数组：秒为单位的unix时间戳和对应的整数序号
    df = pd.read_csv(timestamp_path, sep = '\t', header = None,
                     names = ['unix_time', 'seq'])
    return df['unix_time'].values, df['seq'].astype(int).values

def find_closest_radar_image(target_time, times, seqs, radar_image_dir):
    # 根据目标时间戳，找到最接近的雷达图像,直接返回灰度图和对应的整数序号
    if len(times) == 0:
        return None, None

    idx = np.argmin(np.abs(times - target_time))
    seq = seqs[idx]

    filename = f"{seq:06d}.png"
    image_path = radar_image_dir / filename

    if image_path.exists():
        image = np.array(Image.open(image_path).convert('L'))
        return image, seq
    else:
        return None, None

# 主函数
def main():
    # 先加载ekf轨迹
    ekf_path = r"D:\Undergraduate\大一 下\拓展\project\ekf_results.csv"
    ekf_df = pd.read_csv(ekf_path, sep = ',')
    ekf_times = ekf_df['unix_time'].values

    # 用位置差分计算速度（ekf自己的速度准确性为了位置准确性牺牲了）
    vx, vy, speed = compute_speed_from_positions(ekf_df['x'].values, ekf_df['y'].values, ekf_df['unix_time'].values)
    ekf_df['vx'] = vx
    ekf_df['vy'] = vy
    ekf_df['speed'] = speed

    # 加载外参，lidar文件列表
    ext_path = r"E:\PohangCanalDataset\calibration\extrinsics.json"
    extrinsics = load_extrinsics(ext_path)

    R_front, t_front = build_transform(extrinsics['lidar_front'])
    R_port, t_port = build_transform(extrinsics['lidar_port'])
    R_stbd, t_stbd = build_transform(extrinsics['lidar_starboard'])

    front_dir = Path(r"E:\PohangCanalDataset\lidar\lidar_front\points")
    port_dir = Path(r"E:\PohangCanalDataset\lidar\lidar_port\points")
    starboard_dir = Path(r"E:\PohangCanalDataset\lidar\lidar_starboard\points")

    front_files = sorted(front_dir.glob("*.bin"))
    port_files = sorted(port_dir.glob("*.bin"))
    starboard_files = sorted(starboard_dir.glob("*.bin"))

    # 加载雷达时间戳txt和图像列表
    radar_image_dir = Path(r"E:\PohangCanalDataset\radar\images")
    radar_timestamps_dir = Path(r"E:\PohangCanalDataset\radar\timestamp.txt")
    radar_times, radar_seqs = load_radar_timestamps(radar_timestamps_dir)

    # 主循环
    output_rows = []
    port_idx = 0
    front_idx = 0
    starboard_idx = 0
    radar_idx = 0

    # 优化一下，ekf数据太多，只在lidar改变时，才进行一次检测
    last_port_idx = -1
    last_walls = []
    last_lidar_obstacles = []
    last_radar_image = None
    last_radar_obstacles = []
    last_radar_seq = -1

    for i, ekf_row in ekf_df.iterrows():
        t = ekf_row['unix_time']

        # 立刻进行时间戳对齐

        # LiDAR时间戳对齐
        t_ns = t * 1e9  # 同步和lidar文件名字一样的纳秒
        f_match, front_idx = find_closest_file(t_ns, front_files, front_idx)
        p_match, port_idx = find_closest_file(t_ns, port_files, port_idx)
        s_match, starboard_idx = find_closest_file(t_ns, starboard_files, starboard_idx)

        # Radar时间戳对齐，因为radar采集不全，需要单独列出
        radar_image, radar_seq = find_closest_radar_image(
            t, radar_times, radar_seqs, radar_image_dir
        )

        # 判断所属区域
        region = classify_region(port_idx)

        # 主循环中，只在lidar帧变化时才重新进行一次处理
        if port_idx != last_port_idx:
            last_port_idx = port_idx

            # 重置缓存
            last_walls = []
            last_lidar_obstacles = []

            # 读取找到的三帧的点云
            front_pts = read_lidar_bin(f_match)
            port_pts = read_lidar_bin(p_match)
            starboard_pts = read_lidar_bin(s_match)

            # 坐标变换，转到AHRS坐标系
            front_pts_ahrs = transform_points(front_pts, R_front, t_front)
            port_pts_ahrs = transform_points(port_pts, R_port, t_port)
            starboard_pts_ahrs = transform_points(starboard_pts, R_stbd, t_stbd)

            # 运河
            if region == 'canal':
                last_walls = detect_walls_one_frame(
                front_pts_ahrs, port_pts_ahrs, starboard_pts_ahrs
                )

            # 内港
            elif region == 'inner_port':
                # 额外加载ekf姿态进行水平补偿
                ekf_state = {'roll': ekf_row['roll'], 'pitch': ekf_row['pitch'],}
                last_lidar_obstacles = lidar_object_detect_one_frame(
                    front_pts_ahrs, port_pts_ahrs, starboard_pts_ahrs, ekf_state
                )

        # 外港&近海
        if radar_image is not None and (last_radar_image is None or radar_seq != last_radar_seq):
            last_radar_image = radar_image
            last_radar_seq = radar_seq
            last_radar_obstacles = radar_object_detect_one_frame(
                radar_image,
                cx = 1024, cy =1024,
                R_max = 1654.8, R_pixel = 1024,
                intensity_threshold = 80,
                num_sectors = 48
                )
        elif radar_image is None:
            last_radar_obstacles = []

        # 组装这一时刻的感知输出行
        row = {
            'unix_time': t,
            'x': ekf_row['x'],
            'y': ekf_row['y'],
            'z': ekf_row['z'],
            'roll': ekf_row['roll'],
            'pitch': ekf_row['pitch'],
            'yaw': ekf_row['yaw'],
            'vx': ekf_row['vx'],
            'vy': ekf_row['vy'],
            'speed': ekf_row['speed'],
            # 墙壁数据左右各取第一条
            'wall_left_cx': last_walls[0]['center_x'] if len(last_walls) > 0 else None,
            'wall_left_cy': last_walls[0]['center_y'] if len(last_walls) > 0 else None,
            'wall_left_angle': last_walls[0]['angle'] if len(last_walls) > 0 else None,
            'wall_left_length': last_walls[0]['length'] if len(last_walls) > 0 else None,
            'wall_right_cx': last_walls[1]['center_x'] if len(last_walls) > 1 else None,
            'wall_right_cy': last_walls[1]['center_y'] if len(last_walls) > 1 else None,
            'wall_right_angle': last_walls[1]['angle'] if len(last_walls) > 1 else None,
            'wall_right_length': last_walls[1]['length'] if len(last_walls) > 1 else None,
            # 障碍物数据存储为字符串
            'lidar_obstacles': str(last_lidar_obstacles),
            'radar_obstacles': str(last_radar_obstacles),
        }
        output_rows.append(row)

        if i % 1000 == 0:
            print(f"处理到EKF帧 {i} / {len(ekf_df)}，当前区域：{region}")

        # 最终保存
    result_df = pd.DataFrame(output_rows)
    result_df.to_csv("perception_results.csv", index = False)
    print(f"感知输出已保存到 perception_output.csv，共 {len(result_df)} 行")
if __name__ == "__main__":
    main()