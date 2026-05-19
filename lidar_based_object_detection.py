"""基于LiDAR点云的障碍物检测，用于内港区域"""

"""
一共有两个主要的函数，一个是主函数main，用来调试检测函数，一个是检测函数，用来在主程序中调用
检测程序分为四个部分：
一，数据读取和坐标转换：读取lidar点云数据并转到ahrs坐标系，涉及两个函数
二，调用EKF的结果进行姿态补偿
三，滤波函数，一共有三个函数，滤除高度范围外，水平范围外，自身的点云
四，核心部分，欧几里得聚类，两个函数，一个进行DBSCAN聚类，一个计算每个簇的几何中心
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.spatial.transform import Rotation
import pandas as pd
from sklearn.cluster import DBSCAN

# 外参读取
def load_extrinsics(json_path):
    with open(json_path,'r') as f:
        extrinsics = json.load(f) # 这一步自动已经转换为字典了
    return extrinsics

# 根据外参构建坐标系变换矩阵
def build_transform(extrinsics_entry):
    # 输入外参条目，返回用来坐标系变换（从传感器坐标系到AHRS坐标系）的变换矩阵
    # 读取一个字典外参条目，返回3*3旋转矩阵R和3*1平移向量t
    # 旋转矩阵乘以传感器坐标加上平移向量等于AHRS下坐标
    q = extrinsics_entry['quaternion']
    t = np.array(extrinsics_entry['translation'])
    R = Rotation.from_quat(q).as_matrix() # .as_matrix方法可以把二维数组转换成3*3旋转矩阵
    # 原来数据集的json文件很有序，十分方便
    return R,t

# 提取时间戳的简单小函数
def extract_timestamp(filepath):
    # 输入文件路径，返回时间戳
    stem = filepath.stem # 获取不带后缀的文件名
    timestamp = int(stem)
    return timestamp

# 一个用来优化时间戳对齐的滑动窗口函数，原来llm告诉我的算法是暴力min方法，太慢了
def find_closest_file(target_timestamp,file_list,current_index):
    # 文件列表既然已经排好序，可以从current_index开始就近搜索时间戳最接近的文件，并且最多往右一步
    # 输入的参数如括号里所示，返回的是最接近的.bin文件和在file_list里的索引(0-based)
    if current_index >= len(file_list) - 1:
        return file_list[-1], len(file_list) - 1
    diff_cur = abs(extract_timestamp(file_list[current_index])-target_timestamp)
    diff_next = abs(extract_timestamp(file_list[current_index+1])-target_timestamp)

    if diff_cur > diff_next:
        return file_list[current_index + 1], current_index + 1
    else:
        return file_list[current_index], current_index

# 读取某一帧的点云
# 官方定义的数据结构（每个点 8 个字段，28 字节）
lidar_dtype = np.dtype([
    ('x',            np.float32),
    ('y',            np.float32),
    ('z',            np.float32),
    ('intensity',    np.float32),
    ('time',         np.uint32),
    ('reflectivity', np.uint16),
    ('ambient',      np.uint16),
    ('range',        np.uint32)
])
def read_lidar_bin(file_path):
    # 读取一个.bin文件，返回N×3的xyz点云数组
    scan = np.fromfile(file_path, dtype=lidar_dtype)
    points = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)
    return points

# 坐标变换函数，使用外参将传感器点云转到AHRS坐标系下，注意向量大小问题，points是N*3的矩阵
def transform_points(points,R,t):
    return (R @ points.T).T + t

# 两个滤波函数，分别过滤掉不在z高度阈值和x,y范围阈值的点
def filter_by_height(points,z_min,z_max):
    # 输出滤波后的N'*3点云，使用的方法很巧妙，用了numpy中的布尔索引
    z = points[:,2]
    z_filter = (z >= z_min) & (z <= z_max)
    return points[z_filter]

def filter_by_range(points,min_x,max_x,min_y,max_y):
    x = points[:,0]
    y = points[:,1]
    range_filter = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
    return points[range_filter]

# 滤除自身的点云
def remove_self_points(points, min_dist):
    dist = np.sqrt(points[:,0]**2 + points[:,1]**2)
    mask = dist >= min_dist
    return points[mask]

# 提取最接近指定时间戳的ekf姿态
def get_ekf_state_at_time(ekf_df, unix_time):
    idx = np.argmin(np.abs(ekf_df['unix_time'].values - unix_time))
    row = ekf_df.iloc[idx]
    return {'roll': row['roll'], 'pitch': row['pitch']}

# EKF姿态补偿函数
def compensate_attitude_from_ekf(points, roll, pitch):
    # 用ekf的roll和pitch，把点云旋转到平行水平面
    cos_r = np.cos(-roll)
    sin_r = np.sin(-roll)
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_r, -sin_r],
        [0.0, sin_r, cos_r]
    ])

    cos_p = np.cos(-pitch)
    sin_p = np.sin(-pitch)
    R_y = np.array([
        [cos_p, 0.0, sin_p],
        [0.0, 1.0, 0.0],
        [-sin_p, 0.0, cos_p]
    ])

    R_total = R_y @ R_x
    compensated = (R_total @ points.T).T
    return compensated

# 欧几里得聚类函数
def cluster_obstacles(points, eps, min_samples):
    # eps表示邻域距离，小于这个距离的会被当成同一个障碍物；min_samples表示归类为同一个障碍物所至少需要的点数。
    # 返回labels，也就是每个点的簇的标签，如果是-1就表示噪声，还返回n_clusters，表示簇的总数
    # 直接调用现成的DBSCAN，非常方便
    db = DBSCAN(eps = eps, min_samples = min_samples).fit(points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters

# 提取障碍物聚类的中心
def get_cluster_centers(points, labels, n_clusters):
    # 返回数组centers，内部元素是元组
    centers = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_pts = points[mask]
        cx = np.mean(cluster_pts[:,0])
        cy = np.mean(cluster_pts[:,1])
        centers.append((cx,cy))
    return centers

# 可视化函数，直接搬的llm
def draw_obstacle_detection(merged, centers, frame_idx, output_dir="obstacle_frames"):
    """
    画滤波后点云的俯视图，并用红色星号标记障碍物中心。
    图片自动保存到 output_dir 文件夹。

    参数:
        merged:    滤波后的点云 (N, 3)，AHRS坐标系
        centers:   障碍物中心列表 [(x1,y1), (x2,y2), ...]
        frame_idx: 当前帧序号
        output_dir: 图片保存目录
    """
    import os
    from matplotlib.patches import Circle

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))

    # 画点云俯视图（用高度上色）
    plt.scatter(merged[:, 0], merged[:, 1], s=1, c=merged[:, 2], cmap='viridis', alpha=0.5, label='LiDAR points')

    # 标记障碍物中心，并且画安全半径圆圈
    ax = plt.gca()
    for cx, cy in centers:
        # 中心点
        ax.plot(cx, cy, 'ro', markersize = 5)
        # 圆圈
        circle = Circle((cx,cy), radius=5.0, fill=False, color='red',linewidth=2,alpha=0.8)
        ax.add_patch(circle)

    plt.xlim(0, 80)
    plt.ylim(-40, 40)
    plt.xlabel('X (forward) [m]', fontsize=12)
    plt.ylabel('Y (left) [m]', fontsize=12)
    plt.title(f'Frame {frame_idx}: {len(centers)} obstacles detected', fontsize=14)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.colorbar(label='Height Z [m]')
    plt.tight_layout()

    filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# 检测函数，供主程序调用
def lidar_object_detect_one_frame(front_pts_ahrs, port_pts_ahrs, starboard_pts_ahrs, ekf_state):
    # 对一帧进行障碍物检测，返回每个障碍物的几何中心坐标，尺寸在table 3里给出了

    # 合并三个点云
    merged = np.vstack([front_pts_ahrs, port_pts_ahrs, starboard_pts_ahrs])

    #  内存不足，运行很慢，都导致需要进行下采样，下采样越狠，时间和空间都会更节省
    merged = merged[::10]

    # EKF姿态补偿
    merged = compensate_attitude_from_ekf(merged, ekf_state['roll'], ekf_state['pitch'])

    # 几个滤波
    merged = filter_by_height(merged, z_min = -1.0, z_max = 5.0)
    merged = filter_by_range(merged, min_x = 0.0, max_x = 80.0, min_y = -40.0, max_y = 40.0)
    merged = remove_self_points(merged, min_dist = 7.0) # 经过尝试，7.0m才能过滤自身

    # 欧几里得聚类
    if merged.shape[0] == 0:
        print("[WARN] 滤波后无点云，跳过本帧障碍物检测")
        return []
    labels, n_clusters = cluster_obstacles(merged, eps = 2.0, min_samples = 10)

    # 提取每个簇的几何中心，也就是x，y的均值
    centers = get_cluster_centers(merged, labels, n_clusters)

    return centers


# 主函数，共调试使用
def main():
    # 一个单独的主函数，只在这个程序中会被执行，用来调试

    # 加载外参
    ext_path = r"E:\PohangCanalDataset\calibration\extrinsics.json"
    extrinsics = load_extrinsics(ext_path)

    # 提取三个外参，得到旋转矩阵和平移向量
    R_front, t_front = build_transform(extrinsics['lidar_front'])
    R_port, t_port = build_transform(extrinsics['lidar_port'])
    R_starboard, t_starboard = build_transform(extrinsics['lidar_starboard'])

    print(f"[INIT] R_front shape: {R_front.shape}, t_front: {t_front}")
    print(f"[INIT] R_port shape: {R_port.shape}, t_port: {t_port}")
    print(f"[INIT] R_starboard shape: {R_starboard.shape}, t_starboard: {t_starboard}")

    # 用变量记一下LiDAR目录，方便自动读取全部数据
    front_dir = Path(r"E:\PohangCanalDataset\lidar\lidar_front\points")
    port_dir = Path(r"E:\PohangCanalDataset\lidar\lidar_port\points")
    starboard_dir = Path(r"E:\PohangCanalDataset\lidar\lidar_starboard\points")

    front_files = sorted(front_dir.glob("*.bin"))
    port_files = sorted(port_dir.glob("*.bin"))
    starboard_files = sorted(starboard_dir.glob("*.bin"))

    print(f"[INIT] front文件数目: {len(front_files)}")
    print(f"[INIT] port文件数目: {len(port_files)}")
    print(f"[INIT] starboard文件数目: {len(starboard_files)}")

    # 把输入放到循环外提速
    ekf_df = pd.read_csv("ekf_results.csv", sep=',')
    print(f"[INIT] EKF 轨迹加载完成，共 {len(ekf_df)} 帧")

    front_idx = 0
    starboard_idx = 0

    # 收集全部障碍物数据的容器
    all_obstacles = []

    for i,port_file in enumerate(port_files):

        # 时间对齐，和wall_detection一样的优化思路。但是注意！必须在跳过索引之前更新，否则将完全跳过同步，造成错位
        t_port = extract_timestamp(port_file)
        f_match, front_idx = find_closest_file(t_port, front_files, front_idx)
        s_match, starboard_idx = find_closest_file(t_port, starboard_files, starboard_idx)

        # 控制语句，只检测大概的inner_port段
        if i < 6000 or i > 15000:
            continue

        # 读取找到的三帧的点云
        front_pts = read_lidar_bin(f_match)
        port_pts = read_lidar_bin(port_file)
        starboard_pts = read_lidar_bin(s_match)

        # print(f"[DEBUG] 原始点数: front={front_pts.shape[0]}, port={port_pts.shape[0]}, stbd={starboard_pts.shape[0]}")

        # 坐标变换，转到AHRS坐标系
        front_pts_ahrs = transform_points(front_pts, R_front, t_front)
        port_pts_ahrs = transform_points(port_pts, R_port, t_port)
        starboard_pts_ahrs = transform_points(starboard_pts, R_starboard, t_starboard)

        # 这一步开始不一样了，需要调用EKF的数据得到对应帧的姿态
        ekf_state = get_ekf_state_at_time(ekf_df, t_port)
        # print(f"EKF 姿态: roll={np.degrees(ekf_state['roll']):.2f}°, "
        #       f"pitch={np.degrees(ekf_state['pitch']):.2f}°")

        # 调用主检测函数
        obstacles = lidar_object_detect_one_frame(
            front_pts_ahrs, port_pts_ahrs, starboard_pts_ahrs, ekf_state
        )

        # 每隔100帧画一幅图
        if i % 100 == 0:
            print(f"[Frame {i:>5}] 检测到 {len(obstacles)} 个障碍物")
            # 从检测函数里单独提取滤波后的点云用于画图
            # 这里简单起见，直接重新跑一次滤波逻辑得到 merged_filtered
            merged = np.vstack([front_pts_ahrs, port_pts_ahrs, starboard_pts_ahrs]) # 真实点图不进行下采样
            merged = compensate_attitude_from_ekf(merged, ekf_state['roll'], ekf_state['pitch'])
            merged = filter_by_height(merged, -1.0, 5.0)
            merged = filter_by_range(merged, 0.0, 80.0, -40.0, 40.0)
            merged = remove_self_points(merged, 5.0)
            draw_obstacle_detection(merged, obstacles, i)

        # 每一帧都要收集数据
        for (cx,cy) in obstacles:
            all_obstacles.append({
                'frame': i,
                'center_x': cx,
                'center_y': cy,
            })

        if i % 10 == 0:
            print(f"[Frame {i:>5}] 已完成")

    # 保存数据
    df_obstacles = pd.DataFrame(all_obstacles)
    df_obstacles.to_csv("lidar_obstacles.csv", sep=',', index=False)
    print(f"障碍物检测结果已保存到 lidar_obstacles.csv，共 {len(df_obstacles)} 条记录")
    print("\n处理完成。")

if __name__ == '__main__':
    main()