"""
感知定位第二部分：
使用三个lidar点云和calibration外参和内参数据，基于Hough变换拟合进行墙壁检测
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.spatial.transform import Rotation
from skimage.transform import hough_line, hough_line_peaks
import pandas as pd
from skimage.morphology import dilation

"""
***************************
准备工作，所有要用到的函数都在这里
***************************
"""
# 外参读取
def load_extrinsics(json_path):
    # 读取extrinsics.json，返回外参
    # 输入路径，返回字典，键为传感器名称，值为字典，包括平移向量和旋转矩阵
    # 好消息：json文件格式与python的字典结构天然对应，所以实现非常简单
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

# 点云投影函数
def project(points,x_range,y_range,resolution):
    # resolution是分辨率，单位是米/像素
    # 网格初始化
    x_min, x_max = x_range  # x_range和y_range是两个数组，解包出最大最小值
    y_min, y_max = y_range
    cols = int(np.ceil((x_max - x_min) / resolution))
    rows = int(np.ceil((y_max - y_min) / resolution))
    grid = np.zeros((rows, cols), dtype=np.uint16)
    # 筛选填入
    x, y = points[:, 0], points[:, 1]
    in_range = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)  # 和之前一样的布尔索引进行筛选
    x_in = x[in_range]
    y_in = y[in_range]
    col_idx = np.floor((x_in - x_min) / resolution).astype(int)
    row_idx = np.floor((y_max - y_in) / resolution).astype(int)  # matplotlib把第零行放在最上面，所以y轴需要翻转
    grid[row_idx,col_idx] = 1  # 累加每个格子的点数
    # 计算网格边界物理坐标，方便后续画图
    x_edges = np.linspace(x_min, x_max, cols + 1)  # 生成的是一个数组，元素为分界线的坐标
    y_edges = np.linspace(y_min, y_max, rows + 1)
    return grid, x_edges, y_edges

# Hough拟合，用ρ，θ两个参数表示一条直线，每个点都对应参数空间的一条曲线，找出重叠交点比较多的位置就是要拟合的直线
def hough_wall_detect(grid,x_edges,y_edges,min_length):
    # grid是待处理的原始数据网格，x_edges,y_edges是两个方向网格边界的物理坐标，在上面的投影函数中顺便求出的
    # min_length是最小线段长度，小于此长度的线段被丢弃
    # 返回lines是一个列表，每个元素是一条线段两端点的物理坐标

    # Hough变换得到参数空间（实际的大量计算全部被包括在了这个函数里，非常方便）
    hspace, angles, dists = hough_line(grid)

    # 找出几个峰值
    _, theta_peaks, rho_peaks = hough_line_peaks(
        hspace,angles,dists,
        threshold = 0.5
    )

    # 把上一行中找出的距离和角度转换成线段端点
    lines = []
    for rho, theta in zip(rho_peaks, theta_peaks):
        x_min = x_edges[0]
        x_max = x_edges[-1]
        y_min = y_edges[0]
        y_max = y_edges[-1]

        # 因为x*cosθ+y*sinθ=ρ会涉及对三角函数的除法，所以必须要讨论数值或者水平
        points = []
        if abs(np.sin(theta)) > 1e-6:
            y_at_xmin = (rho - x_min * np.cos(theta)) / np.sin(theta) # 这里，之前打错了变量名，耽搁了很久
            y_at_xmax = (rho - x_max * np.cos(theta)) / np.sin(theta)

            if y_min <= y_at_xmin <= y_max:
                points.append((x_min, y_at_xmin))
            if y_min <= y_at_xmax <= y_max:
                points.append((x_max, y_at_xmax))

        if abs(np.cos(theta)) > 1e-6:
            x_at_ymin = (rho - y_min * np.sin(theta)) / np.cos(theta)
            x_at_ymax = (rho - y_max * np.sin(theta)) / np.cos(theta)

            if x_min <= x_at_ymin <= x_max:
                points.append((x_at_ymin, y_min))
            if x_min <= x_at_ymax <= x_max:
                points.append((x_at_ymax, y_max))

        if len(points) >= 2:
            (x1,y1),(x2,y2) = points[0],points[1]
            seg_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if seg_length >= min_length:
                lines.append((x1,y1,x2,y2))
        # 这段代码很精妙，尽管数学上就是一个很简单的问题，但是通过讨论角度大小和直线-矩形交点的xy轴范围，巧妙使得points基本都是2个数据

    return lines

# 最后一个函数是把拟合出的线段参数转换成论文公式的形式,，加上了角度几何约束
def parameterize_lines(lines,angle_tol_deg = 15):
    angle_tol = np.radians(angle_tol_deg)
    # 输出列表，每个元素是一个字典
    segments = []
    for (x1,y1,x2,y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy,dx)
        angle_mod = angle % np.pi
        if angle_mod < angle_tol or np.pi - angle_mod < angle_tol:
            segments.append({
                'center_x': (x1 + x2) / 2.0,
                'center_y': (y1 + y2) / 2.0,
                'angle': angle,
                'length': length,
            })
    return segments

# 用作回退的保险函数，最基本的转换
def parameterize_lines_basic(lines):
    segments = []
    for (x1, y1, x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)
        segments.append({
            'center_x': (x1 + x2) / 2.0,
            'center_y': (y1 + y2) / 2.0,
            'angle': angle,
            'length': length,
        })
    return segments

# 一个延长短拟合直线的函数
# def extrapolate_to_x_bounds(seg, x_min, x_max):
#     cx, cy = seg['center_x'], seg['center_y']
#     angle = seg['angle']
#     if abs(np.cos(angle)) > 1e-6:
#         dx_left = (x_min - cx) / np.cos(angle)
#         dx_right = (x_max - cx) / np.cos(angle)
#         x1, y1 = x_min, cy + dx_left * np.sin(angle)
#         x2, y2 = x_max, cy + dx_right * np.sin(angle)
#     else:
#         x1, y1 = x_min, cy
#         x2, y2 = x_max, cy
#     new_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#     return {'center_x': (x1 + x2) / 2.0, 'center_y': (y1 + y2) / 2.0,
#             'angle': angle, 'length': new_length}


# 可视化，直接搬的llm的代码
def draw_wall_2d(segments, merged, left_pts, right_pts, frame_idx, output_dir="wall_frames"):
    """
    画三张子图：左侧点云+左墙、右侧点云+右墙、合并点云+全部墙。
    图片自动保存到 output_dir 文件夹，不弹出窗口，程序可连续运行。

    参数:
        segments:   当前帧检测到的墙壁线段列表（每个是字典：center_x, center_y, angle, length）
        merged:     合并后的点云 (N, 3)
        left_pts:   左墙点云 (N, 3)
        right_pts:  右墙点云 (N, 3)
        frame_idx:  当前帧序号
        output_dir: 图片保存目录，默认 "wall_frames"
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 从 segments 中提取左右墙
    left_wall = next((s for s in segments if s['center_y'] > 0), None)
    right_wall = next((s for s in segments if s['center_y'] < 0), None)

    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # ---------- 左图：左侧点云 + 左墙 ----------
    ax = axes[0]
    if len(left_pts) > 0:
        ax.scatter(left_pts[:, 0], left_pts[:, 1], s=1, c=left_pts[:, 2], cmap='viridis')
    if left_wall is not None:
        cx, cy = left_wall['center_x'], left_wall['center_y']
        dx = 0.5 * left_wall['length'] * np.cos(left_wall['angle'])
        dy = 0.5 * left_wall['length'] * np.sin(left_wall['angle'])
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], 'r-', linewidth=2.5)
    ax.set_xlim(0, 50)
    ax.set_ylim(-20, 20)
    ax.set_title(f'Frame {frame_idx}: Left wall')
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---------- 中图：右侧点云 + 右墙 ----------
    ax = axes[1]
    if len(right_pts) > 0:
        ax.scatter(right_pts[:, 0], right_pts[:, 1], s=1, c=right_pts[:, 2], cmap='viridis')
    if right_wall is not None:
        cx, cy = right_wall['center_x'], right_wall['center_y']
        dx = 0.5 * right_wall['length'] * np.cos(right_wall['angle'])
        dy = 0.5 * right_wall['length'] * np.sin(right_wall['angle'])
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], 'r-', linewidth=2.5)
    ax.set_xlim(0, 50)
    ax.set_ylim(-20, 20)
    ax.set_title(f'Frame {frame_idx}: Right wall')
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---------- 右图：合并点云 + 左右墙 ----------
    ax = axes[2]
    ax.scatter(merged[:, 0], merged[:, 1], s=1, c=merged[:, 2], cmap='viridis', alpha=0.5)
    for wall in (left_wall, right_wall):
        if wall is None:
            continue
        cx, cy = wall['center_x'], wall['center_y']
        dx = 0.5 * wall['length'] * np.cos(wall['angle'])
        dy = 0.5 * wall['length'] * np.sin(wall['angle'])
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], 'r-', linewidth=2.5)
    ax.set_xlim(0, 50)
    ax.set_ylim(-20, 20)
    ax.set_title(f'Frame {frame_idx}: Merged + walls')
    ax.set_xlabel('X (forward) [m]')
    ax.set_ylabel('Y (left) [m]')
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 保存图片
    filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)   # 关闭图形，释放内存


"""
********
主函数框架
********
"""

def main():
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

    # 初始化一个空列表，用来保存所有帧的墙壁检测结果
    all_segments = []

    # 逐帧处理每一个文件
    front_idx = 0
    starboard_idx = 0
    for i, port_file in enumerate(port_files):
        # if i >= 2001:
        #     break # 控制语句

        # 时间同步,这一步进行了优化
        t_port = extract_timestamp(port_file)
        f_match, front_idx = find_closest_file(t_port, front_files, front_idx)
        s_match, starboard_idx = find_closest_file(t_port, starboard_files, starboard_idx)

        # 读取找到的三帧的点云
        front_pts = read_lidar_bin(f_match)
        port_pts = read_lidar_bin(port_file)
        starboard_pts = read_lidar_bin(s_match)

        # 坐标变换，转到AHRS坐标系
        front_pts_ahrs = transform_points(front_pts, R_front, t_front)
        port_pts_ahrs = transform_points(port_pts, R_port, t_port)
        starboard_pts_ahrs = transform_points(starboard_pts, R_starboard, t_starboard)

        # 合并三个点云，就像论文里说的那样，这样hough只需要对一个对象进行了
        merged = np.vstack([front_pts_ahrs, port_pts_ahrs, starboard_pts_ahrs])  # 垂直堆叠数组

        # 滤波，也就是过滤掉一些阈值以外的点
        merged = filter_by_height(merged, z_min=-1.0, z_max=5.0)
        merged = filter_by_range(merged, min_x=0.0, max_x=50.0, min_y=-20.0, max_y=20.0)

        # 左右分离，分别做hough变换
        left_pts = merged[merged[:,1] > 1.0]
        right_pts = merged[merged[:,1] < -1.0]

        segments_this_frame = []

        for side_pts,y_range_side in [
            (left_pts, (2.0, 15.0)),
            (right_pts, (-15.0, -2.0)),
        ]:
            if len(side_pts) < 50:
                continue

            # 动态确定 y_range,根据y的均值
            y_vals = side_pts[:,1]
            y_lo = np.percentile(y_vals, 10)
            y_hi = np.percentile(y_vals, 90)
            y_range_side = (y_lo - 1.0, y_hi + 1.0)
            # y_mean = np.mean(y_vals)
            # y_std = np.std(y_vals)
            # y_range_side = (y_mean - 2*y_std, y_mean + 2*y_std)
            # 投影到加权网格，准备Hough变换
            grid, x_edges, y_edges = project(
                side_pts,
                x_range = (0,30),
                y_range = y_range_side, # ！！！极其关键的一步！！！x和y的范围选取直接决定了投影矩阵的形状，也就决定了对哪个方向会更敏感
                resolution = 0.3
            )

            # grid_bin = grid > 0
            # grid_bin = dilation(grid_bin, np.ones((3, 3)))
            # grid_bin = dilation(grid_bin, np.ones((3, 3)))
            # grid_bin = grid_bin.astype(np.uint8)

            # Hough变换
            raw_lines = hough_wall_detect(grid, x_edges, y_edges,
                                          min_length = 3.92)

            # 角度优先，并且必须有回退防止没合适角度
            segs_angled = parameterize_lines(raw_lines)
            if segs_angled:
                longest = max(segs_angled, key=lambda s: s['length'])
                # longest = extrapolate_to_x_bounds(longest,0,50)
                segments_this_frame.append(longest)
            else:
                segs_basic = parameterize_lines_basic(raw_lines)
                if segs_basic:
                    longest = max(segs_basic, key=lambda s: s['length'])
                    # longest = extrapolate_to_x_bounds(longest, 0, 50)
                    segments_this_frame.append(longest)

        # 保存这一帧的结果
        for seg in segments_this_frame:
            all_segments.append({
                'frame': i,
                'center_x': seg['center_x'],
                'center_y': seg['center_y'],
                'angle': seg['angle'],
                'length': seg['length'],
            })

        if i % 100 == 0:
            print(f"[Frame {i:>5}] 检测到 {len(segments_this_frame)} 条墙")
            draw_wall_2d(segments_this_frame, merged, left_pts, right_pts, i)

    # ---- 保存结果 ----
    df = pd.DataFrame(all_segments)
    df.to_csv("wall_detection_results.csv", index=False)
    print(f"\n全量处理完成，共 {len(all_segments)} 条墙壁线段，已保存。")

if __name__ == "__main__":
    main()

