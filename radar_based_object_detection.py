"""
基于radar的障碍物检测，用于外港和近海区域
"""

"""
两个主要的函数：检测函数和调试函数
调试函数分为几个部分：
一，将笛卡尔坐标的像素点转为极坐标网格，并进行强度过滤
二，对于每个角度的扇区，选最近的点作为障碍物
三，将极坐标重新转回与原来radar图像一样的笛卡尔坐标
四，返回障碍物中心列表
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# 将原始雷达图像转换成极坐标网格，并且只保留强度高于阈值的像素点
def build_polar_grid(image, cx, cy, R_max, R_pixel, intensity_threshold):
    # image是2D灰度图，像素值0-255
    # 返回points，每行都是距离和方位角

    # 筛选出所有强度高于阈值的像素坐标
    intensity_mask = image > intensity_threshold
    rows, cols = np.where(intensity_mask)

    if len(rows) == 0:
        return np.empty((0,2)) # 稳健处理

    # 每个像素点相对于图像中心的偏移
    dx= cols - cx
    dy= cy - rows

    pixel_dist = np.sqrt(dx**2 + dy**2) # 像素距离
    range_m = (pixel_dist / R_pixel) * R_max # 物理距离

    bearing_rad = np.arctan2(dy, dx)

    # 组合成（距离，方位角）数组
    points = np.column_stack((range_m, bearing_rad))
    return points

# 对输入的极坐标points拆分成不同扇区，并选出最近的点
def select_closest_per_sector(points, num_sectors):
    # num_sectors是扇区数目
    # 返回的是一个M*2的数组，每个扇区最多一个点
    ranges = points[:,0]
    bearings = points[:,1]
    bearings_pos = bearings % (2 * np.pi) # 把方位角映射到0到2Π，之前也用过这个

    # 计算每个点的扇区
    sector_width = 2 * np.pi / num_sectors
    sector_idx = np.floor(bearings_pos / sector_width).astype(int)
    sector_idx = np.clip(sector_idx, 0, num_sectors - 1) # 截断函数，把小于min的设为min，大于max的设为max，防止可能的越界

    # 每个扇区里找出距离最小的点的索引
    df = pd.DataFrame({'range': ranges, 'sector': sector_idx})
    idx_min = df.groupby('sector')['range'].idxmin().values
    closest = points[idx_min]
    closest = closest[closest[:,0] >= 0.1]
    return closest

# 将极坐标系重新转回笛卡尔坐标系
def polar_to_cartesian(ranges, bearings):
    # 非常简单的函数，注意数据格式即可
    x = ranges * np.cos(bearings)
    y = ranges * np.sin(bearings)
    return np.column_stack((x,y))

# 可视化
def draw_radar_detection(image, centers, frame_idx, cx, cy, R_max, R_pixel,
                         output_dir="radar_frames"):
    """
    在原始雷达图像上叠加检测到的障碍物红圈。

    参数:
        image:      2D 灰度图 (H, W)
        centers:    障碍物中心列表 [(x, y), ...] 米制坐标（AHRS坐标系）
        frame_idx:  当前帧序号
        cx, cy:     图像中心像素坐标
        R_max:      雷达最大探测距离 (米)
        R_pixel:    图像半径对应的像素数
        output_dir: 图片保存目录
    """
    import os
    from matplotlib.patches import Circle

    os.makedirs(output_dir, exist_ok=True)

    # 1. 画雷达灰度图
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray', origin='upper')
    ax = plt.gca()

    # 2. 画每个障碍物的红圈
    for x_m, y_m in centers:
        # 2.1 米制坐标 → 距离和方位角
        range_m = np.sqrt(x_m**2 + y_m**2)
        bearing = np.arctan2(y_m, x_m)

        # 2.2 距离 → 像素距离
        pixel_dist = range_m / R_max * R_pixel

        # 2.3 像素偏移量（注意 y 轴翻转：物理 y 向上，图像 y 向下）
        dx = pixel_dist * np.cos(bearing)
        dy = -pixel_dist * np.sin(bearing)

        # 2.4 图像像素坐标
        col = cx + dx
        row = cy + dy

        # 2.5 安全半径 D_R = 7.0 米 → 像素半径
        radius_pixel = 7.0 / R_max * R_pixel * 5.0 # 仅用于可视化的放大

        # 2.6 画简洁红圈（空心，线宽适中）
        circle = Circle((col, row), radius=radius_pixel,
                        fill=False, edgecolor='red', linewidth=1.5)
        ax.add_patch(circle)

    # 3. 设置标题，隐藏坐标轴
    plt.title(f'Frame {frame_idx}: {len(centers)} obstacles', fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    # 4. 保存图片
    filename = os.path.join(output_dir, f"radar_{frame_idx:05d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# 检测函数，供主程序调用
def radar_object_detect_one_frame(image,cx,cy,R_max,R_pixel,
                                  intensity_threshold,num_sectors,min_range=20.0):
    # image是2D灰度图，cx，cy是图像中心
    # R_max是雷达最大探测距离，R_pixel是图像半径对应像素数
    # intensity_threshold是强度阈值，低于它的点都会滤去
    # num_sectors是扇区数量

    # 将原始雷达图像像素点转换成极坐标网格，并且只保留强度高于阈值的像素点
    points = build_polar_grid(image, cx, cy, R_max, R_pixel, intensity_threshold)

    # 稳健处理，之前有过类似措施
    if len(points) == 0:
        return []

    # 先过滤掉自身附近的点防止误检
    valid_mask = points[:, 0] >= min_range
    points = points[valid_mask]

    # 把360°分成num_sectors个扇区，对每个扇区，挑选最近的点
    closest = select_closest_per_sector(points, num_sectors)

    # 接下来要重新转回笛卡尔坐标
    ranges = closest[:,0]
    bearings = closest[:,1]
    xy = polar_to_cartesian(ranges,bearings)

    # 最后返回障碍物中心列表
    obstacle_centers = [(xy[i,0], xy[i,1]) for i in range(xy.shape[0])]
    return obstacle_centers

# 主函数，调试用
def main():
    # 几个文件地址
    RADAR_DIR = Path(r"E:\PohangCanalDataset\radar")
    IMAGE_DIR = RADAR_DIR / "images"
    # TIMESTAMP_FILE = RADAR_DIR / "timestamp.txt"
    # 实际这个模块用不到时间戳，暂时不用读取

    # 论文给出的雷达参数
    R_MAX = 1654.8
    R_PIXEL = 1024
    CX,CY = 1024, 1024
    INTENSITY_THRESHOLD = 80
    NUM_SECTORS = 48
    MIN_RANGE = 20.0 # 最小探测距离，过滤圆心的杂波

    # # 测试第一张
    # timestamps = pd.read_csv(TIMESTAMP_FILE, sep = '\t',
    #                          header = None, names = ['unix_time', 'seq'])
    # seq_int = int(timestamps['seq'].iloc[0]) #  默认浮点数，要转为int
    # first_image_name = f"{seq_int:06d}.png"
    # image_path = IMAGE_DIR / first_image_name

    # # 读取图像，也就是灰度
    # from PIL import Image
    # img = Image.open(image_path).convert('L')
    # image = np.array(img)
    #
    # print(f"第一张雷达图像尺寸： {image.shape}")
    # print(f"第一张雷达图像强度范围： {image.min()} - {image.max()}")
    #
    # # 调用检测函数
    # obstacles = radar_object_detect_one_frame(
    #     image,CX,CY,R_MAX,R_PIXEL,INTENSITY_THRESHOLD,NUM_SECTORS
    # )
    #
    # # 打印结果
    # print(f"检测到 {len(obstacles)} 个障碍物")
    # for i, (x, y) in enumerate(obstacles):
    #     print(f"障碍物 {i+1}：（{x:.2f},{y:.2f}）")
    #
    # # 可视化
    # draw_radar_detection(image, obstacles, frame_idx=0,
    #                      cx=1024, cy=1024,
    #                      R_max=1654.8, R_pixel=1024)

    # 加载时间戳（暂时不需要）
    # timestamps = pd.read_csv(TIMESTAMP_FILE, sep = '\t',
    #                          header = None, names = ['unix_time', 'seq'])
    # print(f"[INIT] 时间戳加载完成，共 {len(timestamps)} 帧")

    image_files = sorted(IMAGE_DIR.glob("*.png"))
    print(f"[INIT] 雷达图像数量： {len(image_files)}")

    # 数据容器，准备全部保存到csv中
    all_obstacles = []

    for i,image_path in enumerate(image_files):

        # 读取图像
        from PIL import Image
        img = Image.open(image_path).convert('L')
        image = np.array(img)

        # 检测
        obstacles = radar_object_detect_one_frame(
            image,CX,CY,R_MAX,R_PIXEL,
            intensity_threshold=INTENSITY_THRESHOLD,
            num_sectors=NUM_SECTORS,
            min_range=MIN_RANGE
        )

        # 收集存储
        for (cx,cy) in obstacles:
            all_obstacles.append({
                'frame': i,
                'center_x':cx,
                'center_y':cy,
            })

        # 每隔十帧打印进度
        if i % 10 == 0:
            print(f"[Frame {i:>5}] 检测到 {len(obstacles)} 个障碍物")

        # 每隔二十帧画图
        if i % 20 == 0:
            draw_radar_detection(image, obstacles, i,CX, CY, R_MAX, R_PIXEL)

    df_obstacles = pd.DataFrame(all_obstacles)
    df_obstacles.to_csv("radar_obstacles.csv", index=False)
    print(f"\n全量处理完成。共 {len(df_obstacles)} 条记录，已保存到 radar_obstacles.csv")



if __name__ == "__main__":
    main()