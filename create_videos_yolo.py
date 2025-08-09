import os
import cv2
import numpy as np
import argparse
import json
import re
from collections import defaultdict


def natural_sort_key(s):
    """
    提供自然排序的键，例如 '2' 会排在 '10' 之前。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def draw_yolo_labels_on_image(image_path, label_path, class_names, colors, show_label=True):
    """
    在单张图片上绘制YOLO格式的标注。

    :param show_label: 布尔值，如果为True，则绘制类别名称标签。
    返回: 绘制了标注的图片对象 (numpy array)。
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [Warning] Failed to read image: {image_path}")
        return None

    h, w, _ = image.shape

    # 如果标签文件不存在，直接返回原图
    if not os.path.exists(label_path):
        return image

    # 读取并解析标签文件
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # 解析YOLO格式数据
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])

            # 将归一化坐标转换为像素坐标
            abs_w = width * w
            abs_h = height * h
            x1 = int((x_center * w) - (abs_w / 2))
            y1 = int((y_center * h) - (abs_h / 2))
            x2 = int(x1 + abs_w)
            y2 = int(y1 + abs_h)

            # 获取颜色
            try:
                color = colors[class_id % len(colors)]
            except IndexError:
                color = (255, 255, 255)  # 默认白色

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # --- [核心修改] 根据 show_label 参数决定是否绘制文本 ---
            if show_label:
                try:
                    label = class_names[class_id]
                except IndexError:
                    label = f"Class_{class_id}"

                # 绘制标签文本背景
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image, (x1, y1 - 20), (x1 + text_w, y1), color, -1)

                # 绘制标签文本
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # --- [修改结束] ---

    return image


def visualize_and_create_videos(data_root, output_dir, fps, specific_folders=None, show_label=True):
    """
    主函数：遍历子目录，生成带标注的视频。
    :param show_label: 布尔值，控制是否在视频帧上显示标签。
    """
    # 1. 定义路径
    images_root = os.path.join(data_root, 'images')
    labels_root = os.path.join(data_root, 'labels')
    class_json_path = os.path.join(data_root, 'class.json')

    # 2. 检查路径是否存在
    if not os.path.isdir(images_root):
        print(f"Error: Images directory not found at '{images_root}'")
        return
    if not os.path.isdir(labels_root):
        print(f"Error: Labels directory not found at '{labels_root}'")
        return
    if not os.path.isfile(class_json_path):
        print(f"Error: Class file not found at '{class_json_path}'")
        return

    # 3. 加载类别信息
    with open(class_json_path, 'r') as f:
        class_info = json.load(f)
    class_names = class_info['names']
    print(f"Loaded {len(class_names)} classes: {class_names}")

    # 4. 定义颜色
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128)
    ]

    # 5. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Videos will be saved to: '{output_dir}'")

    # 6. 确定要遍历的data子目录
    if specific_folders:
        sub_dirs = sorted(specific_folders, key=natural_sort_key)
        print(f"Only processing specified folders: {sub_dirs}")
    else:
        print("No specific folders provided, scanning all sub-directories...")
        sub_dirs = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])

    for sub_dir_name in sub_dirs:
        image_folder = os.path.join(images_root, sub_dir_name)
        if not os.path.isdir(image_folder):
            print(f"\n[Warning] Specified folder '{sub_dir_name}' not found in images directory. Skipping.")
            continue

        print(f"\nProcessing video for folder: '{sub_dir_name}'...")
        label_folder = os.path.join(labels_root, sub_dir_name)
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg','bmp'))],
            key=natural_sort_key
        )

        if not image_files:
            print(f"  [Warning] No images found in '{image_folder}'. Skipping.")
            continue

        first_image_path = os.path.join(image_folder, image_files[0])
        first_image = cv2.imread(first_image_path)
        if first_image is None:
            print(f"  [Error] Failed to read the first image. Skipping '{sub_dir_name}'.")
            continue

        height, width, _ = first_image.shape
        video_path = os.path.join(output_dir, f"{sub_dir_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for i, image_name in enumerate(image_files):
            if (i + 1) % 50 == 0 or i == 0 or i == len(image_files) - 1:
                print(f"  Processing frame {i + 1}/{len(image_files)}: {image_name}")

            image_path = os.path.join(image_folder, image_name)
            label_path = os.path.join(label_folder, os.path.splitext(image_name)[0] + '.txt')

            # --- [核心修改] 将 show_label 参数传递给绘图函数 ---
            annotated_image = draw_yolo_labels_on_image(image_path, label_path, class_names, colors, show_label)

            if annotated_image is not None:
                video_writer.write(annotated_image)

        video_writer.release()
        print(f"✅ Video for '{sub_dir_name}' successfully saved to '{video_path}'")


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO labels and create videos.")
    parser.add_argument('--data_root', type=str, default='./scene4',
                        help="Path to the root directory of the YOLO dataset (e.g., 'scene4').")
    parser.add_argument('--output_dir', type=str, default='./visualization_videos',
                        help="Directory to save the output videos.")
    parser.add_argument('--fps', type=int, default=30,
                        help="Frames per second for the output video.")
    parser.add_argument('--folders', nargs='+', default=None,
                        help="Specify which data folders to process (e.g., --folders data08 data09).")

    # --- [新增] 添加一个控制是否显示标签的开关 ---
    parser.add_argument('--no_label', action='store_true',
                        help="Do not draw class name labels on the bounding boxes.")

    args = parser.parse_args()

    # --- [修改] 将新参数传递给主函数 ---
    # `not args.no_label` 的意思是：如果命令中包含了 --no_label，则 args.no_label 为 True, 那么 show_label 就是 False。
    visualize_and_create_videos(args.data_root, args.output_dir, args.fps, args.folders, show_label=not args.no_label)


if __name__ == "__main__":
    main()