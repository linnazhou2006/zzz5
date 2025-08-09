# organize_yolo_dataset.py

import os
import shutil

# --- 可配置变量 ---
# 源目录：包含 'data01', 'data02' 等文件夹的路径
# '.' 表示当前目录
source_dir = r"D:\tiaozhanbei\code\dsfnet-trae\data\dif\scene2processed_data\processed_data\scene2frame_diff"

# 目标目录：将要创建的YOLO数据集的根文件夹名称
dest_dir = r"D:\tiaozhanbei\code\dsfnet-trae\data\dif\scene2processed_data\processed_data\scene2"

# 图片文件的扩展名列表
image_extensions = ('.bmp', '.jpg', '.jpeg', '.png')
# --- 配置结束 ---

# 创建目标根目录
os.makedirs(dest_dir, exist_ok=True)

# 遍历源目录中的条目
for item_name in os.listdir(source_dir):
    source_item_path = os.path.join(source_dir, item_name)

    # 检查是否是目录并且以 'data' 开头
    if os.path.isdir(source_item_path) and item_name.startswith('data'):

        # 定义目标子目录路径
        dest_images_path = os.path.join(dest_dir, 'images', item_name)
        dest_labels_path = os.path.join(dest_dir, 'labels', item_name)

        # 创建目标子目录
        os.makedirs(dest_images_path, exist_ok=True)
        os.makedirs(dest_labels_path, exist_ok=True)

        # 遍历 'dataXX' 文件夹中的所有文件
        for filename in os.listdir(source_item_path):
            source_file_path = os.path.join(source_item_path, filename)

            if not os.path.isfile(source_file_path):
                continue

            # 根据文件扩展名移动文件
            if filename.lower().endswith(image_extensions):
                shutil.move(source_file_path, os.path.join(dest_images_path, filename))
            elif filename.lower().endswith('.txt'):
                shutil.move(source_file_path, os.path.join(dest_labels_path, filename))

print(f"文件整理完成。数据集已创建于 '{dest_dir}' 目录。")