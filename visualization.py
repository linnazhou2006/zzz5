import json
import cv2
import os
import numpy as np
from collections import defaultdict


class COCOVideoGenerator:
    def __init__(self, annotation_file, image_base_path, output_dir="output_videos"):
        """
        初始化COCO视频生成器

        Args:
            annotation_file: COCO格式的annotation文件路径
            image_base_path: 图片的基础路径
            output_dir: 输出视频的目录
        """
        self.annotation_file = annotation_file
        self.image_base_path = image_base_path
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载COCO数据
        self.coco_data = self.load_coco_data()
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}

        # 按文件夹分组图片和标注
        self.grouped_data = self.group_by_folder()

        # 定义颜色映射（为不同类别分配不同颜色）
        self.colors = self.generate_colors()

    def load_coco_data(self):
        """加载COCO格式的annotation文件"""
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_actual_image_path(self, file_name):
        """查找实际的图片路径"""
        # 尝试多种可能的路径组合
        possible_paths = [
            os.path.join(self.image_base_path, file_name),  # 直接拼接
            os.path.join(self.image_base_path, file_name.replace('/', os.sep)),  # 处理路径分隔符
            file_name,  # 原始路径
            os.path.normpath(file_name),  # 标准化路径
        ]

        # 如果file_name包含子路径，尝试提取文件夹结构
        if '/' in file_name:
            parts = file_name.split('/')
            # 尝试不同的路径组合
            for i in range(len(parts)):
                sub_path = '/'.join(parts[i:])
                possible_paths.append(os.path.join(self.image_base_path, sub_path))
                possible_paths.append(os.path.join(self.image_base_path, sub_path.replace('/', os.sep)))

        # 查找存在的路径
        for path in possible_paths:
            if os.path.exists(path):
                return path

        # 如果都不存在，尝试在当前目录及子目录中搜索
        base_name = os.path.basename(file_name)
        for root, dirs, files in os.walk('.'):
            if base_name in files:
                found_path = os.path.join(root, base_name)
                print(f"在 {found_path} 找到图片 {base_name}")
                return found_path

        return None

    def group_by_folder(self):
        """按data文件夹分组图片和对应的标注，并检查路径"""
        # 创建图片ID到图片信息的映射
        image_dict = {img['id']: img for img in self.coco_data['images']}

        # 创建图片ID到标注的映射
        annotations_dict = defaultdict(list)
        for ann in self.coco_data['annotations']:
            annotations_dict[ann['image_id']].append(ann)

        # 按data文件夹分组
        grouped = defaultdict(list)
        missing_files = []

        for img_id, img_info in image_dict.items():
            file_name = img_info['file_name']
            file_name = file_name.replace('/', '\\')
            file_name = os.path.join(r"D:\tiaozhanbei\code\dsfnet-trae\data\mydataset", file_name)

            # 查找实际的图片路径
            actual_path = self.find_actual_image_path(file_name)

            if actual_path is None:
                missing_files.append(file_name)
                continue

            # 🔧 关键修改：提取data文件夹名称用于分组
            data_folder_name = self.extract_data_folder_name(img_info['file_name'])

            if data_folder_name is None:
                print(f"警告：无法从 {img_info['file_name']} 提取data文件夹名称")
                continue

            # 更新图片信息中的实际路径
            img_info_copy = img_info.copy()
            img_info_copy['actual_path'] = actual_path

            grouped[data_folder_name].append({
                'image_info': img_info_copy,
                'annotations': annotations_dict[img_id]
            })

        # 报告找不到的文件
        if missing_files:
            print(f"警告: 找不到以下 {len(missing_files)} 个图片文件:")
            for i, file in enumerate(missing_files[:10]):  # 只显示前10个
                print(f"  {file}")
            if len(missing_files) > 10:
                print(f"  ... 还有 {len(missing_files) - 10} 个文件")

        # 按文件名排序每个文件夹中的图片
        for folder in grouped:
            grouped[folder].sort(key=lambda x: x['image_info']['file_name'])

        return grouped

    def extract_data_folder_name(self, file_path):
        """从文件路径中提取data文件夹名称"""
        # 🔧 关键修改：专门提取data文件夹

        # 标准化路径分隔符
        normalized_path = file_path.replace('\\', '/')
        path_parts = normalized_path.split('/')

        # 查找包含'data'的部分
        data_folder = None
        for part in path_parts:
            # 匹配 data + 数字的格式
            if part.startswith('data') and len(part) > 4:
                # 检查data后面是否跟数字
                suffix = part[4:]  # 去掉'data'前缀
                if suffix.isdigit():
                    data_folder = part
                    break

        if data_folder:
            return data_folder

        # 如果上面的方法没找到，尝试其他方式
        for part in path_parts:
            if 'data' in part.lower():
                return part

        # 最后的备选方案：使用包含图片的直接父目录
        if len(path_parts) >= 2:
            return path_parts[-2]  # 倒数第二个部分通常是直接父目录

        return None

    def generate_colors(self):
        """为不同类别生成不同颜色"""
        colors = {}
        np.random.seed(42)  # 固定随机种子以确保颜色一致

        for cat_id in self.categories:
            colors[cat_id] = tuple(map(int, np.random.randint(0, 255, 3)))

        return colors

    def draw_annotations(self, image, annotations):
        """在图片上绘制标注"""
        for ann in annotations:
            category_id = ann['category_id']
            category_name = self.categories[category_id]['name']
            color = self.colors[category_id]

            # 绘制边界框
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)

                # 绘制矩形框
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                # 绘制类别标签
                label = f"{category_name}"
                if 'score' in ann:  # 如果有置信度分数
                    label += f": {ann['score']:.2f}"

                # 计算文本尺寸
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

                # 绘制文本背景
                cv2.rectangle(image, (x, y - text_height - 10),
                              (x + text_width, y), color, -1)

                # 绘制文本
                cv2.putText(image, label, (x, y - 5), font, font_scale,
                            (255, 255, 255), thickness)

        return image

    def create_video_for_folder(self, folder_name, fps=30):
        """为指定data文件夹创建视频"""
        if folder_name not in self.grouped_data:
            print(f"警告：data文件夹 '{folder_name}' 在annotation中未找到")
            return False

        folder_data = self.grouped_data[folder_name]
        if not folder_data:
            print(f"警告：data文件夹 '{folder_name}' 中没有有效的图片数据")
            return False

        # 🔧 修改输出文件名，明确标识这是data文件夹
        output_path = os.path.join(self.output_dir, f"{folder_name}_video.mp4")

        print(f"正在为data文件夹 '{folder_name}' 创建视频...")
        print(f"找到 {len(folder_data)} 张图片")

        # 使用实际路径读取第一张图片
        first_image_path = folder_data[0]['image_info']['actual_path']
        print(f"第一张图片路径: {first_image_path}")

        first_image = cv2.imread(first_image_path)

        if first_image is None:
            print(f"错误：无法读取第一张图片 {first_image_path}")
            # 尝试读取其他图片
            for data in folder_data[:5]:  # 尝试前5张
                test_path = data['image_info']['actual_path']
                test_image = cv2.imread(test_path)
                if test_image is not None:
                    first_image = test_image
                    print(f"使用替代图片: {test_path}")
                    break

            if first_image is None:
                print("错误：无法读取任何图片")
                return False

        height, width = first_image.shape[:2]
        print(f"视频尺寸: {width} x {height}")

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            print(f"错误：无法创建视频文件 {output_path}")
            return False

        try:
            processed_frames = 0
            for i, data in enumerate(folder_data):
                image_info = data['image_info']
                annotations = data['annotations']

                # 使用实际路径读取图片
                image_path = image_info['actual_path']
                image = cv2.imread(image_path)

                if image is None:
                    print(f"警告：无法读取图片 {image_path}，跳过")
                    continue

                # 确保图片尺寸一致
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height))

                # 绘制标注
                annotated_image = self.draw_annotations(image.copy(), annotations)

                # 🔧 修改帧信息显示，包含data文件夹名称
                frame_info = f"{folder_name} | Frame: {i + 1}/{len(folder_data)} | Objects: {len(annotations)}"
                cv2.putText(annotated_image, frame_info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 添加背景矩形使文字更清晰
                text_size = cv2.getTextSize(frame_info, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated_image, (8, 8), (text_size[0] + 12, 40), (0, 0, 0), -1)
                cv2.putText(annotated_image, frame_info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 写入视频
                video_writer.write(annotated_image)
                processed_frames += 1

                if (i + 1) % 50 == 0:
                    print(f"已处理 {i + 1}/{len(folder_data)} 张图片")

        finally:
            video_writer.release()

        if processed_frames > 0:
            print(f"✅ 视频创建完成: {output_path}")
            print(f"实际处理帧数: {processed_frames}")
            print(f"视频信息: {processed_frames} 帧, {fps} FPS, 时长约 {processed_frames / fps:.2f} 秒")
            return True
        else:
            print("错误：没有成功处理任何图片")
            return False

    def create_all_videos(self, fps=30):
        """为所有data文件夹创建视频"""
        print("🎬 开始为每个data文件夹创建视频...")

        # 按data文件夹名称排序
        sorted_folders = sorted(self.grouped_data.keys(), key=lambda x: (len(x), x))
        print(f"找到以下data文件夹: {sorted_folders}")
        print("=" * 60)

        success_count = 0
        for i, folder_name in enumerate(sorted_folders, 1):
            print(f"📁 处理第{i} / {len(sorted_folders)}个文件夹: {folder_name}")
            if self.create_video_for_folder(folder_name, fps):
                success_count += 1
            print("-" * 60)

            print(f"🎉 完成！成功创建{success_count} / {len(sorted_folders)}个视频")
            print(f"📁 输出目录: {self.output_dir}")

            # 列出生成的视频文件
            if success_count > 0:
                print(" 生成的视频文件: ")
            video_files = [f for f in os.listdir(self.output_dir) if f.endswith('.mp4')]
            for video_file in sorted(video_files):
                file_path = os.path.join(self.output_dir, video_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  {video_file} ({file_size:.1f} MB)")

    def create_videos_for_specific_folders(self, folder_names, fps=30):
        """为指定的data文件夹创建视频"""
        print(f"🎬 为指定的data文件夹创建视频: {folder_names}")

        success_count = 0
        for folder_name in folder_names:
            if folder_name in self.grouped_data:
                print(f"📁 处理文件夹: {folder_name}")
                if self.create_video_for_folder(folder_name, fps):
                    success_count += 1
                else:
                    print(f"⚠️  文件夹 '{folder_name}' 未找到")
                print("-" * 60)

                print(f"🎉 完成！成功创建{success_count} / {len(folder_names)}个视频 ")

    def print_statistics(self):
        """打印数据统计信息"""
        print("📊 数据统计:")
        print(f"总图片数: {len(self.coco_data['images'])}")
        print(f"总标注数: {len(self.coco_data['annotations'])}")
        print(f"类别数: {len(self.categories)}")

        print("📋 类别信息: ")
        for cat_id, cat_info in self.categories.items():
            print(f"  {cat_id}: {cat_info['name']}")

        print(f"📁 Data文件夹分布({len(self.grouped_data)}个):")
        sorted_folders = sorted(self.grouped_data.keys(), key=lambda x: (len(x), x))

        total_images = 0
        total_annotations = 0

        for folder in sorted_folders:
            data = self.grouped_data[folder]
            folder_annotations = sum(len(item['annotations']) for item in data)
            total_images += len(data)
            total_annotations += folder_annotations
            print(f"  {folder}: {len(data)} 张图片, {folder_annotations} 个标注")

        print(f"✅ 汇总: {len(sorted_folders)}个data文件夹, {total_images}张图片, {total_annotations}个标注")


def main():
    # 配置参数
    annotation_file = r"D:\tiaozhanbei\code\dsfnet-trae\data\mydataset\annotations\instances_train2017.json"  # 🔧 修改文件名
    image_base_path = "."  # 图片的基础路径
    output_dir = r"D:\tiaozhanbei\code\dsfnet-trae\data\mydataset\train\data_videos"  # 🔧 修改输出目录名
    fps = 30  # 视频帧率

    # 检查文件是否存在
    if not os.path.exists(annotation_file):
        print(f"❌ 错误：annotation文件 '{annotation_file}' 不存在！")
        return

    print("🔍 正在搜索图片文件...")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"图片基础路径: {image_base_path}")
    print(f"Annotation文件: {annotation_file}")

    # 创建视频生成器
    generator = COCOVideoGenerator(annotation_file, image_base_path, output_dir)

    # 打印统计信息
    generator.print_statistics()
    print("=" * 80)

    # 选择处理方式
    process_all = True  # 设置为False可以只处理特定文件夹

    if process_all:
        # 为所有data文件夹创建视频
        generator.create_all_videos(fps)
    else:
        # 只为特定的data文件夹创建视频（示例）
        specific_folders = ['data1', 'data30', 'data60', 'data90']  # 测试集文件夹
        generator.create_videos_for_specific_folders(specific_folders, fps)


if __name__ == "__main__":
    main()
