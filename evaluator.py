import os
import numpy as np
from collections import defaultdict
import json


class DetectionEvaluator:
    def __init__(self, config):
        """
        初始化评估器
        config: 配置字典，包含以下参数：
        - gt_root: 真实标签根目录
        - pred_root: 预测结果根目录
        - iou_threshold: IoU阈值
        - class_names: 类别名称列表（可选）
        - output_file: 结果保存文件名（可选）
        - consistency_iou_threshold: 时序一致性IoU阈值（默认0.3）
        - stability_threshold: 时空稳定性阈值（默认0.8，即80%）
        """
        self.gt_root = config['gt_root']
        self.pred_root = config['pred_root']
        self.iou_threshold = config['iou_threshold']
        self.class_names = config.get('class_names', None)
        self.output_file = config.get('output_file', 'evaluation_results.json')

        # 时序一致性和稳定性相关参数
        self.consistency_iou_threshold = config.get('consistency_iou_threshold', 0.3)
        self.stability_threshold = config.get('stability_threshold', 0.8)

        # 总体统计信息
        self.total_tp = defaultdict(int)
        self.total_fp = defaultdict(int)
        self.total_fn = defaultdict(int)
        self.total_gt_count = defaultdict(int)
        self.total_pred_count = defaultdict(int)

        # 每个视频的结果
        self.video_results = {}

    def parse_yolo_line(self, line):
        """解析YOLO格式的一行数据"""
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0

        return {
            'class_id': class_id,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height,
            'confidence': confidence
        }

    def load_yolo_file(self, file_path):
        """加载YOLO格式文件"""
        boxes = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # 跳过空行
                            box = self.parse_yolo_line(line)
                            if box is not None:
                                boxes.append(box)
            except Exception as e:
                print(f"警告: 读取文件 {file_path} 时出错: {e}")
        return boxes

    def get_sorted_txt_files(self, directory):
        """获取目录中所有txt文件并按文件名排序"""
        if not os.path.exists(directory):
            return []

        txt_files = []
        for file in os.listdir(directory):
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(directory, file))

        # 按文件名排序（自然排序）
        txt_files.sort(key=lambda x: self.natural_sort_key(os.path.basename(x)))
        return txt_files

    def natural_sort_key(self, filename):
        """自然排序的key函数，支持数字排序"""
        import re
        def convert(text):
            return int(text) if text.isdigit() else text.lower()

        return [convert(c) for c in re.split('([0-9]+)', filename)]

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        # 转换为左上角和右下角坐标
        x1_min = box1['x_center'] - box1['width'] / 2
        y1_min = box1['y_center'] - box1['height'] / 2
        x1_max = box1['x_center'] + box1['width'] / 2
        y1_max = box1['y_center'] + box1['height'] / 2

        x2_min = box2['x_center'] - box2['width'] / 2
        y2_min = box2['y_center'] - box2['height'] / 2
        x2_max = box2['x_center'] + box2['width'] / 2
        y2_max = box2['y_center'] + box2['height'] / 2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算并集
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def match_boxes(self, gt_boxes, pred_boxes):
        """匹配真实框和预测框"""
        matched_gt = set()
        matched_pred = set()
        matches = []

        # 按置信度排序预测框
        pred_boxes_sorted = sorted(enumerate(pred_boxes),
                                   key=lambda x: x[1]['confidence'], reverse=True)

        for pred_idx, pred_box in pred_boxes_sorted:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                # 只匹配相同类别的框
                if gt_box['class_id'] != pred_box['class_id']:
                    continue

                iou = self.calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                matches.append((best_gt_idx, pred_idx, best_iou))
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)

        return matches, matched_gt, matched_pred

    def match_boxes_for_consistency(self, gt_boxes, pred_boxes):
        """
        为时序一致性计算匹配真实框和预测框
        使用时序一致性的IoU阈值
        """
        matched_gt = set()
        matched_pred = set()
        matches = []

        # 按置信度排序预测框
        pred_boxes_sorted = sorted(enumerate(pred_boxes),
                                   key=lambda x: x[1]['confidence'], reverse=True)

        for pred_idx, pred_box in pred_boxes_sorted:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                # 只匹配相同类别的框
                if gt_box['class_id'] != pred_box['class_id']:
                    continue

                iou = self.calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # 使用时序一致性的IoU阈值
            if best_iou >= self.consistency_iou_threshold and best_gt_idx not in matched_gt:
                matches.append((best_gt_idx, pred_idx, best_iou))
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)

        return matches, matched_gt, matched_pred

    def check_frame_consistency(self, gt_boxes, pred_boxes):
        """
        检查单帧的时序一致性
        规则：
        - 当真值目标 <= 5个时，要求所有目标检出，且类别正确，且IoU > 0.3
        - 当真值目标 > 5个时，要求80%的目标检出，且类别正确，且IoU > 0.3
        """
        if len(gt_boxes) == 0:
            # 如果该帧没有真实目标，则认为是一致的（空帧）
            return True

        if len(pred_boxes) == 0:
            # 如果有真实目标但没有预测，则不一致
            return False

        # 使用时序一致性专用的匹配函数
        matches, matched_gt, matched_pred = self.match_boxes_for_consistency(gt_boxes, pred_boxes)

        # 计算检出的目标数量
        detected_targets = len(matched_gt)
        total_targets = len(gt_boxes)

        # 根据目标数量应用不同的一致性标准
        if total_targets <= 5:
            # 目标数量 <= 5：要求所有目标都被检出
            required_detections = total_targets
            is_consistent = detected_targets >= required_detections
        else:
            # 目标数量 > 5：要求80%的目标被检出
            required_detections = int(total_targets * 0.8)
            is_consistent = detected_targets >= required_detections

        return is_consistent

    def evaluate_frame_pair(self, gt_file, pred_file, video_stats):
        """评估一对对应的真实标签和预测结果文件"""
        gt_boxes = self.load_yolo_file(gt_file)
        pred_boxes = self.load_yolo_file(pred_file)

        # 检查时序一致性
        is_consistent = self.check_frame_consistency(gt_boxes, pred_boxes)
        video_stats['consistency_frames'] += 1 if is_consistent else 0
        video_stats['total_frames'] += 1

        # 记录详细的一致性统计信息（用于调试和分析）
        gt_count = len(gt_boxes)
        if gt_count > 0:
            if gt_count <= 5:
                video_stats['frames_with_targets_le5'] += 1
                if is_consistent:
                    video_stats['consistent_frames_le5'] += 1
            else:
                video_stats['frames_with_targets_gt5'] += 1
                if is_consistent:
                    video_stats['consistent_frames_gt5'] += 1

        # 按类别分组
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)

        for box in gt_boxes:
            gt_by_class[box['class_id']].append(box)
            video_stats['total_gt'][box['class_id']] += 1

        for box in pred_boxes:
            pred_by_class[box['class_id']].append(box)
            video_stats['total_pred'][box['class_id']] += 1

        # 获取所有出现的类别
        all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())

        for class_id in all_classes:
            gt_class_boxes = gt_by_class[class_id]
            pred_class_boxes = pred_by_class[class_id]

            if len(gt_class_boxes) == 0 and len(pred_class_boxes) == 0:
                continue

            # 匹配该类别的框（使用标准IoU阈值）
            matches, matched_gt, matched_pred = self.match_boxes(gt_class_boxes, pred_class_boxes)

            # 统计TP, FP, FN
            tp_count = len(matches)
            fp_count = len(pred_class_boxes) - len(matched_pred)
            fn_count = len(gt_class_boxes) - len(matched_gt)

            video_stats['tp'][class_id] += tp_count
            video_stats['fp'][class_id] += fp_count
            video_stats['fn'][class_id] += fn_count

    def calculate_metrics_for_stats(self, stats, use_macro_average=True):
        """
        根据统计数据计算指标
        use_macro_average: 是否使用宏平均计算总体指标
        """
        results = {}
        all_classes = set(stats['tp'].keys()) | set(stats['fp'].keys()) | set(stats['fn'].keys())

        # 用于宏平均的指标列表
        class_precisions = []
        class_recalls = []
        class_f1s = []
        class_false_alarm_rates = []

        # 用于微平均的总计数
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # 记录参与宏平均计算的类别（数据集中存在的类别）
        valid_classes = []

        for class_id in sorted(all_classes):
            tp = stats['tp'][class_id]
            fp = stats['fp'][class_id]
            fn = stats['fn'][class_id]

            # 计算召回率 (Recall)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # 计算精确率 (Precision)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # 计算虚警率 (False Alarm Rate = 1 - Precision)
            false_alarm_rate = 1.0 - precision

            # 计算F1分数
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            class_name = self.class_names[class_id] if self.class_names and class_id < len(
                self.class_names) else f"Class_{class_id}"

            results[class_id] = {
                'class_name': class_name,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'recall': recall,
                'precision': precision,
                'false_alarm_rate': false_alarm_rate,
                'f1': f1,
                'total_gt': stats['total_gt'][class_id],
                'total_pred': stats['total_pred'][class_id]
            }

            # 判断是否为数据集中存在的类别：TP + FN > 0（有真实目标的类别）
            if tp + fn > 0:  # 数据集中存在的类别
                class_precisions.append(precision)
                class_recalls.append(recall)
                class_f1s.append(f1)
                class_false_alarm_rates.append(false_alarm_rate)
                valid_classes.append(class_id)

            # 累计用于微平均
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # 计算总体指标
        if use_macro_average and len(class_precisions) > 0:
            # 宏平均：先计算各有效类别指标，再取平均
            overall_recall = sum(class_recalls) / len(class_recalls)
            overall_precision = sum(class_precisions) / len(class_precisions)
            overall_false_alarm_rate = sum(class_false_alarm_rates) / len(class_false_alarm_rates)
            overall_f1 = sum(class_f1s) / len(class_f1s)
            averaging_method = "macro"
            valid_classes_count = len(valid_classes)
        else:
            # 微平均：先累加所有TP/FP/FN，再计算指标
            overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            overall_false_alarm_rate = 1.0 - overall_precision
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (
                                                                                                                        overall_precision + overall_recall) > 0 else 0.0
            averaging_method = "micro"
            valid_classes_count = len(all_classes)

        # 计算时序一致性
        temporal_consistency = stats['consistency_frames'] / stats['total_frames'] if stats['total_frames'] > 0 else 0.0

        results['overall'] = {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'recall': overall_recall,
            'precision': overall_precision,
            'false_alarm_rate': overall_false_alarm_rate,
            'f1': overall_f1,
            'total_gt': sum(stats['total_gt'].values()),
            'total_pred': sum(stats['total_pred'].values()),
            'averaging_method': averaging_method,
            'valid_classes_count': valid_classes_count,
            'valid_classes': valid_classes if use_macro_average else list(all_classes),
            'temporal_consistency': temporal_consistency,
            'consistent_frames': stats['consistency_frames'],
            'total_frames': stats['total_frames'],
            # 添加详细的一致性统计
            'frames_with_targets_le5': stats.get('frames_with_targets_le5', 0),
            'consistent_frames_le5': stats.get('consistent_frames_le5', 0),
            'frames_with_targets_gt5': stats.get('frames_with_targets_gt5', 0),
            'consistent_frames_gt5': stats.get('consistent_frames_gt5', 0)
        }

        return results

    def evaluate_video(self, video_name):
        """评估单个视频的结果"""
        gt_video_path = os.path.join(self.gt_root, video_name)
        pred_video_path = os.path.join(self.pred_root, video_name)

        if not os.path.exists(gt_video_path):
            print(f"警告: 真实标签目录不存在: {gt_video_path}")
            return None

        if not os.path.exists(pred_video_path):
            print(f"警告: 预测结果目录不存在: {pred_video_path}")
            return None

        # 获取真实标签和预测结果的所有txt文件，按文件名排序
        gt_files = self.get_sorted_txt_files(gt_video_path)
        pred_files = self.get_sorted_txt_files(pred_video_path)

        if len(gt_files) == 0:
            print(f"警告: 视频 {video_name} 的真实标签目录中没有找到txt文件")
            return None

        if len(pred_files) == 0:
            print(f"警告: 视频 {video_name} 的预测结果目录中没有找到txt文件")
            return None

        # 初始化该视频的统计数据
        video_stats = {
            'tp': defaultdict(int),
            'fp': defaultdict(int),
            'fn': defaultdict(int),
            'total_gt': defaultdict(int),
            'total_pred': defaultdict(int),
            'consistency_frames': 0,  # 一致的帧数
            'total_frames': 0,  # 总帧数
            # 详细的一致性统计
            'frames_with_targets_le5': 0,  # 目标数<=5的帧数
            'consistent_frames_le5': 0,  # 目标数<=5且一致的帧数
            'frames_with_targets_gt5': 0,  # 目标数>5的帧数
            'consistent_frames_gt5': 0  # 目标数>5且一致的帧数
        }

        # 按顺序对比每一帧
        min_frames = min(len(gt_files), len(pred_files))
        max_frames = max(len(gt_files), len(pred_files))

        if len(gt_files) != len(pred_files):
            print(
                f"警告: 视频 {video_name} 的真实标签帧数({len(gt_files)})与预测结果帧数({len(pred_files)})不匹配，将比较前{min_frames}帧")

        # 比较每一帧
        for i in range(min_frames):
            gt_file = gt_files[i]
            pred_file = pred_files[i]
            self.evaluate_frame_pair(gt_file, pred_file, video_stats)

        # 计算该视频的指标（使用宏平均）
        video_metrics = self.calculate_metrics_for_stats(video_stats, use_macro_average=True)

        # 更新总体统计
        for class_id in video_stats['tp'].keys():
            self.total_tp[class_id] += video_stats['tp'][class_id]
            self.total_fp[class_id] += video_stats['fp'][class_id]
            self.total_fn[class_id] += video_stats['fn'][class_id]
            self.total_gt_count[class_id] += video_stats['total_gt'][class_id]
            self.total_pred_count[class_id] += video_stats['total_pred'][class_id]

        # 保存该视频的结果
        self.video_results[video_name] = {
            'frames_processed': min_frames,
            'frames_gt': len(gt_files),
            'frames_pred': len(pred_files),
            'metrics': video_metrics
        }

        # 打印视频时序一致性信息
        temporal_consistency = video_metrics['overall']['temporal_consistency']
        frames_le5 = video_metrics['overall']['frames_with_targets_le5']
        consistent_le5 = video_metrics['overall']['consistent_frames_le5']
        frames_gt5 = video_metrics['overall']['frames_with_targets_gt5']
        consistent_gt5 = video_metrics['overall']['consistent_frames_gt5']

        print(
            f"视频 {video_name}: 处理了 {min_frames} 帧 (GT: {len(gt_files)}, Pred: {len(pred_files)}) - 时序一致性: {temporal_consistency:.3f}")
        print(f"  <=5目标帧: {consistent_le5}/{frames_le5}, >5目标帧: {consistent_gt5}/{frames_gt5}")
        return min_frames

    def evaluate_all(self):
        """评估所有视频"""
        if not os.path.exists(self.gt_root):
            print(f"错误: 真实标签根目录不存在: {self.gt_root}")
            return

        if not os.path.exists(self.pred_root):
            print(f"错误: 预测结果根目录不存在: {self.pred_root}")
            return

        # 获取所有视频目录
        gt_video_dirs = [d for d in os.listdir(self.gt_root)
                         if os.path.isdir(os.path.join(self.gt_root, d))]

        pred_video_dirs = [d for d in os.listdir(self.pred_root)
                           if os.path.isdir(os.path.join(self.pred_root, d))]

        # 找到共同的视频目录
        common_videos = set(gt_video_dirs) & set(pred_video_dirs)

        if not common_videos:
            print("错误: 没有找到共同的视频目录")
            print(f"真实标签目录中的视频: {sorted(gt_video_dirs)}")
            print(f"预测结果目录中的视频: {sorted(pred_video_dirs)}")
            return

        missing_in_pred = set(gt_video_dirs) - common_videos
        missing_in_gt = set(pred_video_dirs) - common_videos

        if missing_in_pred:
            print(f"警告: 以下视频在预测结果中缺失: {sorted(missing_in_pred)}")

        if missing_in_gt:
            print(f"警告: 以下视频在真实标签中缺失: {sorted(missing_in_gt)}")

        print(f"开始评估，共发现 {len(common_videos)} 个共同视频")
        print(f"IoU阈值: {self.iou_threshold}")
        print(f"时序一致性IoU阈值: {self.consistency_iou_threshold}")
        print(f"时空稳定性阈值: {self.stability_threshold}")
        print(f"时序一致性规则:")
        print(f"  - 目标数 ≤ 5: 要求所有目标检出，且类别正确，且IoU > {self.consistency_iou_threshold}")
        print(f"  - 目标数 > 5: 要求80%目标检出，且类别正确，且IoU > {self.consistency_iou_threshold}")
        print(f"总体指标计算方式: 直接视频级平均 (Direct video-level averaging)")
        print("-" * 80)

        total_frames = 0
        for video_name in sorted(common_videos):
            frames_processed = self.evaluate_video(video_name)
            if frames_processed is not None:
                total_frames += frames_processed

        print("-" * 80)
        print(f"评估完成，总共处理了 {total_frames} 帧")

    def calculate_overall_metrics_direct_video_average(self):
        """直接基于视频总体指标进行平均"""
        if not self.video_results:
            return {}

        # 直接收集各视频的总体指标（不分类别）
        video_overall_precisions = []
        video_overall_recalls = []
        video_overall_f1s = []
        video_overall_false_alarm_rates = []
        video_temporal_consistencies = []

        # 累计统计用于显示
        total_stats = {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'total_gt': 0,
            'total_pred': 0,
            'total_frames': 0,
            'total_consistent_frames': 0,
            'total_frames_with_targets_le5': 0,
            'total_consistent_frames_le5': 0,
            'total_frames_with_targets_gt5': 0,
            'total_consistent_frames_gt5': 0
        }

        # 统计时空稳定性
        stable_videos = 0  # 时序一致性超过阈值的视频数

        for video_name, video_result in self.video_results.items():
            metrics = video_result['metrics']

            # 直接收集视频总体指标
            if 'overall' in metrics:
                overall_metric = metrics['overall']
                video_overall_precisions.append(overall_metric['precision'])
                video_overall_recalls.append(overall_metric['recall'])
                video_overall_f1s.append(overall_metric['f1'])
                video_overall_false_alarm_rates.append(overall_metric['false_alarm_rate'])
                video_temporal_consistencies.append(overall_metric['temporal_consistency'])

                # 累计统计
                total_stats['tp'] += overall_metric['tp']
                total_stats['fp'] += overall_metric['fp']
                total_stats['fn'] += overall_metric['fn']
                total_stats['total_gt'] += overall_metric['total_gt']
                total_stats['total_pred'] += overall_metric['total_pred']
                total_stats['total_frames'] += overall_metric['total_frames']
                total_stats['total_consistent_frames'] += overall_metric['consistent_frames']
                total_stats['total_frames_with_targets_le5'] += overall_metric.get('frames_with_targets_le5', 0)
                total_stats['total_consistent_frames_le5'] += overall_metric.get('consistent_frames_le5', 0)
                total_stats['total_frames_with_targets_gt5'] += overall_metric.get('frames_with_targets_gt5', 0)
                total_stats['total_consistent_frames_gt5'] += overall_metric.get('consistent_frames_gt5', 0)

                # 检查是否为稳定视频
                if overall_metric['temporal_consistency'] >= self.stability_threshold:
                    stable_videos += 1

        # 直接计算总体的视频级平均指标
        overall_precision = sum(video_overall_precisions) / len(
            video_overall_precisions) if video_overall_precisions else 0.0
        overall_recall = sum(video_overall_recalls) / len(video_overall_recalls) if video_overall_recalls else 0.0
        overall_f1 = sum(video_overall_f1s) / len(video_overall_f1s) if video_overall_f1s else 0.0
        overall_false_alarm_rate = sum(video_overall_false_alarm_rates) / len(
            video_overall_false_alarm_rates) if video_overall_false_alarm_rates else 0.0
        overall_temporal_consistency = sum(video_temporal_consistencies) / len(
            video_temporal_consistencies) if video_temporal_consistencies else 0.0

        # 计算时空序列稳定性
        spatiotemporal_stability = stable_videos / len(self.video_results) if len(self.video_results) > 0 else 0.0

        results = {
            'overall': {
                'tp': total_stats['tp'],
                'fp': total_stats['fp'],
                'fn': total_stats['fn'],
                'recall': overall_recall,
                'precision': overall_precision,
                'false_alarm_rate': overall_false_alarm_rate,
                'f1': overall_f1,
                'total_gt': total_stats['total_gt'],
                'total_pred': total_stats['total_pred'],
                'averaging_method': 'direct_video_level',
                'video_count': len(self.video_results),
                'temporal_consistency': overall_temporal_consistency,
                'total_frames': total_stats['total_frames'],
                'total_consistent_frames': total_stats['total_consistent_frames'],
                'spatiotemporal_stability': spatiotemporal_stability,
                'stable_videos': stable_videos,
                'stability_threshold': self.stability_threshold,
                'consistency_iou_threshold': self.consistency_iou_threshold,
                # 添加详细的一致性统计
                'total_frames_with_targets_le5': total_stats['total_frames_with_targets_le5'],
                'total_consistent_frames_le5': total_stats['total_consistent_frames_le5'],
                'total_frames_with_targets_gt5': total_stats['total_frames_with_targets_gt5'],
                'total_consistent_frames_gt5': total_stats['total_consistent_frames_gt5']
            }
        }

        return results

    def calculate_overall_metrics(self):
        """计算总体指标（使用直接视频级平均）"""
        return self.calculate_overall_metrics_direct_video_average()

    def print_video_result(self, video_name, video_result):
        """打印单个视频的结果"""
        print(f"\n视频: {video_name}")
        print(
            f"处理帧数: {video_result['frames_processed']} (GT: {video_result['frames_gt']}, Pred: {video_result['frames_pred']})")

        metrics = video_result['metrics']
        if not metrics:
            print("  无有效结果")
            return

        # 不显示F1指标
        print(f"{'类别':<12} {'召回率':<8} {'精确率':<8} {'虚警率':<8} {'TP':<4} {'FP':<4} {'FN':<4}")
        print("-" * 60)

        # 打印每个类别的结果
        for class_id in sorted([k for k in metrics.keys() if k != 'overall']):
            r = metrics[class_id]
            print(f"{r['class_name']:<12} {r['recall']:<8.3f} {r['precision']:<8.3f} {r['false_alarm_rate']:<8.3f} "
                  f"{r['tp']:<4} {r['fp']:<4} {r['fn']:<4}")

        # 打印总体结果
        if 'overall' in metrics:
            print("-" * 60)
            r = metrics['overall']
            print(f"{'视频总体':<12} {r['recall']:<8.3f} {r['precision']:<8.3f} {r['false_alarm_rate']:<8.3f} "
                  f"{r['tp']:<4} {r['fp']:<4} {r['fn']:<4}")

            # 打印详细的时序一致性信息
            print(f"时序一致性: {r['temporal_consistency']:.3f} ({r['consistent_frames']}/{r['total_frames']} 帧)")
            frames_le5 = r.get('frames_with_targets_le5', 0)
            consistent_le5 = r.get('consistent_frames_le5', 0)
            frames_gt5 = r.get('frames_with_targets_gt5', 0)
            consistent_gt5 = r.get('consistent_frames_gt5', 0)

            if frames_le5 > 0:
                le5_rate = consistent_le5 / frames_le5
                print(f"  目标数≤5帧: {consistent_le5}/{frames_le5} ({le5_rate:.3f}) - 要求100%检出")

            if frames_gt5 > 0:
                gt5_rate = consistent_gt5 / frames_gt5
                print(f"  目标数>5帧: {consistent_gt5}/{frames_gt5} ({gt5_rate:.3f}) - 要求80%检出")

    def print_overall_results(self, overall_metrics):
        """打印总体结果 - 显示每个视频的性能指标"""
        print("\n" + "=" * 88)
        print("总体评估结果 (各视频性能指标)")
        print("=" * 88)

        if not overall_metrics or not self.video_results:
            print("没有结果可显示")
            return

        # 显示各视频的性能指标（不显示F1、≤5目标、>5目标）
        print(
            f"{'视频名称':<20} {'召回率':<8} {'精确率':<8} {'虚警率':<8} {'时序一致性':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'帧数':<6}")
        print("-" * 88)

        # 按视频名称排序显示各视频结果
        for video_name in sorted(self.video_results.keys()):
            video_result = self.video_results[video_name]
            metrics = video_result['metrics']

            if 'overall' in metrics:
                r = metrics['overall']
                frames = video_result['frames_processed']
                temporal_consistency = r['temporal_consistency']
                stability_mark = "✓" if temporal_consistency >= self.stability_threshold else " "

                print(f"{video_name:<20} {r['recall']:<8.3f} {r['precision']:<8.3f} {r['false_alarm_rate']:<8.3f} "
                      f"{temporal_consistency:<9.3f}{stability_mark} {r['tp']:<6} {r['fp']:<6} {r['fn']:<6} {frames:<6}")

        # 显示总体平均结果
        if 'overall' in overall_metrics:
            print("-" * 88)
            r = overall_metrics['overall']
            video_count = r.get('video_count', 0)
            total_frames = sum(video_result['frames_processed'] for video_result in self.video_results.values())

            print(f"{'总体平均':<20} {r['recall']:<8.3f} {r['precision']:<8.3f} {r['false_alarm_rate']:<8.3f} "
                  f"{r['temporal_consistency']:<10.3f} {r['tp']:<6} {r['fp']:<6} {r['fn']:<6} {total_frames:<6}")

            print(f"\n详细统计:")
            print(f"总真实目标数: {r['total_gt']}")
            print(f"总预测目标数: {r['total_pred']}")
            print(f"总处理帧数: {total_frames}")
            print(f"IoU阈值: {self.iou_threshold}")
            print(f"评估视频数: {video_count}")
            print(f"时序一致性IoU阈值: {r['consistency_iou_threshold']}")
            print(
                f"时空序列稳定性: {r['spatiotemporal_stability']:.3f} ({r['stable_videos']}/{video_count} 个视频超过{r['stability_threshold']:.0%}阈值)")

            # 计算总体的≤5目标和>5目标一致性率
            total_frames_le5 = r.get('total_frames_with_targets_le5', 0)
            total_consistent_le5 = r.get('total_consistent_frames_le5', 0)
            total_frames_gt5 = r.get('total_frames_with_targets_gt5', 0)
            total_consistent_gt5 = r.get('total_consistent_frames_gt5', 0)

            overall_le5_rate = total_consistent_le5 / total_frames_le5 if total_frames_le5 > 0 else 0.0
            overall_gt5_rate = total_consistent_gt5 / total_frames_gt5 if total_frames_gt5 > 0 else 0.0

            print(f"")
            print(f"时序一致性统计:")
            print(f"  目标数≤5帧: {total_consistent_le5}/{total_frames_le5} ({overall_le5_rate:.3f}) - 要求100%检出")
            print(f"  目标数>5帧: {total_consistent_gt5}/{total_frames_gt5} ({overall_gt5_rate:.3f}) - 要求80%检出")
            print(f"")
            print(f"时序一致性规则:")
            print(f"  - 目标数 ≤ 5: 要求所有目标检出，且类别正确，且IoU > {r['consistency_iou_threshold']}")
            print(f"  - 目标数 > 5: 要求80%目标检出，且类别正确，且IoU > {r['consistency_iou_threshold']}")
            print(f"总体指标计算方式: 直接视频级平均")

    def save_results(self):
        """保存结果到文件"""
        overall_metrics = self.calculate_overall_metrics()

        results = {
            'evaluation_config': {
                'gt_root': self.gt_root,
                'pred_root': self.pred_root,
                'iou_threshold': self.iou_threshold,
                'class_names': self.class_names,
                'averaging_method': 'direct_video_level',
                'averaging_description': '直接基于视频总体指标进行平均',
                'consistency_iou_threshold': self.consistency_iou_threshold,
                'stability_threshold': self.stability_threshold,
                'temporal_consistency_rules': {
                    'targets_le5': '要求所有目标检出，且类别正确，且IoU > ' + str(self.consistency_iou_threshold),
                    'targets_gt5': '要求80%目标检出，且类别正确，且IoU > ' + str(self.consistency_iou_threshold)
                }
            },
            'video_results': {},
            'overall_results': {}
        }

        # 转换视频结果
        for video_name, video_result in self.video_results.items():
            results['video_results'][video_name] = {
                'frames_processed': video_result['frames_processed'],
                'frames_gt': video_result['frames_gt'],
                'frames_pred': video_result['frames_pred'],
                'metrics': self.convert_metrics_for_json(video_result['metrics'])
            }

        # 转换总体结果
        results['overall_results'] = self.convert_metrics_for_json(overall_metrics)

        # 保存到文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n详细结果已保存到: {self.output_file}")
        return results

    def convert_metrics_for_json(self, metrics):
        """转换指标数据为JSON可序列化格式"""
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                json_metrics[str(key)] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                          for k, v in value.items()}
            else:
                json_metrics[str(key)] = float(value) if isinstance(value, (np.integer, np.floating)) else value
        return json_metrics

    def run_evaluation(self):
        """运行完整的评估流程"""
        # 执行评估
        self.evaluate_all()

        if not self.video_results:
            print("没有找到有效的评估结果")
            return

        # 打印每个视频的结果
        print("\n" + "=" * 80)
        print("各视频详细评估结果")
        print("=" * 80)

        for video_name in sorted(self.video_results.keys()):
            self.print_video_result(video_name, self.video_results[video_name])

        # 计算和打印总体结果
        overall_metrics = self.calculate_overall_metrics()
        self.print_overall_results(overall_metrics)

        # 保存结果
        self.save_results()

        return overall_metrics


def main():
    # ==================== 配置区域 ====================
    config = {
        # 真实标签根目录 (每个视频一个子文件夹)
        'gt_root': r"D:\tiaozhanbei\code\dsfnet-trae\data\scene4\test\true",

        # 预测结果根目录 (每个视频一个子文件夹)
        'pred_root': r"D:\tiaozhanbei\code\dsfnet-trae\data\scene4\test\test",

        # IoU阈值，用于判断预测框是否匹配真实框
        'iou_threshold': 0.3,

        # 类别名称列表 (可选，如果不提供则使用Class_0, Class_1等)
        'class_names': ['drone', 'car', 'ship', 'bus', "pedestrian", "cyclist"],  # 根据你的数据集修改
        # 如果不需要指定类别名称，可以设置为 None
        # 'class_names': None

        # 结果保存文件名
        'output_file': 'detection_evaluation_results_simplified.json',

        # 时序一致性IoU阈值（默认0.3）
        'consistency_iou_threshold': 0.3,

        # 时空稳定性阈值（默认0.8，即80%）
        'stability_threshold': 0.8
    }
    # ================================================

    # 创建评估器并运行评估
    evaluator = DetectionEvaluator(config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()