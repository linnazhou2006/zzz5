from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import torch
from datetime import datetime

from lib.utils.opts import opts

from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.coco import COCO

from lib.external.nms import soft_nms

from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process

import cv2

from progress.bar import Bar
import matplotlib.pyplot as plt
import json

# 设置服务器环境
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib

matplotlib.use('Agg')

COLORS = [(255, 0, 0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # 计算并集
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def process(model, image, return_time=False):
    """模型推理函数"""
    start_time = time.time()

    with torch.no_grad():
        output = model(image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        torch.cuda.synchronize()
        forward_time = time.time()
        dets = ctdet_decode(hm, wh, reg=reg)

    if return_time:
        return output, dets, forward_time - start_time
    else:
        return output, dets


def post_process(dets, meta, num_classes=1, scale=1):
    """后处理函数"""
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]


def pre_process(image, scale=1):
    """预处理函数"""
    height, width = image.shape[2:4]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height,
            'out_width': inp_width}
    return meta


def merge_outputs(detections, num_classes, max_per_image):
    """合并输出函数"""
    results = {}
    for j in range(1, num_classes + 1):
        if len(detections) > 0 and j in detections[0]:
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

            if len(results[j]) > 0:
                soft_nms(results[j], Nt=0.5, method=2)
        else:
            results[j] = np.array([]).reshape(-1, 5).astype(np.float32)

    # 获取所有分数
    scores = []
    for j in range(1, num_classes + 1):
        if len(results[j]) > 0:
            scores.extend(results[j][:, 4].tolist())

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(np.array(scores), kth)[kth]
        for j in range(1, num_classes + 1):
            if len(results[j]) > 0:
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]

    return results


def load_original_annotation_data(dataset):
    """
    加载原始annotation文件，确保与测试时使用的完全一致
    """
    print(f"📋 加载原始annotation文件: {dataset.annot_path}")

    try:
        with open(dataset.annot_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        print(f"✅ 成功加载原始annotation文件")
        print(f"   - 文件路径: {dataset.annot_path}")
        print(f"   - 图像数量: {len(original_data.get('images', []))}")
        print(f"   - 原始annotation数量: {len(original_data.get('annotations', []))}")
        print(f"   - 类别数量: {len(original_data.get('categories', []))}")

        # 创建图像ID到图像信息的映射
        images_dict = {}
        for img in original_data.get('images', []):
            images_dict[img['id']] = img

        print(f"   - 建立了 {len(images_dict)} 个图像ID映射")

        # 显示一些图像路径示例
        print(f"📁 原始annotation中的图像路径示例:")
        for i, img in enumerate(original_data.get('images', [])[:5]):
            print(f"   - ID {img['id']}: {img['file_name']}")

        return original_data, images_dict

    except Exception as e:
        print(f"❌ 加载原始annotation文件失败: {e}")
        return None, {}


def generate_annotation_file_with_original_paths(opt, split, modelPath, confidence_thresh=0.2,
                                                 results_name="detection_results"):
    """
    生成COCO格式的annotation文件，确保图片路径与原始annotation完全一致
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print(f"🎯 开始生成COCO格式annotation文件")
    print(f"Model: {opt.model_name}")
    print(f"Loading model from: {modelPath}")
    print(f"置信度阈值: {confidence_thresh}")

    dataset = COCO(opt, split)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2}, opt.model_name)
    model = load_model(model, modelPath)
    model = model.cuda()
    model.eval()

    scale = 1
    num_classes = dataset.num_classes
    max_per_image = opt.K

    # 加载原始annotation数据
    original_data, original_images_dict = load_original_annotation_data(dataset)

    if original_data is None:
        print("❌ 无法加载原始annotation文件，程序退出")
        return False

    # 创建新的annotation结构，完全复制原始的info、licenses、categories
    coco_results = {
        "info": original_data.get("info", {}),
        "licenses": original_data.get("licenses", []),
        "images": [],  # 将从原始数据复制对应的图像
        "annotations": [],  # 将填充检测结果
        "categories": original_data.get("categories", [])
    }

    # 创建输出目录
    output_dir = "generated_annotations"
    os.makedirs(output_dir, exist_ok=True)

    print("🚀 开始推理阶段...")
    num_iters = len(data_loader)
    bar = Bar('Processing', max=num_iters)

    annotation_id = 1
    total_detections = 0
    valid_detections = 0
    processed_images = set()
    missing_images = []

    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        # 获取图像ID
        img_id_val = img_id.numpy().astype(np.int32)[0]

        # 从原始annotation中获取图像信息
        if img_id_val in original_images_dict:
            # 直接复制原始图像信息，保持完全一致
            original_image_info = original_images_dict[img_id_val]

            # 只在未处理过的情况下添加图像信息
            if img_id_val not in processed_images:
                coco_results["images"].append(original_image_info.copy())
                processed_images.add(img_id_val)

        else:
            # 记录缺失的图像ID
            missing_images.append(img_id_val)
            print(f"⚠️  警告: 图像ID {img_id_val} 在原始annotation中未找到")
            continue

        # 模型推理
        detection = []
        meta = pre_process(pre_processed_images['input'], scale)
        image = pre_processed_images['input'].cuda()

        # 检测
        output, dets = process(model, image, return_time=False)

        # 后处理
        dets = post_process(dets, meta, num_classes)
        detection.append(dets)
        ret = merge_outputs(detection, num_classes, max_per_image)

        # 处理检测结果
        if 1 in ret and len(ret[1]) > 0:
            for det in ret[1]:
                total_detections += 1

                # 转换检测结果格式 [x1, y1, x2, y2, conf]
                x1, y1, x2, y2, conf = float(det[0]), float(det[1]), float(det[2]), float(det[3]), float(det[4])

                # 应用置信度阈值过滤
                if conf >= confidence_thresh:
                    valid_detections += 1

                    # 转换为COCO格式边界框 [x, y, width, height]
                    bbox_x = x1
                    bbox_y = y1
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    # 确保边界框有效
                    if bbox_width > 0 and bbox_height > 0:
                        area = bbox_width * bbox_height

                        # 使用原始categories中的第一个category_id
                        if coco_results["categories"]:
                            category_id = coco_results["categories"][0]["id"]
                        else:
                            category_id = 1

                        annotation_entry = {
                            "id": annotation_id,
                            "image_id": int(img_id_val),
                            "category_id": category_id,
                            "segmentation": [],
                            "area": float(area),
                            "bbox": [float(bbox_x), float(bbox_y), float(bbox_width), float(bbox_height)],
                            "iscrowd": 0,
                            "score": float(conf)
                        }

                        coco_results["annotations"].append(annotation_entry)
                        annotation_id += 1

        bar.next()
    bar.finish()

    print(f"✅ 推理完成!")
    print(f"📊 统计信息:")
    print(f"   - 成功处理图像数: {len(coco_results['images'])}")
    print(f"   - 缺失图像数: {len(missing_images)}")
    print(f"   - 总检测数: {total_detections}")
    print(f"   - 有效检测数 (conf >= {confidence_thresh}): {valid_detections}")
    print(f"   - 生成annotation数: {len(coco_results['annotations'])}")
    if total_detections > 0:
        print(f"   - 有效检测率: {valid_detections / total_detections * 100:.1f}%")

    # 显示图像路径验证
    print(f"📁 生成annotation中的图像路径验证:")
    for i, img in enumerate(coco_results["images"][:5]):
        original_img = original_images_dict.get(img['id'], {})
        print(f"   - ID {img['id']}: {img['file_name']} ✅")
        if img['file_name'] != original_img.get('file_name', ''):
            print(f"     ❌ 路径不匹配! 原始: {original_img.get('file_name', 'N/A')}")

    if len(missing_images) > 0:
        print(f"⚠️  缺失的图像ID: {missing_images[:10]}{'...' if len(missing_images) > 10 else ''}")

    # 保存annotation文件
    annotation_file = os.path.join(output_dir, f"{results_name}_conf_{confidence_thresh}_annotations.json")

    try:
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(coco_results, f, indent=2, ensure_ascii=False)
        print(f"📁 Annotation文件已保存: {annotation_file}")

        # 验证生成的文件
        print("🔍 验证生成的annotation文件...")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        print(f"✅ 验证成功:")
        print(f"   - Images: {len(loaded_data['images'])}")
        print(f"   - Annotations: {len(loaded_data['annotations'])}")
        print(f"   - Categories: {len(loaded_data['categories'])}")

        # 验证图像路径一致性
        print(f"🔍 验证图像路径一致性...")
        path_consistency = True
        for img in loaded_data['images'][:10]:  # 检查前10个
            original_img = original_images_dict.get(img['id'], {})
            if img.get('file_name') != original_img.get('file_name'):
                print(f"❌ 路径不一致 ID {img['id']}: 生成={img.get('file_name')}, 原始={original_img.get('file_name')}")
                path_consistency = False

        if path_consistency:
            print(f"✅ 图像路径一致性检验通过")

        # 保存详细统计信息
        stats = {
            "generation_info": {
                "model": opt.model_name,
                "model_path": modelPath,
                "confidence_threshold": confidence_thresh,
                "dataset_split": split,
                "original_annotation_file": dataset.annot_path,
                "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "statistics": {
                "total_processed_images": len(coco_results['images']),
                "missing_images": len(missing_images),
                "total_detections": total_detections,
                "valid_detections": valid_detections,
                "final_annotations": len(coco_results['annotations']),
                "detection_rate": valid_detections / total_detections if total_detections > 0 else 0,
                "path_consistency": path_consistency
            },
            "data_structure": {
                "images_fields": list(coco_results['images'][0].keys()) if coco_results['images'] else [],
                "annotations_fields": list(coco_results['annotations'][0].keys()) if coco_results[
                    'annotations'] else [],
                "categories": coco_results['categories']
            },
            "sample_image_paths": [
                {
                    "id": img['id'],
                    "file_name": img['file_name'],
                    "original_match": img['file_name'] == original_images_dict.get(img['id'], {}).get('file_name', '')
                }
                for img in coco_results["images"][:10]
            ],
            "missing_image_ids": missing_images[:20] if missing_images else []
        }

        stats_file = os.path.join(output_dir, f"{results_name}_generation_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"📁 统计信息已保存: {stats_file}")

    except Exception as e:
        print(f"❌ 保存annotation文件失败: {e}")
        return False

    return True


def load_ground_truth(dataset):
    """加载真值数据"""
    gt_dict = {}

    try:
        for img_id in dataset.images:
            anns = dataset.coco.getAnnIds(imgIds=img_id)
            anns = dataset.coco.loadAnns(anns)

            gt_boxes = []
            for ann in anns:
                bbox = ann['bbox']
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                gt_boxes.append([x1, y1, x2, y2])

            gt_dict[img_id] = gt_boxes

    except Exception as e:
        print(f"⚠️  加载真值数据时出错: {e}")
        return {}

    return gt_dict


def evaluate_at_threshold(detections_all, gt_all, confidence_thresh, iou_thresh=0.5):
    """在指定置信度阈值下评估性能"""
    tp = 0
    fp = 0
    total_gt = 0
    valid_detections_count = 0

    for img_id in detections_all.keys():
        dets = detections_all[img_id]
        gts = gt_all.get(img_id, [])
        total_gt += len(gts)

        valid_dets = []
        if len(dets) > 0:
            for det in dets:
                if len(det) >= 5 and det[4] >= confidence_thresh:
                    valid_dets.append(det)
                    valid_detections_count += 1

        gt_matched = [False] * len(gts)

        for det in valid_dets:
            det_box = [det[0], det[1], det[2], det[3]]
            matched = False

            for gt_idx, gt in enumerate(gts):
                if not gt_matched[gt_idx]:
                    gt_box = [gt[0], gt[1], gt[2], gt[3]]
                    iou = calculate_iou(det_box, gt_box)

                    if iou >= iou_thresh:
                        tp += 1
                        gt_matched[gt_idx] = True
                        matched = True
                        break

            if not matched:
                fp += 1

    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / len(detections_all) if len(detections_all) > 0 else 0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Total_GT': total_gt,
        'Precision': precision,
        'Recall': recall,
        'FAR': far,
        'Valid_Detections': valid_detections_count
    }


if __name__ == '__main__':
    opt = opts().parse()

    split = 'test'
    confidence_thresh = 0.2

    if not os.path.exists(opt.save_results_dir):
        os.mkdir(opt.save_results_dir)

    if opt.load_model != '':
        modelPath = opt.load_model
    else:
        modelPath = './checkpoints/DSFNet.pth'

    print(f"📍 Model path: {modelPath}")

    try:
        results_name = opt.model_name + '_' + modelPath.split('/')[-1].split('.')[0]

        print("🎯 生成与原始annotation路径完全一致的检测结果annotation文件")
        success = generate_annotation_file_with_original_paths(opt, split, modelPath, confidence_thresh, results_name)

        if success:
            print("✅ Annotation文件生成完成！路径与原始annotation保持一致")
        else:
            print("❌ Annotation文件生成失败！")

    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback

        traceback.print_exc()
