from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import json
from progress.bar import Bar
import matplotlib.pyplot as plt

# 确保 matplotlib 在无头服务器上正常工作
import matplotlib

matplotlib.use('Agg')

from lib.utils.opts import opts
from lib.models.stNet import get_det_net, load_model

# ========================= [核心变更 1: 导入正确的依赖] =========================
# 我们将使用 coco.py，因为它已经被我们修复以支持您的数据集格式
from lib.dataset.coco import COCO as CustomDataset
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process

# 导入我们强大的评估器
from evaluator import DetectionEvaluator


# =========================================================================

# --- 辅助函数 ---

def process(model, image, opt):
    """模型推理函数"""
    with torch.no_grad():
        # 模型输入期望是 5D 张量: [B, C, T, H, W]
        if len(image.shape) == 4:
            # 如果输入是 4D [B, C*T, H, W]，需要 reshape
            # 这取决于数据加载器的最终输出
            # 假设数据加载器返回 (B, C, T, H, W)
            pass

        output = model(image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output.get('reg', None)
        dets = ctdet_decode(hm, wh, reg=reg, K=opt.K)
    return dets


def post_process(dets, meta, num_classes):
    """后处理函数，将热图坐标转换为原始图像坐标"""
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    # ctdet_post_process 使用 meta 字典中的 'c' 和 's' 来进行逆变换
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)

    for j in range(1, num_classes + 1):
        if len(dets[0][j]) > 0:
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        else:
            dets[0][j] = np.empty([0, 5], dtype=np.float32)

    return dets[0]


def save_predictions_as_yolo(predictions, original_img_shape, save_path, class_id_map):
    """将检测结果保存为 YOLO .txt 格式"""
    img_h, img_w = original_img_shape
    with open(save_path, 'w') as f:
        # predictions 的 key 是 COCO 类别 ID (从1开始)
        for coco_cls_id in predictions:
            # 从 COCO ID 映射回 YOLO ID (从0开始)
            yolo_cls_id = class_id_map.get(coco_cls_id)
            if yolo_cls_id is None:
                continue

            for bbox in predictions[coco_cls_id]:
                score = bbox[4]
                # 保存所有结果，不过滤置信度
                x1, y1, x2, y2 = bbox[:4]

                # 确保坐标在图像范围内
                x1, x2 = np.clip([x1, x2], 0, img_w)
                y1, y2 = np.clip([y1, y2], 0, img_h)

                box_w, box_h = x2 - x1, y2 - y1
                center_x, center_y = x1 + box_w / 2, y1 + box_h / 2

                # 归一化
                center_x_norm, w_norm = center_x / img_w, box_w / img_w
                center_y_norm, h_norm = center_y / img_h, box_h / img_h

                f.write(
                    f"{yolo_cls_id} {center_x_norm:.6f} {center_y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {score:.6f}\n")


# ========================= [核心变更 2: 评估主函数] =========================
def test_and_evaluate_multi_confidence(opt, modelPath):
    """
    运行推理，保存YOLO格式结果，并使用不同置信度进行评估。
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # --- 1. 数据加载和模型准备 ---
    print(f"📍 Model path: {modelPath}")
    print(f"Model: {opt.model_name}")

    # 使用我们修复好的 CustomDataset (coco.py)
    dataset = CustomDataset(opt, 'test')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2}, opt.model_name)
    print(f"Loading model from: {modelPath}")
    model = load_model(model, modelPath)
    model = model.to(opt.device)
    model.eval()

    # 创建 COCO ID -> YOLO ID 的逆映射
    # dataset.cat_ids 是 {COCO_ID: YOLO_ID}, 我们需要反过来
    yolo_id_to_coco_id = {v: k for k, v in dataset.cat_ids.items()}
    coco_id_to_yolo_id = {k: v for k, v in dataset.cat_ids.items()}

    # --- 2. 推理并保存所有结果 (conf=0.0) ---
    model_file_name = os.path.splitext(os.path.basename(modelPath))[0]
    pred_root_dir = os.path.join('./results', f'{opt.model_name}_{model_file_name}', 'yolo_predictions_raw')
    if not os.path.exists(pred_root_dir):
        os.makedirs(pred_root_dir)
    print(f"Raw predictions (conf=0.0) will be saved to: {pred_root_dir}")

    bar = Bar(f'🚀 Inference Phase', max=len(data_loader))
    for ind, (img_id, batch) in enumerate(data_loader):
        # 从 batch 字典中获取数据
        image = batch['input'].to(opt.device)
        meta = {k: v.numpy()[0] for k, v in batch['meta'].items()}
        img_h, img_w = meta['original_height'], meta['original_width']

        # 从数据加载器获取原始文件名（包含相对路径）
        file_rel_path = dataset.coco.loadImgs(ids=[img_id.item()])[0]['file_name'].lstrip('/')

        # 推理和后处理
        dets_raw = process(model, image, opt)
        dets = post_process(dets_raw, meta, dataset.num_classes)

        # 构建保存路径
        video_name = os.path.dirname(file_rel_path).split('/')[-1]  # e.g., 'data08'
        frame_name_no_ext = os.path.splitext(os.path.basename(file_rel_path))[0]

        save_video_dir = os.path.join(pred_root_dir, video_name)
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        save_path = os.path.join(save_video_dir, frame_name_no_ext + '.txt')

        # 保存为 YOLO 格式
        save_predictions_as_yolo(dets, (img_h, img_w), save_path, coco_id_to_yolo_id)
        bar.next()
    bar.finish()
    print(f"✅ Inference complete!")

    # --- 3. 多置信度评估 ---
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results_summary = {}

    print("\n📊 Starting multi-confidence evaluation...")
    for i, conf in enumerate(confidence_thresholds):
        print(f"🎯 Evaluating confidence threshold {i + 1}/{len(confidence_thresholds)}: {conf:.2f}")

        filtered_pred_root_dir = os.path.join('./results', f'{opt.model_name}_{model_file_name}',
                                              f'filtered_preds_conf_{conf:.2f}')
        if not os.path.exists(filtered_pred_root_dir):
            os.makedirs(filtered_pred_root_dir)

        # 过滤预测文件
        for video_name in os.listdir(pred_root_dir):
            src_video_dir = os.path.join(pred_root_dir, video_name)
            if not os.path.isdir(src_video_dir): continue

            dst_video_dir = os.path.join(filtered_pred_root_dir, video_name)
            if not os.path.exists(dst_video_dir): os.makedirs(dst_video_dir)

            for fname in os.listdir(src_video_dir):
                with open(os.path.join(src_video_dir, fname), 'r') as f_in, \
                        open(os.path.join(dst_video_dir, fname), 'w') as f_out:
                    for line in f_in:
                        if float(line.strip().split()[-1]) >= conf:
                            f_out.write(line)

        # 调用 evaluator.py
        eval_config = {
            'gt_root': os.path.join(opt.data_dir, 'labels'),
            'pred_root': filtered_pred_root_dir,
            'iou_threshold': opt.iou_thresh,
            'class_names': dataset.class_name[1:],  # 移除 __background__
        }
        evaluator = DetectionEvaluator(eval_config)
        evaluator.evaluate_all()
        overall_metrics = evaluator.calculate_overall_metrics()

        if 'overall' in overall_metrics:
            metrics = overall_metrics['overall']
            results_summary[conf] = {
                'recall': metrics['recall'], 'precision': metrics['precision'], 'f1': metrics['f1'],
                'tp': metrics['tp'], 'fp': metrics['fp'], 'fn': metrics['fn']
            }
        else:
            results_summary[conf] = {'recall': 0, 'precision': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': 0}

    print("✅ Evaluation complete!")
    return results_summary


def save_and_plot_results(results, model_name, model_path):
    """保存并绘制结果曲线"""
    # ... (此函数与 test_scene1.py 中的版本完全相同) ...
    model_file_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join('./confidence_analysis_results', f'{model_name}_{model_file_name}')
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'confidence_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 Full results saved to: {json_path}")
    thresholds = sorted(results.keys())
    if len(thresholds) < 2: return
    recalls = [results[t]['recall'] for t in thresholds]
    precisions = [results[t]['precision'] for t in thresholds]
    f1s = [results[t]['f1'] for t in thresholds]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(recalls, precisions, marker='o')
    for i, th in enumerate(thresholds):
        if i % 2 == 0: plt.text(recalls[i], precisions[i], f'{th:.1f}')
    plt.title('Precision-Recall Curve');
    plt.xlabel('Recall');
    plt.ylabel('Precision')
    plt.grid(True);
    plt.xlim([0, 1.0]);
    plt.ylim([0, 1.05])
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, f1s, marker='o', label='F1-Score')
    best_f1_idx = np.argmax(f1s)
    best_conf, best_f1 = thresholds[best_f1_idx], f1s[best_f1_idx]
    plt.title(f'F1-Score vs. Confidence\nBest F1={best_f1:.3f} @ Conf={best_conf:.2f}')
    plt.xlabel('Confidence Threshold');
    plt.ylabel('F1-Score')
    plt.grid(True);
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'confidence_curves.png')
    plt.savefig(plot_path)
    print(f"📈 Performance curves saved to: {plot_path}")


# ========================= [核心变更 3: 更新主执行块] =========================
if __name__ == '__main__':
    opt = opts().parse()

    if opt.load_model == '':
        modelPath = './checkpoint/DSFNet.pth'
    else:
        modelPath = opt.load_model

    # 运行新的评估流程
    results_summary = test_and_evaluate_multi_confidence(opt, modelPath)

    # 打印总结表格并保存结果
    if results_summary:
        print("\n" + "=" * 60)
        print("📊 Confidence Threshold Performance Summary")
        print("=" * 60)
        print(f"{'Conf.':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<8} {'FP':<8} {'FN':<8}")
        print("-" * 60)
        for conf, metrics in sorted(results_summary.items()):
            print(f"{conf:<6.1f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} {metrics['tp']:<8} {metrics['fp']:<8} {metrics['fn']:<8}")
        print("=" * 60)
        save_and_plot_results(results_summary, opt.model_name, modelPath)

    print("\n✅ Multi-confidence evaluation finished!")