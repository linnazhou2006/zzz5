from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data
from torch import nn
import json
import numpy as np
from progress.bar import Bar
from copy import deepcopy
import time
import matplotlib
import shutil

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.external.nms import soft_nms # 确保 soft_nms 被导入

# --- Distributed Imports ---
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap

# --- Project-specific Imports ---
from lib.utils.opts import opts
from lib.models.Gumbel_MoE_DSFNet import Gumbel_MoE_DSFNet
from lib.models.DSFNet import DSFNet as DSFNet_expert
from lib.dataset.coco import COCO as CustomDataset
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
from detection_evaluator import DetectionEvaluator


# ==================== [Helper Functions] ====================

def process(model, image, opt):
    with torch.no_grad():
        model_return = model(image);
        output = model_return[0][-1] if isinstance(model_return, tuple) else model_return[-1]
        hm = output['hm'].sigmoid_();
        wh = output['wh'];
        reg = output.get('reg', None)
        dets = ctdet_decode(hm, wh, reg=reg, K=opt.K)
    return dets


def post_process(dets, meta, num_classes):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5) if len(dets[0][j]) > 0 else np.empty([0, 5],
                                                                                                                dtype=np.float32)
    return dets[0]


def save_predictions_as_yolo(predictions, resized_shape, original_img_shape, save_path, coco_id_to_yolo_id_map):
    resized_h, resized_w = resized_shape
    original_h, original_w = original_img_shape

    scale_w = original_w / resized_w if resized_w > 0 else 1
    scale_h = original_h / resized_h if resized_h > 0 else 1

    with open(save_path, 'w') as f:
        for coco_cls_id in predictions:
            yolo_cls_id = coco_id_to_yolo_id_map.get(coco_cls_id)
            if yolo_cls_id is None: continue

            for bbox in predictions[coco_cls_id]:
                score = bbox[4]
                x1_r, y1_r, x2_r, y2_r = bbox[:4]
                x1_o = x1_r * scale_w;
                y1_o = y1_r * scale_h
                x2_o = x2_r * scale_w;
                y2_o = y2_r * scale_h
                x1, x2 = np.clip([x1_o, x2_o], 0, original_w - 1)
                y1, y2 = np.clip([y1_o, y2_o], 0, original_h - 1)
                box_w, box_h = x2 - x1, y2 - y1
                if box_w <= 0 or box_h <= 0: continue
                center_x = x1 + box_w / 2
                center_y = y1 + box_h / 2
                f.write(
                    f"{yolo_cls_id} {center_x / original_w:.6f} {center_y / original_h:.6f} {box_w / original_w:.6f} {box_h / original_h:.6f} {score:.6f}\n")


# ==================== [ 核心修正 1/3: 添加 merge_outputs 函数 ] ====================
def merge_outputs(detections, num_classes, max_per_image):
    """
    Applies Soft-NMS to merge detection results.
    """
    results = {}
    for j in range(1, num_classes + 1):
        # Detections is a list of detection dicts. In test mode, it's a list with one item.
        all_dets_for_class = [det[j] for det in detections if j in det and len(det[j]) > 0]
        if not all_dets_for_class:
            results[j] = np.empty([0, 5], dtype=np.float32)
            continue

        results[j] = np.concatenate(all_dets_for_class, axis=0).astype(np.float32)

        if len(results[j]) > 0:
            soft_nms(results[j], Nt=0.5, method=2)  # Apply Soft-NMS

    # Filter by top K scores across all classes
    scores = np.hstack(
        [results[j][:, 4] for j in range(1, num_classes + 1) if len(results[j]) > 0])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            if len(results[j]) > 0:
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
    return results


# ==============================================================================

def save_and_plot_results(results, model_name, model_path, opt):
    # This function remains unchanged from the previous version.
    model_file_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join(opt.exp_dir, f'confidence_analysis_{model_file_name}')
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'confidence_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 Full evaluation results saved to: {json_path}")
    thresholds = sorted([float(k) for k in results.keys()])
    if len(thresholds) < 1: return
    recalls = [results[str(t)]['recall'] for t in thresholds]
    fars = [results[str(t)]['false_alarm_rate'] for t in thresholds]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Confidence Threshold');
    ax1.set_ylabel('Recall', color=color)
    ax1.plot(thresholds, recalls, marker='o', color=color, label='Recall')
    ax1.tick_params(axis='y', labelcolor=color);
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim([0, max(1.0, max(recalls) * 1.1 if recalls else 1.0)])
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('False Alarm Rate', color=color)
    ax2.plot(thresholds, fars, marker='s', linestyle='--', color=color, label='False Alarm Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, max(1.0, max(fars) * 1.1 if fars else 1.0)])
    fig.suptitle('Recall & False Alarm Rate vs. Confidence Threshold', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    lines, labels = ax1.get_legend_handles_labels();
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plot_path = os.path.join(output_dir, 'Recall_FAR_vs_Confidence.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"📈 Performance curves saved to: {plot_path}")


def setup(opt):
    rank = int(os.environ['RANK']);
    world_size = int(os.environ['WORLD_SIZE']);
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group("nccl");
    torch.cuda.set_device(local_rank);
    opt.local_rank = local_rank
    return rank, world_size


def gather_files(pred_root_dir, world_size, rank):
    if world_size == 1: return
    dist.barrier()
    if rank == 0:
        print("\nAggregating prediction files...")
        for r in range(1, world_size):
            rank_dir = f"{pred_root_dir}_rank_{r}"
            if os.path.exists(rank_dir):
                for video_name in os.listdir(rank_dir):
                    src_video_dir = os.path.join(rank_dir, video_name);
                    dst_video_dir = os.path.join(pred_root_dir, video_name)
                    os.makedirs(dst_video_dir, exist_ok=True)
                    for fname in os.listdir(src_video_dir): os.rename(os.path.join(src_video_dir, fname),
                                                                      os.path.join(dst_video_dir, fname))
                try:
                    shutil.rmtree(rank_dir)
                    print(f"   - Cleaned up temporary directory: {rank_dir}")
                except OSError as e:
                    print(f"   - ⚠️ Could not remove temp directory {rank_dir}: {e}")
    dist.barrier()


# ==================== [Main Test Worker] ====================

def test_main(opt):
    rank, world_size = setup(opt)

    dataset = CustomDataset(opt, 'test')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=sampler
    )

    head = {'hm': dataset.num_classes, 'wh': 2, 'reg': 2}
    head_conv = 128

    # --- 18专家初始化配方 (与训练脚本完全一致) ---
    num_experts_target = 18
    TOP_K_VALUE = 3
    base_pretrained_paths = {
        's1': './checkpoint/DSFNet_s1.pth', 's2': './checkpoint/DSFNet_s2.pth',
        's3': './checkpoint/DSFNet_s3.pth', 's4': './checkpoint/DSFNet_s4.pth'
    }
    perturbation_recipe = {'s1': 4, 's2': 4, 's3': 4, 's4': 2}

    print(f"🚀 [Rank {rank}] Re-Constructing {num_experts_target}-expert MoE model...")
    expert_list = nn.ModuleList()
    loaded_base_experts = {}
    for name, path in base_pretrained_paths.items():
        loaded_base_experts[name] = DSFNet_expert(head, head_conv)
    for name in ['s1', 's2', 's3', 's4']:
        if name in loaded_base_experts: expert_list.append(deepcopy(loaded_base_experts[name]))
    for name, num_copies in perturbation_recipe.items():
        if name in loaded_base_experts:
            for _ in range(num_copies): expert_list.append(deepcopy(loaded_base_experts[name]))

    model = Gumbel_MoE_DSFNet(
        heads=head, head_conv=head_conv, num_experts=num_experts_target,
        top_k=TOP_K_VALUE, expert_modules=expert_list
    ).to(rank)

    for i, expert_layer in enumerate(model.experts):
        model.experts[i] = wrap(expert_layer)
    fsdp_model = FSDP(model)

    # --- 分片加载逻辑 ---
    if not os.path.isdir(opt.load_model): raise ValueError(
        f"For FSDP, --load_model must be a directory. Got: {opt.load_model}")
    shard_file = os.path.join(opt.load_model, f"shard_rank_{rank}.pth")
    if not os.path.exists(shard_file): raise FileNotFoundError(f"Shard file not found for rank {rank}: {shard_file}")
    sharded_state_dict = torch.load(shard_file, map_location="cpu")
    fsdp_model.load_state_dict(sharded_state_dict)
    if rank == 0: print("✅ Trained MoE model loaded via sharded checkpoints.")
    dist.barrier()
    fsdp_model.eval()

    coco_id_to_yolo_id = dataset.cat_ids

    pred_root_dir = os.path.join(opt.save_results_dir, 'predictions_raw')
    rank_pred_dir = f"{pred_root_dir}_rank_{rank}"
    os.makedirs(rank_pred_dir, exist_ok=True)

    if rank == 0:
        bar = Bar('🚀 Inference Phase', max=len(data_loader))

    for ind, (img_id, batch) in enumerate(data_loader):
        image = batch['input'].to(rank)
        meta = {k: v.numpy()[0] for k, v in batch['meta'].items()}

        original_h, original_w = meta['original_height'], meta['original_width']
        # ==================== [ 核心修正 1/2 ] ====================
        # 从 meta 字典中获取模型输入的（resize后的）尺寸
        resized_h, resized_w = meta['out_height'], meta['out_width']
        # =========================================================

        file_rel_path = dataset.coco.loadImgs(ids=[img_id.item()])[0]['file_name']

        dets_raw = process(fsdp_model, image, opt)
        dets_processed = post_process(dets_raw, meta, dataset.num_classes)
        final_dets = merge_outputs([dets_processed], dataset.num_classes, opt.K)

        path_parts = file_rel_path.replace('\\', '/').split('/')
        video_name = path_parts[-2] if len(path_parts) > 1 else 'video_root'
        frame_name_no_ext = os.path.splitext(os.path.basename(file_rel_path))[0]
        save_video_dir = os.path.join(rank_pred_dir, video_name)
        os.makedirs(save_video_dir, exist_ok=True)
        save_path = os.path.join(save_video_dir, frame_name_no_ext + '.txt')
        # ==================== [ 核心修正 3/3: 使用 NMS 后的结果 ] ====================
        save_predictions_as_yolo(
            final_dets, # Use the merged detections
            (resized_h, resized_w),
            (original_h, original_w),
            save_path,
            coco_id_to_yolo_id
        )
        # ========================================================================
        # ==================== [ 核心修正 2/2 ] ====================
        # 将 resize后的尺寸 和 原始尺寸 都传递给函数
        save_predictions_as_yolo(
            dets_processed,
            (resized_h, resized_w),  # <--- 传入 resize 后的尺寸
            (original_h, original_w),
            save_path,
            coco_id_to_yolo_id
        )
        # =========================================================

        if rank == 0: bar.next()

    if rank == 0: bar.finish()

    gather_files(pred_root_dir, world_size, rank)
    dist.destroy_process_group()


        # ==================== [Main Execution Block] ====================
if __name__ == '__main__':
    opts_parser = opts()
    opt = opts_parser.parse()
    opt.model_name = 'Gumbel_MoE_18_Experts'
    opt = opts_parser.init(opt)

    if opt.load_model == '':
        print("❌ Error: Please specify model path with --load_model")
        exit()

    test_main(opt)

    # --- Evaluation now happens strictly on rank 0 after all processes finish ---
    if int(os.environ.get('RANK', '0')) == 0:
        results_summary = {}
        confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

        pred_root_dir_final = os.path.join(opt.save_results_dir, 'predictions_raw')

        print("\n📊 Starting multi-confidence evaluation on aggregated results...")
        for i, conf in enumerate(confidence_thresholds):
            print(f"🎯 Evaluating confidence threshold {i + 1}/{len(confidence_thresholds)}: {conf:.3f}")
            filtered_pred_root_dir = os.path.join(opt.save_results_dir, f'filtered_preds_conf_{conf:.3f}')
            os.makedirs(filtered_pred_root_dir, exist_ok=True)

            for video_name in os.listdir(pred_root_dir_final):
                src_video_dir = os.path.join(pred_root_dir_final, video_name)
                if not os.path.isdir(src_video_dir): continue
                dst_video_dir = os.path.join(filtered_pred_root_dir, video_name)
                os.makedirs(dst_video_dir, exist_ok=True)
                for fname in os.listdir(src_video_dir):
                    src_file_path = os.path.join(src_video_dir, fname)
                    if not os.path.isfile(src_file_path): continue
                    with open(src_file_path, 'r') as f_in, open(os.path.join(dst_video_dir, fname),
                                                                'w') as f_out:
                        for line in f_in:
                            try:
                                if float(line.strip().split()[-1]) >= conf: f_out.write(line)
                            except (ValueError, IndexError):
                                continue

            eval_config = {
                'gt_root': os.path.join(opt.data_dir, 'labels'),
                'pred_root': filtered_pred_root_dir,
                'iou_threshold': opt.iou_thresh,
                'class_names': CustomDataset(opt, 'test').class_name[1:],
            }
            evaluator = DetectionEvaluator(eval_config)
            evaluator.evaluate_all()
            overall_metrics = evaluator.calculate_overall_metrics()

            if 'overall' in overall_metrics:
                metrics = overall_metrics['overall']
                results_summary[str(conf)] = {
                    'recall': metrics.get('recall', 0.0), 'precision': metrics.get('precision', 0.0),
                    'f1': metrics.get('f1', 0.0), 'false_alarm_rate': metrics.get('false_alarm_rate', 1.0),
                    'spatiotemporal_stability': metrics.get('spatiotemporal_stability', 0.0),
                    'tp': metrics.get('tp', 0), 'fp': metrics.get('fp', 0), 'fn': metrics.get('fn', 0)
                }
            else:
                results_summary[str(conf)] = {'recall': 0, 'precision': 0, 'f1': 0, 'false_alarm_rate': 1.0,
                                              'spatiotemporal_stability': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

        print("✅ Evaluation complete!")

        if results_summary:
            print("\n" + "=" * 95)
            print("📊 Confidence Threshold Performance Summary")
            print("=" * 95)
            print(
                f"{'Conf.':<8} {'Recall':<10} {'Precision':<10} {'FAR':<10} {'Stability':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
            print("-" * 95)
            for conf, metrics in sorted(results_summary.items(), key=lambda item: float(item[0])):
                print(f"{float(conf):<8.3f} {metrics['recall']:<10.4f} {metrics['precision']:<10.4f} "
                      f"{metrics['false_alarm_rate']:<10.4f} {metrics['spatiotemporal_stability']:<12.4f} "
                      f"{metrics['tp']:<8} {metrics['fp']:<8} {metrics['fn']:<8}")
            print("=" * 95)
            save_and_plot_results(results_summary, opt.model_name, opt.load_model, opt)

        print("\n✅ Multi-confidence evaluation finished!")

