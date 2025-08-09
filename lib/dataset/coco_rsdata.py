from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import math
import cv2
import torch
import torch.utils.data as data

from lib.utils.image import gaussian_radius, draw_umich_gaussian
from lib.utils.augmentations import Augmentation


class COCO(data.Dataset):
    # ==================== [核心修复 1: 移除 opts.py 依赖] ====================
    # 彻底移除在类定义时就解析参数的危险做法
    # opt = opts().parse()  # <-- 必须删除或注释掉这一行

    # 将内部配置定义为类属性
    reg_offset = True
    num_classes = 6
    default_resolution = [640, 512]
    # ===================================================================

    mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(COCO, self).__init__()
        self.opt = opt  # 仍然接收opt，用于获取路径、batch_size等
        self.split = split
        self.data_dir = self.opt.data_dir

        self.resolution = self.default_resolution
        if split != 'train':
            print(f"==> Test resolution set to: {self.resolution[0]}x{self.resolution[1]}")

        annot_filename = f'instances_{"val" if split == "test" else split}2017.json'
        self.annot_path = os.path.join(self.data_dir, 'annotations', annot_filename)

        self.max_objs = opt.K
        self.seqLen = opt.seqLen
        self.down_ratio = opt.down_ratio

        self.class_name = ['__background__', 'drone', 'car', 'ship', 'bus', 'pedestrian', 'cyclist']
        self._valid_ids = [1, 2, 3, 4, 5, 6]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        print('==> initializing coco 2017 {} data.'.format(split))
        print(f'==> Loading annotations from: {self.annot_path}')

        if not os.path.exists(self.annot_path):
            raise FileNotFoundError(f"Annotation file not found: {self.annot_path}")

        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))
        self.aug = Augmentation() if split == 'train' else None

    def _coco_box_to_bbox(self, box):
        return np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.coco.loadImgs(ids=[img_id])[0]

        absolute_path = os.path.join(self.data_dir, img_info['file_name'])
        img = cv2.imread(absolute_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {absolute_path}")

        original_h, original_w = img.shape[:2]
        target_h, target_w = self.resolution[1], self.resolution[0]

        if original_h != target_h or original_w != target_w:
            img = cv2.resize(img, (target_w, target_h))

        scale_w = target_w / original_w
        scale_h = target_h / original_h

        inp_buffer = np.zeros([target_h, target_w, 3, self.seqLen], dtype=np.float32)
        for ii in range(self.seqLen):
            inp_i = (img.astype(np.float32) / 255.)
            inp_buffer[:, :, :, ii] = (inp_i - self.mean) / self.std
        inp = inp_buffer.transpose(2, 3, 0, 1)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        bbox_tol, cls_id_tol = [], []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            bbox[0] *= scale_w;
            bbox[2] *= scale_w
            bbox[1] *= scale_h;
            bbox[3] *= scale_h
            bbox_tol.append(bbox)
            cls_id_tol.append(self.cat_ids[ann['category_id']])

        output_h = target_h // self.down_ratio
        output_w = target_w // self.down_ratio
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian
        for k in range(num_objs):
            bbox = np.array(bbox_tol[k])
            cls_id = cls_id_tol[k]
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                if not (0 <= ct_int[0] < target_w and 0 <= ct_int[1] < target_h): continue

                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

        meta = {
            'c': np.array([target_w / 2., target_h / 2.], dtype=np.float32),
            's': max(target_h, target_w) * 1.0,
            'out_height': output_h,
            'out_width': output_w,
            'original_height': original_h,
            'original_width': original_w
        }

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'meta': meta}

        # ==================== [核心修复 3] ====================
        # 现在这个检查是稳定可靠的，因为它使用的是类自身的属性
        if self.reg_offset:
            ret.update({'reg': reg})
        # =======================================================

        return img_id, ret

    # 其他方法 (convert_eval_format, run_eval等) 保持不变...
    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id), "category_id": int(category_id),
                        "bbox": bbox_out, "score": float("{:.2f}".format(score))
                    }
                    detections.append(detection)
        return detections

    def run_eval(self, results, save_dir, time_str):
        self.save_results(results, save_dir, time_str)
        coco_dets = self.coco.loadRes(os.path.join(save_dir, f"results_{time_str}.json"))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats, coco_eval.eval['precision']

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def save_results(self, results, save_dir, time_str):
        path = os.path.join(save_dir, f"results_{time_str}.json")
        with open(path, 'w') as f:
            json.dump(self.convert_eval_format(results), f)
        print(f"Results saved to {path}")