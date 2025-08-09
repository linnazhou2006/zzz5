from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from datetime import datetime


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # ==================== 所有 add_argument 定义 ====================
        # basic experiment setting
        self.parser.add_argument('--model_name', default='DSFNet',
                                 help='name of the model. DSFNet | DSFNet_with_Static | DSFNet_with_Dynamic | SparseEnsembleMoE_DSFNet')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',  # 使用 action='store_true' 更符合规范
                                 help='resume an experiment.')
        self.parser.add_argument('--down_ratio', type=int, default=1,
                                 help='output stride. Currently only supports for 1.')
        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')
        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 4.')
        self.parser.add_argument('--lr_step', type=str, default='60,80',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=100,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=4,
                                 help='batch size')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--seqLen', type=int, default=5,
                                 help='number of images for per sample.')
        # test
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=256,
                                 help='max number of output objects.')
        self.parser.add_argument('--test_large_size', action='store_true',  # 使用 action='store_true'
                                 help='whether or not to test image size of 1024.')
        self.parser.add_argument('--show_results', action='store_true',
                                 help='whether or not to show the detection results.')
        self.parser.add_argument('--save_track_results', action='store_true',
                                 help='whether or not to save tracking results of sort.')
        self.parser.add_argument('--output_dir', type=str, default='./results/yolo_predictions',
                                 help='Directory to save YOLO format prediction files.')
        self.parser.add_argument('--save_conf_thresh', type=float, default=0.25,
                                 help='Confidence threshold for saving YOLO files.')
        self.parser.add_argument('--save_vis', action='store_true',
                                 help='Save visualization images.')
        self.parser.add_argument('--vis_gt', action='store_true',
                                 help='Visualize ground truth boxes.')
        self.parser.add_argument('--vis_conf_thresh', type=float, default=0,
                                 help='Confidence threshold for visualization.')
        self.parser.add_argument('--iou_thresh', type=float, default=0.3,
                                 help='iou threshold for evaluation')
        # save & dataset
        self.parser.add_argument('--save_dir', type=str, default='./weights',
                                 help='savepath of model.')
        self.parser.add_argument('--datasetname', type=str, default='rsdata',
                                 help='dataset name.')
        self.parser.add_argument('--data_dir', type=str, default='./data/RsCarData/',
                                 help='path of dataset.')
        # ==================== [核心修复] ====================
        # 添加内部使用的参数定义，并提供默认值
        self.parser.add_argument('--reg_offset', action='store_true',
                                 help='_INTERNAL_ use regression offset.')
        # =======================================================
        # 在 __init__ 方法中，任意位置添加
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    def parse(self, args=''):
        """只解析参数，不做任何修改。"""
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        return opt

    def init(self, opt):
        """用解析出的opt对象来初始化路径和其他派生变量。"""
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        if opt.data_dir.endswith('/'):
            opt.dataName = os.path.basename(os.path.dirname(opt.data_dir))
        else:
            opt.dataName = os.path.basename(opt.data_dir)

        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d-%H-%M-%S")

        # 创建一个总的实验目录
        exp_dir = os.path.join(opt.save_dir, opt.datasetname, f'{opt.model_name}_{time_str}')

        opt.exp_dir = exp_dir
        opt.save_log_dir = os.path.join(exp_dir, 'logs')
        opt.save_weights_dir = os.path.join(exp_dir, 'weights')
        opt.save_results_dir = os.path.join(exp_dir, 'results')

        os.makedirs(opt.save_log_dir, exist_ok=True)
        os.makedirs(opt.save_weights_dir, exist_ok=True)
        os.makedirs(opt.save_results_dir, exist_ok=True)

        return opt