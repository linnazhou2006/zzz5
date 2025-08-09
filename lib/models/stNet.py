from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# --- 在文件顶部添加此行 ---
import torch.distributed as dist
import torch
from thop import profile
from lib.models.DSFNet_with_Static import DSFNet_with_Static
from lib.models.DSFNet import DSFNet
from lib.models.DSFNet_with_Dynamic import DSFNet_with_Dynamic
# ==================== [核心修改] ====================
from lib.models.Sparse_Ensemble_MoE_DSFNet import SparseEnsembleMoE_DSFNet
# =======================================================
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
import json # 确保json被导入

def model_lib(model_chose):
    model_factory = {
        'DSFNet_with_Static': DSFNet_with_Static,
        'DSFNet': DSFNet,
        'DSFNet_with_Dynamic': DSFNet_with_Dynamic,
        # ==================== [核心修改] ====================
        'SparseEnsembleMoE_DSFNet': SparseEnsembleMoE_DSFNet,
        # =======================================================
    }
    return model_factory[model_chose]


def get_det_net(heads, model_name):
    model_name = model_lib(model_name)
    model = model_name(heads)
    return model

def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    # --- [核心修改: 分片加载逻辑] ---
    # 这个函数现在主要用于加载非FSDP模型或单个专家权重
    # FSDP模型的加载将在训练脚本中专门处理

    # 检查 model_path 是文件还是目录
    if os.path.isdir(model_path):
        # 这是一个分片检查点，此函数不处理它
        # FSDP 加载逻辑应该在训练脚本中
        print(f"Warning: '{model_path}' is a directory (sharded checkpoint). "
              f"This load_model function is for single files. FSDP loading should be handled in the training script.")
        return model  # 直接返回，不加载

    # --- 原有的单个文件加载逻辑 ---
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    # 判断加载的是否是state_dict本身
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        state_dict_ = checkpoint
        epoch = -1  # 未知 epoch
    else:
        state_dict_ = checkpoint['state_dict']
        epoch = checkpoint.get('epoch', -1)

    print(f'loaded {model_path}, epoch {epoch}')
    state_dict = {}

    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, FSDP):
        # --- [核心修改: 分片保存逻辑] ---
        # 1. 创建用于保存的目录
        os.makedirs(path, exist_ok=True)

        # 2. (可选但推荐) 使用 barrier 确保所有进程的目录创建操作都已完成，
        #    再继续执行后续的写操作。
        dist.barrier()
        # --- [修正结束] ---
        # 2. 每个进程保存自己的分片
        # state_dict() 在 FSDP 实例上会返回当前 rank 的分片
        sharded_state_dict = model.state_dict()
        shard_file = os.path.join(path, f"shard_rank_{dist.get_rank()}.pth")
        torch.save(sharded_state_dict, shard_file)

        # 3. 只有 rank 0 保存元数据
        if dist.get_rank() == 0:
            meta_data = {
                "epoch": epoch,
                "world_size": dist.get_world_size()
                # 可以在此添加其他元信息, e.g., model_config
            }
            with open(os.path.join(path, "meta.json"), "w") as f:
                json.dump(meta_data, f)

        # 4. 等待所有进程完成写入
        dist.barrier()
        if dist.get_rank() == 0:
            print("Sharded checkpoint saved successfully.")

    else:  # 非分布式或普通DDP
        # ... (原有逻辑不变，但简化为只保存 state_dict)
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if isinstance(model, (torch.nn.DataParallel, DDP)):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save(state_dict, path)
