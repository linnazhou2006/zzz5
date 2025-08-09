# 文件: lib/dataset/misc.py

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor

# --- 核心：处理 torchvision 版本兼容性 ---
import torchvision
# 确保 torchvision 和 torch 版本匹配，这里假设torchvision 0.12.0
# PyTorch >= 1.7.0 (通常 torchvision >= 0.8.0) 推荐直接使用 torchvision.ops.misc.interpolate
# PyTorch 1.11.0 配合 torchvision 0.12.0 应该能直接使用 torchvision.ops.misc.interpolate

# 确定我们是使用旧版本 PyTorch (torchvision < 0.7) 的特定处理，还是使用新版本
#Torchvision版本是0.12.0，所以我们走else分支

# --- interpolate 函数 ---
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    提供一个兼容的 interpolate 接口。
    """
    
    # 检查 PyTorch 和 Torchvision 的版本组合
    # 由于我们已知 PyTorch >= 1.11.0 和 torchvision == 0.12.0
    # 它们都支持 torchvision.ops.misc.interpolate
    
    # 即使 torch.nn.functional.interpolate 已经改进，
    # 但 torchvision.ops.misc.interpolate 仍然可以处理一些边缘情况，
    # 并且可能更稳定，因为它被设计为与 torchvision 的其他 ops 一致。
    
    # 直接使用 torchvision.ops.misc.interpolate
    # 它能处理空批次等情况
    return torchvision.ops.misc.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

# --- 其他辅助函数 (SmoothedValue, all_gather, reduce_dict, MetricLogger, etc.) ---
# 这些函数应该保持不变，它们不直接依赖于 torchvision.ops.misc 的特定函数

class SmoothedValue(object):
    # ... (保持不变) ...
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def all_gather(data):
    world_size = get_world_size()
    if world_size == 1: return [data]
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list

def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2: return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average: values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def get_world_size():
    if not is_dist_avail_and_initialized(): return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized(): return 0
    return dist.get_rank()

def get_local_size():
    if not is_dist_avail_and_initialized(): return 1
    return int(os.environ['LOCAL_SIZE'])

def get_local_rank():
    if not is_dist_avail_and_initialized(): return 0
    return int(os.environ['LOCAL_RANK'])

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process(): torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if int(os.environ.get('DEBUG', '0')) == 0 and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif int(os.environ.get('DEBUG', '0')) == 0 and 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0: return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# =========================================================================
# interpolate 函数是关键，它需要正确处理版本兼容性
# =========================================================================
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    提供一个兼容的 interpolate 接口。
    它应该能处理 PyTorch >= 1.11.0 和 torchvision >= 0.12.0 的情况。
    """
    
    # 检查 PyTorch 和 Torchvision 版本
    # 我们知道 PyTorch 是 1.11.0，Torchvision 是 0.12.0
    # 这意味着 torchvision.__version__[:3] 是 "0.12"
    # float("0.12") 是 0.12
    
    # torchvision.ops.misc.interpolate 是 PyTorch >= 1.7.0 / torchvision >= 0.8.0 推荐的 API
    # 您的环境 (torch=1.11.0, torchvision=0.12.0) 绝对满足这个条件
    
    # print(f"Debug: Using torchvision.ops.misc.interpolate directly. Torchvision version: {torchvision.__version__}")
    
    # 直接调用 torchvision.ops.misc.interpolate，它已经包含了对各种情况的处理
    # 并且可以处理 empty batch sizes
    return torchvision.ops.misc.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if not parameters: return torch.tensor(0.0)
    
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)