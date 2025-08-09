from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data
from torch import nn
from copy import deepcopy

# --- Distributed Training Imports ---
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
import json
import json
# --- Project-specific Imports ---
from lib.utils.opts import opts
from lib.utils.logger import Logger
from lib.models.stNet import save_model
from lib.models.DSFNet import DSFNet as DSFNet_expert
from lib.models.Gumbel_MoE_DSFNet import Gumbel_MoE_DSFNet
from lib.dataset.coco_rsdata import COCO
from lib.Trainer.ctdet import CtdetTrainer


def perturb_model_weights(model, noise_level=1e-4):
    """
    对模型的所有可训练权重和偏置添加一个小的随机高斯噪声。
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * noise_level
                param.add_(noise)
    return model


def setup(opt):
    """
    使用由 torchrun 提供的环境变量初始化分布式进程组。
    """
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    opt.local_rank = local_rank
    return rank, world_size


def main(opt):
    """
    主训练函数，由每个进程独立执行。
    """
    rank, world_size = setup(opt)
    logger = Logger(opt) if rank == 0 else None

    torch.manual_seed(opt.seed + rank)

    DataTrain = COCO(opt, 'train')
    train_sampler = DistributedSampler(DataTrain, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        DataTrain, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler
    )

    val_loader = None
    if rank == 0:
        DataVal = COCO(opt, 'test')
        val_loader = torch.utils.data.DataLoader(
            DataVal, batch_size=1, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True
        )

    dist.barrier()

    print(f"[Rank {rank}] Creating model...")
    head = {'hm': DataTrain.num_classes, 'wh': 2, 'reg': 2}
    head_conv = 128

    # ==================== [核心: 18专家初始化配方] ====================
    num_experts_target = 18
    TOP_K_VALUE = 4

    base_pretrained_paths = {
        's1': './checkpoint/DSFNet_s1.pth',
        's2': './checkpoint/DSFNet_s2.pth',
        's3': './checkpoint/DSFNet_s3.pth',
        's4': './checkpoint/DSFNet_s4.pth'
    }
    perturbation_recipe = {'s1': 4, 's2': 4, 's3': 4, 's4': 2}

    if rank == 0:
        print(f"🚀 初始化 {num_experts_target} 个专家的MoE模型...")

    expert_list = nn.ModuleList()
    loaded_base_experts = {}

    for name, path in base_pretrained_paths.items():
        base_expert = DSFNet_expert(head, head_conv)
        if os.path.exists(path):
            from lib.models.stNet import load_model
            base_expert = load_model(base_expert, path)
            if rank == 0: print(f"   - 成功加载基础权重: {os.path.basename(path)}")
        else:
            if rank == 0: print(f"   - ⚠️ 警告: 路径不存在 {path}。")
        loaded_base_experts[name] = base_expert

    for name in ['s1', 's2', 's3', 's4']:
        if name in loaded_base_experts:
            expert_list.append(deepcopy(loaded_base_experts[name]))

    for name, num_copies in perturbation_recipe.items():
        if name in loaded_base_experts:
            base_expert_to_copy = loaded_base_experts[name]
            for _ in range(num_copies):
                #expert_list.append(deepcopy(perturb_model_weights(base_expert_to_copy)))
                # 创建一个深度拷贝，但不进行权重扰动
                replicated_expert = deepcopy(base_expert_to_copy)
                # Oringinal line: perturbed_expert = perturb_model_weights(perturbed_expert, noise_level=1e-4)
                expert_list.append(replicated_expert)
                # ====================================================

    if len(expert_list) != num_experts_target:
        raise ValueError(f"专家创建失败！期望 {num_experts_target}，实际 {len(expert_list)}")

    if rank == 0: print(f"✅ 成功创建并初始化了 {len(expert_list)} 个专家。")

    model = Gumbel_MoE_DSFNet(
        heads=head, head_conv=head_conv, num_experts=num_experts_target,
        top_k=TOP_K_VALUE, expert_modules=expert_list
    )
    # ==========================================================================

    # ==================== [核心修正: 差分学习率] ====================
    expert_lr = opt.lr * 0.05
    if rank == 0:
        print(f"Setting up optimizer with differential learning rates:")
        print(f"  - Gating Network LR: {opt.lr}")
        print(f"  - Experts LR: {expert_lr}")

    # 在模型被FSDP包装之前定义参数组
    params_to_optimize = [
        {'params': model.gating_network.parameters(), 'lr': opt.lr},
        {'params': model.experts.parameters(), 'lr': expert_lr}
    ]
    # =================================================================

    # 将模型移动到设备，然后进行手动包装
    model = model.to(rank)

    print(f"[Rank {rank}] Manually wrapping {len(model.experts)} expert modules for FSDP...")
    for i, expert_layer in enumerate(model.experts):
        model.experts[i] = wrap(expert_layer)

    fsdp_model = FSDP(model)
    print(f"[Rank {rank}] Model wrapped with FSDP using manual wrapping.")

    # 使用预先定义好的参数组来创建优化器
    optimizer = torch.optim.Adam(params_to_optimize)

    start_epoch = 0
    if opt.load_model != '' and opt.resume:
        # --- [核心修正: PyTorch 1.11.0 分片加载逻辑] ---
        if not os.path.isdir(opt.load_model):
            raise ValueError("For FSDP resume, --load_model must be a directory containing sharded checkpoints.")

        if rank == 0:
            print(f"Loading FSDP sharded checkpoint from directory: {opt.load_model}")

        # 1. 每个进程加载自己的分片文件
        shard_file = os.path.join(opt.load_model, f"shard_rank_{rank}.pth")
        if not os.path.exists(shard_file):
            raise FileNotFoundError(f"Shard file not found for rank {rank}: {shard_file}")

        sharded_state_dict = torch.load(shard_file, map_location="cpu")

        # 2. 加载状态字典
        fsdp_model.load_state_dict(sharded_state_dict)

        # 3. Rank 0 加载元数据来恢复 epoch
        if rank == 0:
            meta_path = os.path.join(opt.load_model, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta_data = json.load(f)
                start_epoch = meta_data.get('epoch', 0)
                print(f"Resuming from epoch {start_epoch}.")
            else:
                print("Warning: meta.json not found. Resuming from epoch 0.")

        # 将 start_epoch 广播给所有进程
        epoch_tensor = torch.tensor([start_epoch], dtype=torch.int64, device=rank)
        dist.broadcast(epoch_tensor, src=0)
        start_epoch = epoch_tensor.item()
        # --- [修正结束] ---
    dist.barrier()

    trainer = CtdetTrainer(opt, fsdp_model, optimizer)
    trainer.set_device(opt.gpus, rank)

    print(f'[Rank {rank}] Starting training...')
    best = -1
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        log_dict_train, _ = trainer.train(epoch, train_loader)

        # ==================== [ 核心修正 ] ====================
        # save_model 必须由所有进程调用，以便每个进程都能保存自己的分片。
        # 我们将它从 "if rank == 0" 块中移出。

        # 1. 所有进程都调用 save_model 来保存最新的模型
        #    函数内部会处理 rank 0 的特殊逻辑（如创建目录和写meta文件）
        save_dir_last = os.path.join(opt.save_weights_dir, 'model_last')
        save_model(save_dir_last, epoch, fsdp_model, optimizer)

        # 2. 日志记录和验证仍然只在 rank 0 上进行
        if rank == 0:
            logger.write(f'epoch: {epoch} |')
            # 注意：上面的 save_model 调用已经被移出
            for k, v in log_dict_train.items():
                logger.write(f'{k} {v:8f} | ')

            if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
                with torch.no_grad():
                    log_dict_val, _, stats = trainer.val(epoch, val_loader, DataVal.coco, DataVal)
                for k, v in log_dict_val.items():
                    logger.write(f'{k} {v:8f} | ')
                if log_dict_val['ap50'] > best:
                    best = log_dict_val['ap50']
                    print(f"\n[Rank 0] New best model found! Saving model_best...")
                    # 同样，所有进程都需要调用 save_model
                    save_dir_best = os.path.join(opt.save_weights_dir, 'model_best')
                    save_model(save_dir_best, epoch, fsdp_model)

            logger.write('\n')

        # 3. 在所有操作（包括可能的 best model 保存）之后进行同步
        #    这个 barrier 现在是可选的，因为 save_model 内部已经有 barrier
        dist.barrier()
        # =========================================================

        if epoch in opt.lr_step:
            lr_new = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            expert_lr_new = expert_lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            if rank == 0:
                print(f'Drop LR to Gating: {lr_new}, Experts: {expert_lr_new}')

            optimizer.param_groups[0]['lr'] = lr_new
            optimizer.param_groups[1]['lr'] = expert_lr_new

    if rank == 0 and logger is not None:
        logger.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    opts_parser = opts()
    opt = opts_parser.parse()
    opt.model_name = 'Gumbel_MoE_18_Experts'
    opt = opts_parser.init(opt)

    main(opt)