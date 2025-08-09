from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from torch.nn.parallel import DataParallel  # 使用官方的DataParallel
from lib.utils.utils import AverageMeter
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
import numpy as np
from lib.external.nms import soft_nms
# ==================== [核心修改 1: 导入FSDP] ====================
from torch.nn.parallel.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, default_auto_wrap_policy
from functools import partial
# =============================================================


def post_process(output, meta, num_classes=1, scale=1):
    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg'] if 'reg' in output else None
    dets = ctdet_decode(hm, wh, reg=reg)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]


def merge_outputs(detections, num_classes, max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        # 检查detections列表和字典键是否存在，以及是否有内容
        valid_dets = [d[j] for d in detections if j in d and len(d[j]) > 0]
        if not valid_dets:
            results[j] = np.empty([0, 5], dtype=np.float32)
            continue
        results[j] = np.concatenate(valid_dets, axis=0).astype(np.float32)
        if len(results[j]) > 0:
            soft_nms(results[j], Nt=0.5, method=2)

    scores = np.hstack([results[j][:, 4] for j in range(1, num_classes + 1) if len(results.get(j, [])) > 0])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            if len(results.get(j, [])) > 0:
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
    return results


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        target_device = batch['input'].device
        model_return = self.model(batch['input'])

        if isinstance(model_return, tuple) and len(model_return) == 2:
            outputs, aux_loss = model_return
            if not torch.is_tensor(aux_loss):
                aux_loss = torch.tensor(aux_loss, device=target_device, dtype=torch.float32)
        else:
            outputs = model_return
            aux_loss = torch.tensor(0.0, device=target_device, dtype=torch.float32)

        loss, loss_stats = self.loss(outputs, batch)
        total_loss = loss + aux_loss.to(target_device)*0.1  # 确保aux_loss在正确设备
        loss_stats['aux_loss'] = aux_loss

        # ==================== [核心修复区域] ====================
        # 为DataParallel准备输出: 确保所有损失都是1维的CUDA张量
        final_loss = total_loss.unsqueeze(0)
        final_loss_stats = {}
        for k, v in loss_stats.items():
            final_loss_stats[k] = v.to(target_device).unsqueeze(0)
        # =======================================================

        return outputs[-1], final_loss, final_loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if isinstance(model_with_loss, DataParallel):
                model_with_loss = model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar(f'{phase} epoch {epoch}', max=num_iters)
        end = time.time()
        for iter_id, (im_id, batch) in enumerate(data_loader):
            data_time.update(time.time() - end)

            for k in batch:
                if k not in ['meta', 'file_name']:
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss_vector, loss_stats_vectors = model_with_loss(batch)

            # 从向量中取均值得到最终损失
            loss = loss_vector.mean()

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = f'({iter_id + 1}/{num_iters}) | Total: {bar.elapsed_td} | ETA: {bar.eta_td} | '
            for l in avg_loss_stats:
                # 确保key存在
                if l in loss_stats_vectors:
                    # 从向量中取均值
                    avg_loss_stats[l].update(loss_stats_vectors[l].mean().item(), batch['input'].size(0))
                    bar.suffix += f'{l}: {avg_loss_stats[l].avg:.4f} | '
            bar.next()

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def run_eval_epoch(self, phase, epoch, data_loader, base_s, dataset):
        model_with_loss = self.model_with_loss
        if isinstance(model_with_loss, DataParallel):
            model_with_loss = model_with_loss.module
        model_with_loss.eval()

        torch.cuda.empty_cache()
        opt = self.opt
        results = {}
        num_iters = len(data_loader)
        bar = Bar(f'val epoch {epoch}', max=num_iters)

        for ind, (img_id, batch) in enumerate(data_loader):
            image = batch['input'].to(opt.device)
            meta = {k: v.numpy()[0] for k, v in batch['meta'].items()}

            with torch.no_grad():
                model_return = model_with_loss.model(image)
                output = model_return[0][-1] if isinstance(model_return, tuple) else model_return[-1]
                dets = post_process(output, meta, dataset.num_classes)

            results[img_id.numpy().astype(np.int32)[0]] = dets
            bar.suffix = f'({ind + 1}/{num_iters})'
            bar.next()

        bar.finish()

        stats, _ = dataset.run_eval(results, opt.save_results_dir, f'val_epoch_{epoch}')
        ret = {'ap50': stats[1]}
        return ret, results, stats

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader, base_s, dataset):
        return self.run_eval_epoch('val', epoch, data_loader, base_s, dataset)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)