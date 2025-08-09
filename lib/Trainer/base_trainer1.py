from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from lib.utils.utils import AverageMeter
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process
import numpy as np
from lib.external.nms import soft_nms
from lib.dataset.coco_eval import CocoEvaluator

def post_process(output, meta, num_classes=1, scale=1):
    # decode
    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg']

    torch.cuda.synchronize()
    dets = ctdet_decode(hm, wh, reg=reg)
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]

def merge_outputs(detections, num_classes ,max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        soft_nms(results[j], Nt=0.5, method=2)

    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        # 移至正确的设备
        target_device = next(self.model.parameters()).device
        for k in batch:
            if k != 'meta' and k != 'file_name' and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device=target_device, non_blocking=True)

        model_return = self.model(batch['input'])

        if isinstance(model_return, tuple) and len(model_return) == 2:
            outputs, aux_loss = model_return
            if not torch.is_tensor(aux_loss):
                aux_loss = torch.tensor(aux_loss, device=target_device, dtype=torch.float32)
        else:
            outputs = model_return
            aux_loss = torch.tensor(0.0, device=target_device, dtype=torch.float32)

        loss, loss_stats = self.loss(outputs, batch)

        # 将辅助损失添加到总损失中 (确保aux_loss在正确的设备上)
        total_loss = loss + aux_loss.to(target_device)
        loss_stats['aux_loss'] = aux_loss

        return outputs[-1], total_loss, loss_stats

class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def set_device(self, gpus, device):
        # FSDP 将在主脚本中处理模型到GPU的分配，这里不再需要 DataParallel
        # 仅将未被FSDP包装的部分（如果有）移动到设备
        # 在我们的案例中，整个 model_with_loss 将被FSDP管理
        if len(gpus) <= 1:
            self.model_with_loss = self.model_with_loss.to(device)

        # 优化器状态由FSDP管理，无需手动移动
        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            # 在FSDP中，模型始终处于包装状态，直接调用 eval() 即可
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        # num_iters = len(data_loader) // 20 # 恢复完整的epoch训练
        num_iters = len(data_loader)
        end = time.time()

        # 为分布式训练设置sampler的epoch
        if opt.gpus[0] != -1 and hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)

        for iter_id, (im_id, batch) in enumerate(data_loader):
            # 移除迭代限制，或使其成为可选参数
            # if iter_id >= num_iters:
            #   break
            data_time.update(time.time() - end)

            # 数据到设备的移动已在 ModelWithLoss 中完成
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)

            # 只在主进程 (rank 0) 打印日志
            if opt.local_rank == 0 and (iter_id + 1) % 20 == 0:
                print(
                    'phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f, wh_loss=%0.4f, off_loss=%0.4f, aux_loss=%0.4f' \
                    % (phase, epoch, iter_id + 1, num_iters, time.time() - end,
                       loss.item(),
                       loss_stats['hm_loss'].mean().item(),
                       loss_stats['wh_loss'].mean().item(),
                       loss_stats['off_loss'].mean().item(),
                       loss_stats['aux_loss'].mean().item()))

            end = time.time()

            for l in avg_loss_stats:
                # 在分布式环境中，损失已经是该进程的平均值，我们需要在所有进程结束后进行同步
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = 1 / 60.

        return ret, results

    def run_eval_epoch(self, phase, epoch, data_loader, base_s, dataset):
        model_with_loss = self.model_with_loss

        if len(self.opt.gpus) > 1:
            model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        end = time.time()

        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
              break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta' and k != 'file_name':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            output, loss, loss_stats = model_with_loss(batch)

            inp_height, inp_width = batch['input'].shape[3],batch['input'].shape[4]
            c = np.array([inp_width / 2., inp_height / 2.], dtype=np.float32)
            s = max(inp_height, inp_width) * 1.0

            meta = {'c': c, 's': s,
                    'out_height': inp_height,
                    'out_width': inp_width}

            dets = post_process(output, meta)
            ret = merge_outputs([dets], num_classes=1, max_per_image=opt.K)
            results[im_id.numpy().astype(np.int32)[0]] = ret

            loss = loss.mean()
            batch_time.update(time.time() - end)

            print('phase=%s, epoch=%5d, iters=%d/%d,time=%0.4f, loss=%0.4f, hm_loss=%0.4f, wh_loss=%0.4f, off_loss=%0.4f' \
                  % (phase, epoch,iter_id+1,num_iters, time.time() - end,
                     loss.mean().cpu().detach().numpy(),
                     loss_stats['hm_loss'].mean().cpu().detach().numpy(),
                     loss_stats['wh_loss'].mean().cpu().detach().numpy(),
                     loss_stats['off_loss'].mean().cpu().detach().numpy()))
            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        stats1, _ = dataset.run_eval(results, opt.save_results_dir, 'latest')
        ret['time'] = 1 / 60.
        ret['ap50'] = stats1[1]

        return ret, results, stats1

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader, base_s, dataset):
        # return self.run_epoch('val', epoch, data_loader)

        return self.run_eval_epoch('val', epoch, data_loader, base_s, dataset)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)