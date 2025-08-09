import torch
from torch import nn
import torch.nn.functional as F
import os

# 导入原始的DSFNet作为我们的“专家”
from .DSFNet import DSFNet as DSFNet_expert


class SparseGatingNetwork(nn.Module):
    """
    稀疏门控网络：为每个专家生成一个分数（logit），并选择分数最高的Top-K个专家。
    """

    def __init__(self, num_experts, top_k):
        super(SparseGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 与之前相同，一个轻量级的CNN来提取全局特征
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, self.num_experts)

    def forward(self, x):
        b = x.shape[0]
        features = self.feature_extractor(x)
        features = self.avg_pool(features)
        features = features.view(b, -1)

        # logits: [B, num_experts], 这是每个专家的分数
        logits = self.fc(features)

        # 选出 Top-K 的专家
        # top_k_gates: 权重, top_k_indices: 索引
        top_k_gates, top_k_indices = torch.topk(logits, self.top_k, dim=1)

        # 稀疏门控：创建一个全零张量，只在被选中的专家位置上填充权重
        # 我们使用softmax来归一化这 top_k 个权重
        sparse_gates = torch.zeros_like(logits)
        sparse_gates.scatter_(1, top_k_indices, F.softmax(top_k_gates, dim=1))

        # 辅助损失 (Load Balancing Loss)
        # 这一步对于稀疏MoE至关重要，它鼓励门控网络使用不同的专家
        # 计算每个专家被选中的频率
        expert_mask = torch.zeros_like(logits)
        expert_mask.scatter_(1, top_k_indices, 1)  # 被选中的专家位置为1

        density = expert_mask.mean(dim=0)
        density_proxy = F.softmax(logits, dim=1).mean(dim=0)
        aux_loss = (density * density_proxy).sum() * self.num_experts

        return sparse_gates, top_k_indices, aux_loss


class SparseEnsembleMoE_DSFNet(nn.Module):
    """
    稀疏模型集成式MoE DSFNet
    """

    def __init__(self, heads, head_conv=128, num_experts=15, top_k=3, loss_coef=1e-2, pretrained_paths=None):
        super(SparseEnsembleMoE_DSFNet, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_coef = loss_coef
        self.heads = heads

        print(f"🚀 初始化稀疏 Ensemble-MoE-DSFNet 模型")
        print(f"   - 专家总数: {self.num_experts}")
        print(f"   - 激活专家数 (Top-K): {self.top_k}")

        # 1. 实例化稀疏门控网络
        self.gating_network = SparseGatingNetwork(self.num_experts, self.top_k)

        # 2. 实例化并加载预训练的DSFNet专家
        self.experts = nn.ModuleList()
        if pretrained_paths is None or len(pretrained_paths) != 3:
            raise ValueError("必须提供一个包含3个预训练模型路径的列表 `pretrained_paths`。")

        experts_per_pth = self.num_experts // len(pretrained_paths)
        print(f"   - 每个预训练模型将被用于初始化 {experts_per_pth} 个专家。")

        for i, path in enumerate(pretrained_paths):
            if not os.path.exists(path):
                print(f"⚠️ 警告: 预训练模型路径不存在: {path}。将使用随机初始化的专家。")
                state_dict = None
            else:
                print(f"   - 正在从 '{os.path.basename(path)}' 加载权重...")
                checkpoint = torch.load(path, map_location='cpu')
                # 兼容不同的checkpoint格式
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

            for j in range(experts_per_pth):
                expert_model = DSFNet_expert(heads, head_conv)
                if state_dict:
                    # 清理 'module.' 前缀（如果存在）
                    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    expert_model.load_state_dict(clean_state_dict, strict=False)

                self.experts.append(expert_model)

        print("✅ 所有专家已成功初始化。")

    def forward(self, x):
        batch_size = x.shape[0]

        # 1. 通过门控网络获取Top-K专家的权重和索引
        sparse_gates, top_k_indices, aux_loss = self.gating_network(x)

        # 2. 初始化用于存储激活专家输出的列表
        expert_outputs_list = []

        # 3. 只计算被选中的专家的输出
        for i in range(batch_size):
            # 获取当前样本选择的专家索引
            active_indices = top_k_indices[i]
            # 获取当前样本的输入
            sample_input = x[i].unsqueeze(0)

            # 分别计算每个激活专家的输出
            active_outputs = [self.experts[idx](sample_input)[0] for idx in active_indices]
            expert_outputs_list.append(active_outputs)

        # 4. 整合结果：对激活专家的输出进行加权平均
        final_outputs = {head: torch.zeros(batch_size, *expert_outputs_list[0][0][head].shape[1:], device=x.device) for
                         head in self.heads}

        for i in range(batch_size):
            active_indices = top_k_indices[i]
            active_gates = sparse_gates[i][active_indices]  # 获取非零的权重

            for k in range(self.top_k):
                expert_output_dict = expert_outputs_list[i][k]
                gate_val = active_gates[k]

                for head_name, head_tensor in expert_output_dict.items():
                    final_outputs[head_name][i] += gate_val * head_tensor.squeeze(0)

        final_result = [final_outputs]

        return final_result, self.loss_coef * aux_loss