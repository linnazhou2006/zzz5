import torch
from torch import nn
import torch.nn.functional as F
import os

# 导入原始的DSFNet和load_model函数
from .DSFNet import DSFNet as DSFNet_expert
from .stNet import load_model


class ConvGatingNetwork(nn.Module):
    """
    一个新的、更强大的门控网络。
    使用一个64通道的卷积层作为主干，后面接FC层。
    输入：[B, C, T, H, W]
    输出：稀疏权重 [B, num_experts] (只有一个元素为1，其余为0)
    """

    def __init__(self, num_experts, top_k=1):
        super(ConvGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 主干网络: Conv -> BN -> ReLU -> MaxPool
        # 输入通道为3(RGB), 时间维度为5
        # 我们使用2D卷积处理时间序列，将时间维度视为通道的一部分
        # (B, C, T, H, W) -> (B, C*T, H, W)
        self.backbone = nn.Sequential(
            nn.Conv2d(3 * 5, 64, kernel_size=7, stride=2, padding=3, bias=False),  # in_channels = 3 * 5 = 15
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # H/4, W/4
        )

        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, self.num_experts)

    def forward(self, x):
        b, c, t, h, w = x.shape
        # 将时间和通道维度合并，以适应2D卷积
        x_reshaped = x.view(b, c * t, h, w)

        features = self.backbone(x_reshaped)
        features = self.avg_pool(features)
        features = features.view(b, -1)

        # logits: [B, num_experts], 这是每个专家的分数
        logits = self.fc(features)

        # ==================== [核心修改: 输出0/1] ====================
        # 选出 Top-K 的专家索引
        _, top_k_indices = torch.topk(logits, self.top_k, dim=1)

        # 生成一个全零的one-hot编码张量
        sparse_gates_one_hot = torch.zeros_like(logits)
        # 将被选中专家的位置设置为1
        sparse_gates_one_hot.scatter_(1, top_k_indices, 1)
        # ==========================================================

        # 辅助损失 (Load Balancing Loss) - 保持不变
        expert_mask = sparse_gates_one_hot
        density = expert_mask.mean(dim=0)
        density_proxy = F.softmax(logits, dim=1).mean(dim=0)
        aux_loss = (density * density_proxy).sum() * self.num_experts

        # 返回 one-hot 编码的稀疏门控、专家索引和辅助损失
        return sparse_gates_one_hot, top_k_indices, aux_loss


class FrozenConvMoE_DSFNet(nn.Module):
    """
    使用新的卷积门控网络，并冻结专家。
    """

    def __init__(self, heads, head_conv=128, num_experts=3, top_k=1, loss_coef=1e-2, pretrained_paths=None):
        super(FrozenConvMoE_DSFNet, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_coef = loss_coef
        self.heads = heads

        print("🚀 初始化 Frozen-Conv-MoE-DSFNet 模型")
        print(f"   - 门控网络: Conv-64F Backbone")
        print(f"   - 专家总数: {self.num_experts}")
        print(f"   - 激活专家数 (Top-K): {self.top_k} (硬选择)")

        # 1. 实例化新的卷积门控网络
        self.gating_network = ConvGatingNetwork(self.num_experts, self.top_k)

        # 2. 实例化、加载并冻结专家 (逻辑与之前相同)
        self.experts = nn.ModuleList()
        if pretrained_paths is None or len(pretrained_paths) != self.num_experts:
            raise ValueError(f"必须提供一个包含 {self.num_experts} 个预训练模型路径的列表。")

        for i, path in enumerate(pretrained_paths):
            print(f"   - 加载并冻结专家 {i + 1}/{self.num_experts} 从: '{os.path.basename(path)}'")
            expert_model = DSFNet_expert(heads, head_conv)

            if os.path.exists(path):
                expert_model = load_model(expert_model, path)
            else:
                print(f"   - ⚠️ 警告: 路径不存在 {path}。")

            expert_model.eval()
            for param in expert_model.parameters():
                param.requires_grad = False

            self.experts.append(expert_model)

        print("✅ 所有专家已加载并冻结。")

    def forward(self, x):
        batch_size = x.shape[0]

        # 1. 获取门控输出 (0/1编码)
        sparse_gates_one_hot, top_k_indices, aux_loss = self.gating_network(x)

        # 2. 初始化最终输出
        final_outputs = {head: torch.zeros(batch_size, *self.experts[0](x)[0][head].shape[1:], device=x.device) for head
                         in self.heads}

        # 3. 遍历批次中的每个样本
        for i in range(batch_size):
            # 获取被选中的专家索引 (因为top_k=1, 这里只有一个索引)
            expert_idx = top_k_indices[i][0].item()
            sample_input = x[i].unsqueeze(0)

            # 4. 只运行被选中的那个专家
            with torch.no_grad():
                expert_output_dict = self.experts[expert_idx](sample_input)[0]

            # 5. 直接将专家输出作为该样本的最终输出
            for head_name, head_tensor in expert_output_dict.items():
                final_outputs[head_name][i] = head_tensor.squeeze(0)

        return [final_outputs], self.loss_coef * aux_loss