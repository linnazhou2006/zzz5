import torch
from torch import nn
import torch.nn.functional as F
import os
from copy import deepcopy

# 导入原始的DSFNet作为我们的“专家”
from .DSFNet import DSFNet as DSFNet_expert
# 导入您项目中的 load_model 函数
from .stNet import load_model
# 导入分布式通信库，这是新forward方法所必需的
import torch.distributed as dist


class GumbelGatingNetwork(nn.Module):
    """
    使用Gumbel-Softmax的门控网络，支持Top-K专家选择。
    - 训练时使用 Gumbel-Softmax 实现对Top-K个专家的可导选择。
    - 评估时使用 ArgMax 实现对Top-K个专家的确定性选择。
    """

    def __init__(self, num_experts, top_k=1):
        """
        初始化门控网络。
        Args:
            num_experts (int): 专家总数。
            top_k (int): 每个输入需要激活的专家数量。
        """
        super(GumbelGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 一个轻量级的CNN，用于从输入中提取特征以决定专家权重
        self.backbone = nn.Sequential(
            # 输入形状: [B, C*T, H, W], 例如 [B, 15, H, W]
            nn.Conv2d(3 * 5, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输出特征图尺寸 H/4, W/4
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, self.num_experts)

    def forward(self, x):
        b, c, t, h, w = x.shape
        # 将时间和通道维度合并，以适应2D卷积: [B, C, T, H, W] -> [B, C*T, H, W]
        x_reshaped = x.view(b, c * t, h, w)

        features = self.backbone(x_reshaped)
        features = self.avg_pool(features)
        features = features.view(b, -1)

        # logits: [B, num_experts], 每个专家的原始分数
        logits = self.fc(features)

        # 选出 Top-K 的专家索引和对应的分数(logits)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)

        # 创建一个用于存储门控权重的稀疏张量
        sparse_gates = torch.zeros_like(logits)

        if self.training:
            # 训练时: 在 top-k 的 logits 上应用 Gumbel-Softmax
            # hard=True 会在前向传播时产生 one-hot 输出，但在反向传播时使用 softmax 的梯度
            gumbel_top_k = F.gumbel_softmax(top_k_logits, tau=1.0, hard=True, dim=-1)
            # 将 Gumbel-Softmax 的结果（近似0/1）放回稀疏张量的正确位置
            sparse_gates.scatter_(1, top_k_indices, gumbel_top_k)
        else:
            # 评估时: 直接将被选中的专家位置赋予权重
            # 这里我们使用 1.0 / self.top_k，实现对 top-k 个专家的等权重平均集成
            sparse_gates.scatter_(1, top_k_indices, 1.0 / self.top_k)

        # 辅助损失 (Load Balancing Loss)，鼓励门控网络均匀地使用所有专家
        expert_mask = torch.zeros_like(logits).scatter_(1, top_k_indices, 1)
        density = expert_mask.mean(dim=0)
        density_proxy = F.softmax(logits, dim=1).mean(dim=0)
        aux_loss = (density * density_proxy).sum() * self.num_experts

        return sparse_gates, top_k_indices, aux_loss


class Gumbel_MoE_DSFNet(nn.Module):
    """
    使用Gumbel-Softmax门控进行端到端训练的MoE模型。
    支持选择 Top-K 个专家，并使用稀疏计算以节省显存。
    """

    def __init__(self, heads, head_conv=128, num_experts=3, top_k=1, loss_coef=1e-2,
                 pretrained_paths=None, expert_modules=None):
        super(Gumbel_MoE_DSFNet, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss_coef = loss_coef
        self.heads = heads

        print(f"🚀 初始化 Gumbel-MoE-DSFNet 模型 (Top-K={self.top_k})")
        print(f"   - 门控机制: 训练时 Top-K Gumbel-Softmax, 评估时 Top-K ArgMax")
        print(f"   - 专家总数: {self.num_experts}")

        self.gating_network = GumbelGatingNetwork(self.num_experts, self.top_k)

        if expert_modules is not None:
            print("   - 使用外部提供的专家模块列表。")
            if len(expert_modules) != self.num_experts:
                raise ValueError(f"提供的专家模块数量({len(expert_modules)})与num_experts({self.num_experts})不匹配。")
            self.experts = expert_modules
        elif pretrained_paths is not None:
            print("   - 根据路径列表创建并初始化专家。")

            # --- [核心修复: 健壮的专家初始化逻辑] ---
            self.experts = nn.ModuleList()
            base_experts = []

            # 1. 首先加载所有基础的预训练专家
            for i, path in enumerate(pretrained_paths):
                print(f"   - 加载基础专家 {i + 1}/{len(pretrained_paths)} 从: '{os.path.basename(path)}'")
                expert_model = DSFNet_expert(heads, head_conv)
                if os.path.exists(path):
                    expert_model = load_model(expert_model, path)
                else:
                    print(f"   - ⚠️ 警告: 路径不存在 {path}。专家将使用随机初始化权重。")
                base_experts.append(expert_model)

            if not base_experts:
                raise ValueError("预训练路径列表为空或所有路径均无效，无法创建专家。")

            # 2. 通过复制基础专家来填满 expert_list 至 num_experts 的数量
            for i in range(self.num_experts):
                # 轮流从 base_experts 中选取一个进行复制
                base_to_copy = base_experts[i % len(base_experts)]
                self.experts.append(deepcopy(base_to_copy))
            # --- [修复结束] ---

        else:
            raise ValueError("必须提供 'pretrained_paths' 或 'expert_modules'。")

        print(f"✅ 所有 {len(self.experts)} 个专家模块已设置。")

    def forward(self, x):
        """
        高效的、稀疏计算的前向传播方法。
        """
        # 1. 从门控网络获取稀疏权重和被选中的专家索引
        sparse_gates, top_k_indices, aux_loss = self.gating_network(x)

        batch_size = x.shape[0]

        # 2. 初始化一个空的输出张量字典
        with torch.no_grad():
            dummy_output = self.experts[0](x)[0]
        final_outputs = {head: torch.zeros(batch_size, *dummy_output[head].shape[1:], device=x.device) for head in
                         self.heads}

        # 3. 遍历批次中的每个样本，只计算被激活的专家
        for i in range(batch_size):
            # 获取当前样本选择的专家索引 (例如: [2, 7, 11] for top_k=3)
            active_indices = top_k_indices[i]
            # 获取当前样本的输入，并保持批次维度为1
            sample_input = x[i].unsqueeze(0)

            # 获取当前样本对应的门控权重 (例如: [0.0, 1.0, 0.0, ..., 1.0, ..., 1.0, ...])
            active_gates = sparse_gates[i][active_indices]

            # 4. 遍历被激活的k个专家
            for k in range(self.top_k):
                expert_idx = active_indices[k].item()
                gate_val = active_gates[k]

                if gate_val == 0:
                    continue

                # 只计算这个被激活的专家的输出
                expert_output_dict = self.experts[expert_idx](sample_input)[0]

                # 使用门控权重对专家的输出进行加权，并累加到最终结果中
                for head_name, head_tensor in expert_output_dict.items():
                    final_outputs[head_name][i] += gate_val * head_tensor.squeeze(0)

        # 5. 返回最终的预测结果和用于训练的辅助损失
        return [final_outputs], self.loss_coef * aux_loss