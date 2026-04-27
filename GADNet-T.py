"""
极简小目标检测模型 - 基于DINOv3
去除所有复杂机制，只保留核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffusionFusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, hidden_dim=32):
        """
        Diffusion-based Lightweight Fusion Module
        :param in_channels: 输入通道数 C
        :param hidden_dim: 隐藏维度 (默认32，参数极少)
        """
        super().__init__()

        # 1x1 投影到共同空间 (超轻: C*hd params)
        self.proj1 = nn.Conv2d(in_channels1, hidden_dim, 1)
        self.proj2 = nn.Conv2d(in_channels2, hidden_dim, 1)

        # 极简UNet-like 噪声预测器
        self.noise_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim,
                      kernel_size=3, padding=1, groups=max(1, hidden_dim // 8)),  # Depthwise
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1)
        )

        # 输出投影 (可选)
        self.out_channels = out_channels
        self.out_proj = nn.Conv2d(hidden_dim, self.out_channels, 1)

        # Diffusion固定参数
        self.register_buffer('sqrt_alpha', torch.tensor(0.9 ** 0.5))
        self.register_buffer('sigma', torch.tensor((1 - 0.9) ** 0.5))


    def forward(self, feat1, feat2):
        """
        :param feat1: [B, C1, H, W]
        :param feat2: [B, C2, H, W] (H/W必须相同!)
        :return: fused [B, out_channels, H, W]
        """
        # Step1: 投影到共同空间
        f1 = self.proj1(feat1)  # [B, hd, H, W]
        f2 = self.proj2(feat2)

        # Step2: Diffusion - f2注入噪声
        noise = f2 * self.sigma
        noisy_feat = f1 * self.sqrt_alpha + noise

        # Step3: 条件预测噪声
        cat_feat = torch.cat([noisy_feat, f2], dim=1)
        pred_noise = self.noise_predictor(cat_feat)

        # Step4: 单步去噪
        denoised = (noisy_feat - self.sigma * pred_noise) / self.sqrt_alpha

        # Step5: 输出 + 残差稳定
        fused = self.out_proj(denoised)
        # Step4: 残差 + 软融合 (稳定训练)
        # fused = feat1 + 0.3 * (denoised - feat1) + 0.3 * feat2
        return fused

class DepthToTextPromptFuser(nn.Module):
    def __init__(self, embed_dim=24, fx=500.0, fy=500.0, depth_scale=1000.0) -> None:
        super(DepthToTextPromptFuser, self).__init__()
        self.fx = fx
        self.fy = fy
        self.depth_scale = depth_scale
        # Self-attention for adaptive feature refinement
        self.self_attn = nn.MultiheadAttention(embed_dim=3, num_heads=1, batch_first=True)
        # Vocabulary for simple text embedding
        self.vocab = [
            'an', 'indoor', 'scene', 'with', 'close', 'far', 'objects', 'and', 'complex', 'simple',
            'geometry', 'has', 'many', 'horizontal', 'surfaces', 'vertical', 'walls', 'low', 'high', 'variance'
        ]
        self.vocab_size = len(self.vocab)
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)

    def _compute_normals(self, depth_map):
        B, C, H, W = depth_map.shape
        # assert C == 1, "Depth map should have 1 channel"
        device = depth_map.device
        depth = depth_map.mean(dim=1, keepdim=False) / self.depth_scale  # [B, H, W]

        # Default intrinsics
        cx = W / 2.0
        cy = H / 2.0
        fx = torch.full((B,), self.fx, device=device)
        fy = torch.full((B,), self.fy, device=device)
        cx = torch.full((B,), cx, device=device)
        cy = torch.full((B,), cy, device=device)

        # Pixel coordinates
        u, v = torch.meshgrid(torch.arange(W, dtype=torch.float, device=device),
                              torch.arange(H, dtype=torch.float, device=device),
                              indexing='xy')
        u = u.unsqueeze(0).expand(B, -1, -1)
        v = v.unsqueeze(0).expand(B, -1, -1)

        # Backproject to 3D
        X = (u - cx.unsqueeze(1).unsqueeze(1)) * depth / fx.unsqueeze(1).unsqueeze(1)
        Y = (v - cy.unsqueeze(1).unsqueeze(1)) * depth / fy.unsqueeze(1).unsqueeze(1)
        Z = depth
        pointcloud = torch.stack([X, Y, Z], dim=1)  # [B, 3, H, W]

        # Compute normals
        p_center = pointcloud[:, :, 1:-1, 1:-1]
        p_right = pointcloud[:, :, 1:-1, 2:]
        p_down = pointcloud[:, :, 2:, 1:-1]
        vec1 = p_right - p_center
        vec2 = p_down - p_center
        cross = torch.cross(vec1, vec2, dim=1)
        normals_inner = F.normalize(cross, dim=1)  # [B, 3, H-2, W-2]

        normals = torch.zeros(B, 3, H, W, device=device)
        normals[:, :, 1:-1, 1:-1] = normals_inner

        # Adaptive refinement with self-attention
        normals_flat = normals.view(B, 3, H * W).permute(0, 2, 1)  # [B, H*W, 3]
        normals_adaptive, _ = self.self_attn(normals_flat, normals_flat, normals_flat)
        normals_adaptive = normals_adaptive.permute(0, 2, 1).view(B, 3, H, W)  # [B, 3, H, W]

        return normals_adaptive

    def _generate_text_prompt(self, depth, normals):
        B = depth.shape[0]
        if depth.ndim == 4:
            depth = depth.mean(dim=1)

        # [B]
        avg_depth = depth.mean(dim=[1, 2])
        var_depth = depth.var(dim=[1, 2])

        prompts = []
        # avg_depth = depth.mean(dim=[1, 2, 3]).squeeze()  # [B]
        # print('avg_depth',avg_depth.shape)
        # var_depth = depth.var(dim=[1, 2, 3]).squeeze()  # [B]
        # print('var_depth',var_depth.shape)


        # Analyze normals for surface types
        normal_abs = normals.abs()  # [B, 3, H, W]
        dominant_dir = normal_abs.mean(dim=[2, 3])  # [B, 3] average absolute normal components

        for b in range(B):
            avg_val = avg_depth[b].item() if avg_depth[b].numel() == 1 else avg_depth[b].mean().item()
            var_val = var_depth[b].item() if var_depth[b].numel() == 1 else var_depth[b].mean().item()

            dist = "close" if avg_val < 5 else "far"
            compl = "complex" if var_val > 0.5 else "simple"
            horiz = "many horizontal surfaces" if dominant_dir[b, 1] > 0.5 else ""  # Assuming Y is up
            vert = "many vertical walls" if (dominant_dir[b, 0] > 0.5 or dominant_dir[b, 2] > 0.5) else ""
            prompt_parts = [ "an indoor scene with", dist, "objects and", compl, "geometry"]
            if horiz:
                prompt_parts.append("has " + horiz)
            if vert:
                prompt_parts.append("has " + vert)
            prompt = " ".join(prompt_parts)
            prompts.append(prompt)
        return prompts

    def forward(self, depth_map):
        # Assume depth_map [B,1,H,W], rgb_features [B,C,H,W]
        B, C_rgb, H, W = depth_map.shape
        normals = self._compute_normals(depth_map)
        prompts = self._generate_text_prompt(depth_map, normals)

        # Tokenize and embed prompts
        token_ids = []
        for p in prompts:
            words = p.lower().split()
            ids = [self.vocab.index(w) for w in words if w in self.vocab] or [0]  # Default to first if none
            token_ids.append(torch.tensor(ids, device=depth_map.device))

        text_embs = []
        for ids in token_ids:
            emb = self.embedding(ids)  # [len, embed_dim]
            emb = emb.mean(dim=0)  # [embed_dim]
            text_embs.append(emb)

        text_emb = torch.stack(text_embs)  # [B, embed_dim]
        text = text_emb.unsqueeze(2).unsqueeze(3).expand(B, -1, H, W)  # [B, embed_dim, H, W]

        return text_emb, text

# ===== LoRA 层定义 =====
class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Linear(linear.in_features, rank, bias=False)
        self.B = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.linear(x) + self.scaling * self.B(self.A(x))


class MoELoRALinear(nn.Module):
    def __init__(self, linear, num_experts=4, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.num_experts = num_experts
        self.experts = nn.ModuleList([LoRALinear(linear, rank, alpha) for _ in range(num_experts)])
        # 动态路由器：基于输入特征选择专家（适用于多模态输入，如RGB+深度）
        self.router = nn.Linear(linear.in_features, num_experts)
        # GFP 集成（创新点2：几何引导特征传播）
        self.gfp_cross_attn = nn.MultiheadAttention(linear.in_features, num_heads=4,
                                                    batch_first=True)  # 简单跨注意力 for 几何增强

    def forward(self, x, geom_features=None):  # geom_features: 可选几何输入，如深度/点云坐标
        base_out = self.linear(x)

        # 路由器计算权重（动态选择专家，适用于室内多类别）
        router_logits = self.router(x.mean(dim=1))  # 平均池化作为全局上下文
        router_weights = F.softmax(router_logits, dim=-1).unsqueeze(1)  # [B, 1, num_experts]

        # 专家输出融合
        expert_out = torch.zeros_like(base_out)
        for i, expert in enumerate(self.experts):
            expert_delta = expert.scaling * expert.B(expert.A(x))
            expert_out += router_weights[:, :, i].unsqueeze(-1) * expert_delta

        # GFP：如果提供几何特征，使用跨注意力增强（针对室内点云/深度数据集）
        if geom_features is not None:
            geom_enhanced, _ = self.gfp_cross_attn(x, geom_features, geom_features)  # 查询x，键/值=geom
            expert_out += geom_enhanced  # 融合几何引导

        return base_out + expert_out


# 原 LoRAMultiheadAttention 类（修改为支持MoE-LoRA）
class LoRAMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None,
                 lora_rank=8, lora_alpha=16, num_experts=4):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first,
                         device, dtype)

        self._qkv_same_embed_dim = False

        # 替换为 MoELoRALinear（集成创新点1和2）
        self.q_proj = MoELoRALinear(nn.Linear(self.embed_dim, self.embed_dim, bias=bias), num_experts=num_experts,
                                    rank=lora_rank, alpha=lora_alpha)
        self.k_proj = MoELoRALinear(nn.Linear(self.kdim, self.embed_dim, bias=bias), num_experts=num_experts,
                                    rank=lora_rank, alpha=lora_alpha)
        self.v_proj = MoELoRALinear(nn.Linear(self.vdim, self.embed_dim, bias=bias), num_experts=num_experts,
                                    rank=lora_rank, alpha=lora_alpha)
        self.out_proj = MoELoRALinear(nn.Linear(self.embed_dim, self.embed_dim, bias=bias), num_experts=num_experts,
                                      rank=lora_rank, alpha=lora_alpha)

        # 初始化权重
        nn.init.xavier_uniform_(self.q_proj.linear.weight)
        nn.init.xavier_uniform_(self.k_proj.linear.weight)
        nn.init.xavier_uniform_(self.v_proj.linear.weight)
        nn.init.xavier_uniform_(self.out_proj.linear.weight)

        if bias:
            nn.init.constant_(self.q_proj.linear.bias, 0)
            nn.init.constant_(self.k_proj.linear.bias, 0)
            nn.init.constant_(self.v_proj.linear.bias, 0)
            nn.init.constant_(self.out_proj.linear.bias, 0)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False, geom_features=None):  # 添加geom_features支持GFP
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        q = self.q_proj(query, geom_features)
        k = self.k_proj(key, geom_features)
        v = self.v_proj(value, geom_features)

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            q, k, v, self.embed_dim, self.num_heads,
            self.dropout, self.bias_k, self.bias_v,
            self.add_zero_attn, self.dropout, self.out_proj.linear.weight.shape[0],
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=False,
            q_proj_weight=None, k_proj_weight=None, v_proj_weight=None,
            static_k=None, static_v=None, average_attn_weights=average_attn_weights,
            is_causal=is_causal)

        attn_output = self.out_proj(attn_output, geom_features)

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            return attn_output, attn_output_weights
        else:
            return attn_output


# 原 apply_lora_to_vit 函数（未修改，但可与新函数结合）
def apply_lora_to_vit(model, rank=8, alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']):
    """
    递归地将 LoRA 应用到 ViT 的指定线性层（注意力投影和 FFN）。
    假设 model 是 Vision Transformer，如 DINOv2/DINOv3。
    """
    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention) or (
                hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear)):
            if isinstance(module, nn.MultiheadAttention):
                module = LoRAMultiheadAttention(
                    embed_dim=module.embed_dim, num_heads=module.num_heads, dropout=module.dropout,
                    bias=module.in_proj_bias is not None, kdim=module.kdim, vdim=module.vdim,
                    batch_first=module.batch_first,
                    lora_rank=rank, lora_alpha=alpha
                )
            else:
                if 'qkv' in name or any(t in name for t in target_modules):
                    setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
        elif isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
        else:
            apply_lora_to_vit(module, rank=rank, alpha=alpha, target_modules=target_modules)
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for p in [module.A.parameters(), module.B.parameters()]:
                for pp in p:
                    pp.requires_grad = True
        elif isinstance(module, LoRAMultiheadAttention):
            for proj in [module.q_proj, module.k_proj, module.v_proj, module.out_proj]:
                for p in [proj.A.parameters(), proj.B.parameters()]:
                    for pp in p:
                        pp.requires_grad = True


# ===== 整体模型（添加 LoRA 到整个 DINO backbone） =====
class LoRADinoWithText(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        REPO_DIR = '/data/Lsy/sam/lsy/mymodelnew/toolbox/Mymodels/DINO/dinov3'
        dinov3_backbone = torch.hub.load(
            REPO_DIR, 'dinov3_vitb16',
            source='local',
            weights='/data/Lsy/sam/lsy/mymodelnew/toolbox/Mymodels/DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        )
        self.backbone = dinov3_backbone
        backbone_dim = 768
        # 新增：应用 LoRA 到整个 backbone（移除简单冻结）
        apply_lora_to_vit(self.backbone)

        self.textprompt = DepthToTextPromptFuser()
        self.fusion = DiffusionFusion(backbone_dim, 24, num_classes)  # Output fused feature

        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear')

    def forward(self, x, depth):
        B,_,H,W = x.shape

        # 1️ 使用带 LoRA 的 backbone 提取特征（移除 no_grad，因为现在可训练 LoRA）
        r = self.backbone.forward_features(x)
        r = r['x_norm_patchtokens']
        B, N, C = r.shape
        r = r.reshape(B, H//16, W//16, C).permute(0, 3, 1, 2)

        d = self.backbone.forward_features(depth)
        d = d['x_norm_patchtokens']
        B, N, C = d.shape
        d = d.reshape(B, H // 16, W // 16, C).permute(0, 3, 1, 2)

        text_emb, text = self.textprompt(d)#######text_emb[B,dim], text[B,C,H,W]

        fuse = self.fusion(r,text)

        return self.up16(fuse)


# 测试代码
if __name__ == '__main__':
    print("="*60)
    print("测试 Dino")
    print("="*60)

    # 创建模拟的backbone
    # REPO_DIR = '/data/Lsy/sam/lsy/mymodelnew/toolbox/Mymodels/DINO/dinov3'
    # dinov3_backbone = torch.hub.load(
    #     REPO_DIR, 'dinov3_vitl16',
    #     source='local',
    #     weights='/data/Lsy/sam/lsy/mymodelnew/toolbox/Mymodels/DINO/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    # )

    # 创建模型
    model = LoRADinoWithText(num_classes=41)

    # 测试前向传播
    x = torch.randn(1, 3, 480, 640)

    print(f"\n输入: {x.shape}")

    # cls_pred, reg_pred = model(x)
    cls_pred = model(x,x)
    print(f"输出:")
    print(f"  Cls pred: {cls_pred.shape}")
    # print(f"  Reg pred: {reg_pred.shape}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"\n参数统计:")
    print(f"  总参数: {total_params:.2f}M")
    print(f"  可训练参数: {trainable_params:.2f}M")

    print("\n✅ 测试通过!")
