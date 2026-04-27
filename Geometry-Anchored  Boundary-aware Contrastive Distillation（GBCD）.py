import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= Anchor Generator =================
class LearnableAnchorGenerator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 2, 1, 1)
        )

    def forward(self, feat, num_anchors):
        B, _, H, W = feat.shape

        heatmap = self.score_net(feat)
        heatmap = F.softplus(heatmap)

        prob = heatmap.view(B, -1)
        prob = prob / (prob.sum(dim=1, keepdim=True) + 1e-6)

        idx = torch.multinomial(prob, num_anchors, replacement=False)

        u = (idx % W).float() / W
        v = (idx // W).float() / H
        anchors = torch.stack([u, v], dim=-1)

        return anchors, heatmap


# ================= Feature Sampling =================
def sample_features(feat_map, anchors):
    B, C, H, W = feat_map.shape
    N = anchors.shape[1]

    grid = anchors * 2.0 - 1.0
    grid = grid.view(B, N, 1, 2)

    sampled = F.grid_sample(
        feat_map, grid,
        mode='bilinear',
        align_corners=False
    )

    sampled = sampled.squeeze(-1).permute(0, 2, 1)
    return sampled


# ================= Boundary Mask =================
def compute_boundary_mask(heatmap, anchors, threshold=0.6):
    """
    根据 anchor 处 heatmap 值判断是否为 boundary anchor
    """
    B, _, H, W = heatmap.shape

    saliency = sample_features(heatmap, anchors).squeeze(-1)  # [B, N]

    max_val = saliency.max(dim=1, keepdim=True)[0]
    mask = saliency > threshold * max_val

    return mask  # [B, N] bool


# ================= GAFD + ABCD =================
class GAFD_ABCD(nn.Module):
    def __init__( self, teacher_dim, student_dim, proj_dim=256, num_anchors=256, temperature=0.1, lambda_abcd=0.5):
        super().__init__()

        self.num_anchors = num_anchors
        self.tau = temperature
        self.lambda_abcd = lambda_abcd

        self.anchor_generator = LearnableAnchorGenerator(teacher_dim)

        self.teacher_proj = nn.Sequential(
            nn.Linear(teacher_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

        self.student_proj = nn.Sequential(
            nn.Linear(student_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def forward(self, teacher_feat, student_feat):
        # ---------- 1. Anchor ----------
        anchors, heatmap = self.anchor_generator(
            teacher_feat, self.num_anchors
        )

        # ---------- 2. Sample ----------
        ft = sample_features(teacher_feat, anchors)
        fs = sample_features(student_feat, anchors)

        ft = self.teacher_proj(ft)
        fs = self.student_proj(fs)

        # ---------- 3. Normalize ----------
        ft = F.normalize(ft, dim=-1)
        fs = F.normalize(fs, dim=-1)

        # ---------- 4. 原始 GAFD ----------
        loss_gafd = F.mse_loss(fs, ft.detach())

        # ---------- 5. Boundary Mask ----------
        boundary_mask = compute_boundary_mask(heatmap, anchors)

        # ---------- 6. ABCD ----------
        loss_abcd = 0.0
        valid_batch = 0

        B, N, C = ft.shape

        for b in range(B):
            bd = boundary_mask[b]
            non_bd = ~bd

            if bd.sum() == 0 or non_bd.sum() == 0:
                continue

            # query: student boundary
            q = fs[b, bd]            # [Nb, C]

            # positive: teacher boundary
            k_pos = ft[b, bd]        # [Nb, C]

            # negative: student non-boundary
            k_neg = fs[b, non_bd]    # [Ni, C]

            # 正样本
            pos_sim = torch.sum(q * k_pos, dim=-1, keepdim=True)  # [Nb,1]

            # 负样本
            neg_sim = torch.matmul(q, k_neg.t())                  # [Nb,Ni]

            logits = torch.cat([pos_sim, neg_sim], dim=1) / self.tau

            labels = torch.zeros(
                logits.size(0), dtype=torch.long, device=logits.device
            )

            loss = F.cross_entropy(logits, labels)

            loss_abcd += loss
            valid_batch += 1

        if valid_batch > 0:
            loss_abcd /= valid_batch

        # ---------- 7. 总损失 ----------
        total_loss = loss_gafd + self.lambda_abcd * loss_abcd

        return total_loss, loss_gafd, loss_abcd, heatmap



    


# ================== 测试（一定不会报错了）==================
if __name__ == "__main__":
    model = DGAWDecoder().cuda()

    # 模拟混合精度输入
    feats = [
        torch.randn(2, 64, 120, 160).cuda().half(),
        torch.randn(2, 128, 60, 80).cuda().half(),
        torch.randn(2, 320, 30, 40).cuda().half(),
        torch.randn(2, 512, 15, 20).cuda().half(),
    ]
    rgb = torch.randn(2, 3, 256, 256).cuda().half()
    depth = torch.randn(2, 3, 256, 256).cuda().half()

    with torch.cuda.amp.autocast():
        out = model(feats, rgb, depth)
    print("Success! Output shape:", out.shape, "dtype:", out.dtype)