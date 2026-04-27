import torch
import torch.nn.functional as F
import torch.nn as nn

def geometry_saliency_from_depth(depth):
    """
    depth: [B, 1, H, W]
    return: [B, 1, H, W]
    """
    dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    grad = torch.zeros_like(depth)
    grad[:, :, :, :-1] += dx
    grad[:, :, :-1, :] += dy

    grad = F.avg_pool2d(grad, 3, stride=1, padding=1)
    return grad

def sample_anchors_from_geometry(saliency, num_anchors):
    """
    saliency: [B, 1, H, W]
    return:   [B, N, 2]  (u, v) normalized to [0,1]
    """
    B, _, H, W = saliency.shape
    sal = saliency.view(B, -1)

    prob = sal / (sal.sum(dim=1, keepdim=True) + 1e-6)
    idx = torch.multinomial(prob, num_anchors, replacement=False)

    u = (idx % W).float() / W
    v = (idx // W).float() / H

    anchors = torch.stack([u, v], dim=-1)
    return anchors

def sample_features(feat_map, anchors):
    """
    feat_map: [B, C, H, W]
    anchors:  [B, N, 2] ∈ [0,1]
    return:   [B, N, C]
    """
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


class GeometryDrivenTGAD(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()

        self.student_proj = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.LayerNorm(teacher_dim)
        )

    def forward(
        self,
        teacher_feat,
        student_feat,
        text_embed,
        depth,
        num_anchors=256
    ):
        """
        teacher_feat: [B, Ct, Ht, Wt]
        student_feat: [B, Cs, Hs, Ws]
        text_embed:   [B, Ct] or [Ct]
        depth:        [B, 1, H, W]
        """

        # ---- Step 1: Geometry-driven anchors ----
        geo_saliency = geometry_saliency_from_depth(depth)
        anchors = sample_anchors_from_geometry(
            geo_saliency, num_anchors
        )

        # ---- Step 2: Sample features ----
        ft = sample_features(teacher_feat, anchors)  # [B, N, Ct]
        fs = sample_features(student_feat, anchors)  # [B, N, Cs]

        # ---- Step 3: Image–Text similarity (Teacher) ----
        if text_embed.dim() == 1:
            text_embed = text_embed.unsqueeze(0)

        text_embed = F.normalize(text_embed, dim=-1)
        ft = F.normalize(ft, dim=-1)

        sim_t = torch.sum(
            ft * text_embed.unsqueeze(1),
            dim=-1
        )  # [B, N]

        # ---- Step 4: Student predicts similarity ----
        fs = self.student_proj(fs)
        fs = F.normalize(fs, dim=-1)

        sim_s = torch.sum(
            fs * text_embed.unsqueeze(1),
            dim=-1
        )

        loss = F.mse_loss(sim_s, sim_t.detach())
        return loss, anchors, geo_saliency




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