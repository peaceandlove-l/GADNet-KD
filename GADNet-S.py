import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import torchvision.models as models
# from mymodelnew.Backbone.P2T.p2t import p2t_base
# from lsy.mymodelnew.Backbone.P2T.p2t import p2t_small
from Backbone.segformer.mix_transformer import mit_b2
# from toolbox.Mymodels.Baseline_lsy_seg.MLPDecoder import DecoderHead
from toolbox.Mymodels.DINO.SADG2L2Decoder import SADDecoder


class GeometryAwarePointConstraint(nn.Module):
    def __init__(self, sigma=0.1, weight=0.05):
        super().__init__()
        self.sigma = sigma
        self.weight = weight

    def forward(self, depth_feat):
        B, C, H, W = depth_feat.shape
        device = depth_feat.device

        # pseudo point cloud
        # z: (B, 1, H, W)
        z = depth_feat.mean(dim=1, keepdim=True)

        # meshgrid: (H, W)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )

        # 扩展到 (B, 1, H, W)
        x = x.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        y = y.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

        # 拼成 pseudo point cloud: (B, 3, H, W)
        pc = torch.cat([x, y, z], dim=1)

        # 3D gradient
        grad_h = torch.norm(pc[:, :, 1:, :] - pc[:, :, :-1, :], dim=1, keepdim=True)
        grad_w = torch.norm(pc[:, :, :, 1:] - pc[:, :, :, :-1], dim=1, keepdim=True)

        # geometry-aware soft weights
        w_h = torch.exp(-grad_h / self.sigma)
        w_w = torch.exp(-grad_w / self.sigma)

        # weighted anisotropic TV
        diff_h = depth_feat[:, :, 1:, :] - depth_feat[:, :, :-1, :]
        diff_w = depth_feat[:, :, :, 1:] - depth_feat[:, :, :, :-1]

        tv = (w_h * diff_h.pow(2)).mean() + (w_w * diff_w.pow(2)).mean()

        return depth_feat - self.weight * tv


def depth_to_pointcloud(depth_feat):
    """
    depth_feat: (B, C, H, W)
    return: pseudo point cloud coordinates (B, H*W, 3)
    """
    B, C, H, W = depth_feat.shape
    device = depth_feat.device

    # Use mean depth over channels as geometry proxy
    z = depth_feat.mean(dim=1)  # (B, H, W)

    # image grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    x = x.unsqueeze(0).expand(B, -1, -1)
    y = y.unsqueeze(0).expand(B, -1, -1)

    # (B, H, W, 3) → (B, HW, 3)
    pc = torch.stack([x, y, z], dim=-1).view(B, -1, 3)
    return pc


class PointWiseTV(nn.Module):
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight

    def forward(self, depth_feat):
        # depth_feat: (B, C, H, W)
        diff_h = depth_feat[:, :, 1:, :] - depth_feat[:, :, :-1, :]
        diff_w = depth_feat[:, :, :, 1:] - depth_feat[:, :, :, :-1]

        tv = (diff_h.pow(2).mean(dim=(1, 2, 3), keepdim=True) +
              diff_w.pow(2).mean(dim=(1, 2, 3), keepdim=True))

        return depth_feat - self.weight * tv


class DepthJumpSuppression(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, depth_feat):
        grad_h = torch.abs(depth_feat[:, :, 1:, :] - depth_feat[:, :, :-1, :])
        grad_w = torch.abs(depth_feat[:, :, :, 1:] - depth_feat[:, :, :, :-1])

        mask_h = grad_h < self.threshold
        mask_w = grad_w < self.threshold

        depth_feat_h = depth_feat[:, :, 1:, :] * mask_h
        depth_feat_w = depth_feat[:, :, :, 1:] * mask_w

        depth_feat = depth_feat.clone()
        depth_feat[:, :, 1:, :] = depth_feat_h
        depth_feat[:, :, :, 1:] = depth_feat_w

        return depth_feat


class LightweightPointCloudConstraint(nn.Module):
    """
    Inserted before BTV
    """

    def __init__(self):
        super().__init__()
        self.pw_tv = PointWiseTV(weight=0.05)
        self.jump = DepthJumpSuppression(threshold=0.1)

    def forward(self, depth_feat):
        depth_feat = self.pw_tv(depth_feat)
        depth_feat = self.jump(depth_feat)
        return depth_feat


class TotalVariationDenoising(nn.Module):
    """Total Variation Denoising module."""

    def __init__(self, weight=0.1):
        super(TotalVariationDenoising, self).__init__()
        self.weight = weight

    def forward(self, x):
        # Calculate the total variation
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        tv_h = torch.pow(diff_h, 2).mean(dim=(1, 2, 3), keepdim=True)
        tv_w = torch.pow(diff_w, 2).mean(dim=(1, 2, 3), keepdim=True)
        # Apply the total variation denoising
        return x + self.weight * (tv_h + tv_w)


class BilateralTotalVariation(nn.Module):
    """Bilateral Total Variation module using downsampled guidance maps."""

    def __init__(self, weight=0.1):
        super(BilateralTotalVariation, self).__init__()
        self.weight = weight

    def forward(self, x, guidance):
        # Downsample the guidance map to reduce computational load
        guidance_down = F.interpolate(guidance, scale_factor=0.5, mode='bilinear', align_corners=False)
        weight_h = torch.exp(-torch.abs(guidance_down[:, :, 1:, :] - guidance_down[:, :, :-1, :]))
        weight_w = torch.exp(-torch.abs(guidance_down[:, :, :, 1:] - guidance_down[:, :, :, :-1]))
        # Upsample weights to match the dimensions of the difference results
        weight_h_up = F.interpolate(weight_h, size=(x.size(2) - 1, x.size(3)), mode='bilinear', align_corners=False)
        weight_w_up = F.interpolate(weight_w, size=(x.size(2), x.size(3) - 1), mode='bilinear', align_corners=False)
        # Compute the bilateral total variation
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        btv_h = torch.pow(diff_h, 2) * weight_h_up
        btv_w = torch.pow(diff_w, 2) * weight_w_up
        btv = btv_h.mean(dim=(1, 2, 3), keepdim=True) + btv_w.mean(dim=(1, 2, 3), keepdim=True)
        return x + self.weight * btv


class PointCloudGuidedVariationalRefinement(nn.Module):
    """Module to fuse features using Total Variation Denoising and Bilateral Total Variation."""
    """PCVR."""

    def __init__(self, in_channels):
        super(PointCloudGuidedVariationalRefinement, self).__init__()
        self.tvd = TotalVariationDenoising()
        self.btv = BilateralTotalVariation()
        self.pc_constraint = LightweightPointCloudConstraint()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, rgb, depth):
        # Apply total variation denoising and bilateral total variation
        rgb = self.tvd(rgb)
        # 2.5D point cloud constraint BEFORE BTV
        depth = self.pc_constraint(depth)

        depth = self.btv(depth, rgb)
        # Fuse features
        fused = self.conv(torch.cat([rgb, depth], dim=1))
        return self.relu(fused)

class st(nn.Module):
    def __init__(self,num_class=41):
        super(st,self).__init__()
        self.backbone = mit_b2()
        # self.backbone = p2t_small()

        dim = [64, 128, 160, 256]
        self.decoder = SADDecoder(in_channels=dim)

        self.PCVR0 = PointCloudGuidedVariationalRefinement(dim[0])
        self.PCVR1 = PointCloudGuidedVariationalRefinement(dim[1])
        self.PCVR2 = PointCloudGuidedVariationalRefinement(dim[2])
        self.PCVR3 = PointCloudGuidedVariationalRefinement(dim[3])

        self.conv320to160 = nn.Conv2d(in_channels=320, out_channels=160, kernel_size=1)
        self.conv512to256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)



    def forward(self, rgb, dep):
        raw_rgb = rgb
        rgb = self.backbone(rgb)
        depth = self.backbone(dep)
        # print(rgb[0].shape)###([1, 16, 240, 320])
        # print(rgb[1].shape)###([1, 24, 120, 160])
        # print(rgb[2].shape)###([1, 32, 60, 80])
        # print(rgb[3].shape)###([1, 96, 30, 40])
        # print(rgb[4].shape)
        rgb[2] = self.conv320to160(rgb[2])
        depth[2] = self.conv320to160(depth[2])
        rgb[3] = self.conv512to256(rgb[3])
        depth[3] = self.conv512to256(depth[3])

        fuse0 = self.PCVR0(rgb[0], depth[0])
        # print('fuse0',fuse0.shape)
        fuse1 = self.PCVR1(rgb[1], depth[1])
        # print('fuse1',fuse1.shape)
        fuse2 = self.PCVR2(rgb[2], depth[2])
        # print('fuse2',fuse2.shape)
        fuse3 = self.PCVR3(rgb[3], depth[3])
        # print('fuse3',fuse3.shape)


        # fuse0 = rgb[0] + depth[0]
        # fuse1 = rgb[1] + depth[1]
        # fuse2 = rgb[2] + depth[2]
        # fuse3 = rgb[3] + depth[3]

        out = self.decoder(fuse0, fuse1, fuse2, fuse3)
        # out = self.Mlp_decoder(fuse0, fuse1, fuse2, fuse3)


        return out

    def load_pre_sa(self, pre_model):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.backbone.load_state_dict(new_state_dict3, strict=False)
        print('self.backbone_seg_mit loading')


if __name__ == '__main__':
    net = st().cuda()
    rgb = torch.randn([3, 3, 480, 640]).cuda()
    d = torch.randn([3, 3, 480, 640]).cuda()
    s = net(rgb, d)
    print(s.shape)
    print("==> Total params: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
    from toolbox.Mymodels.Baseline_lsy_seg.FLOP import CalParams
    CalParams(net, rgb, d)
    # print("s.shape:", s[0][-1].shape)
    # print("s.shape:", s[1][-1].shape)
    # print("s.shape:", s[1][-1].shape)
    # print("s.shape:", s[4][-1].shape)