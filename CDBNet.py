import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d # 处理斜向管线

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=32):
        super().__init__()
        #groups = min(groups, out_ch)
        #while out_ch % groups != 0:
            #groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)

class DetailInjectionStem(nn.Module):
    def __init__(self, out_ch=256):
        super().__init__()

        self.stem1 = ConvGNAct(3, 64, kernel_size=3, stride=2, padding=1)      # H/2
        self.stem2 = ConvGNAct(64, 128, kernel_size=3, stride=2, padding=1)    # H/4
        self.stem3 = ConvGNAct(128, out_ch, kernel_size=3, stride=2, padding=1) # H/8

        self.proj_p2 = ConvGNAct(128, out_ch, kernel_size=1, stride=1, padding=0)
        self.proj_p3 = ConvGNAct(out_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        d1 = self.stem1(x)      # H/2
        d2 = self.stem2(d1)     # H/4
        d3 = self.stem3(d2)     # H/8

        detail_p2 = self.proj_p2(d2)
        detail_p3 = self.proj_p3(d3)

        return detail_p2, detail_p3

class CrossLayerAdaptiveFusion(nn.Module):
    def __init__(self, embed_dim=1024, out_ch=256):
        super().__init__()

        self.proj_s = ConvGNAct(embed_dim, out_ch, kernel_size=1, stride=1, padding=0)
        self.proj_m = ConvGNAct(embed_dim, out_ch, kernel_size=1, stride=1, padding=0)
        self.proj_d = ConvGNAct(embed_dim, out_ch, kernel_size=1, stride=1, padding=0)

        # 为每个空间位置、每个通道生成三层权重
        self.gate = nn.Conv2d(out_ch * 3, out_ch * 3, kernel_size=1)

        self.refine = nn.Sequential(
            ConvGNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            ConvGNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, f_shallow, f_mid, f_deep):
        fs = self.proj_s(f_shallow)
        fm = self.proj_m(f_mid)
        fd = self.proj_d(f_deep)

        feat_cat = torch.cat([fs, fm, fd], dim=1)

        b, _, h, w = feat_cat.shape
        c = fs.shape[1]

        weight = self.gate(feat_cat)
        weight = weight.view(b, 3, c, h, w)
        weight = torch.softmax(weight, dim=1)

        feat_stack = torch.stack([fs, fm, fd], dim=1)

        fused = (weight * feat_stack).sum(dim=1)

        return self.refine(fused)

class CDPFA(nn.Module): #Cross-layer Detail-preserving Foundation Pyramid Adapter
    def __init__(self, embed_dim=1024, out_ch=256):
        super().__init__()

        self.cross_layer_fusion = CrossLayerAdaptiveFusion(
            embed_dim=embed_dim,
            out_ch=out_ch
        )

        self.detail_stem = DetailInjectionStem(out_ch=out_ch)

        # 从 H/16 token feature 生成不同尺度
        self.p4_refine = ConvGNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.p5_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvGNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        self.p3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvGNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        self.p2_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvGNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )

        # gated detail injection
        self.gate_p2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1),
            nn.Sigmoid()
        )

        self.gate_p3 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1),
            nn.Sigmoid()
        )

        self.refine_p2 = ConvGNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.refine_p3 = ConvGNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, image, f_shallow, f_mid, f_deep):
        """
        image: B×3×H×W
        f_shallow/f_mid/f_deep: B×C×H/16×W/16
        return:
            p2: B×256×H/4×W/4
            p3: B×256×H/8×W/8
            p4: B×256×H/16×W/16
            p5: B×256×H/32×W/32
        """

        # 1. DINOv3 跨层融合，得到 H/16 foundation feature
        fused = self.cross_layer_fusion(f_shallow, f_mid, f_deep)

        # 2. 图像细节分支
        detail_p2, detail_p3 = self.detail_stem(image)

        # 3. 生成金字塔
        p4 = self.p4_refine(fused)
        p5 = self.p5_down(p4)

        p3 = self.p3_up(fused)
        p2 = self.p2_up(fused)

        # 4. 细节门控注入
        if p3.shape[-2:] != detail_p3.shape[-2:]:
            detail_p3 = F.interpolate(
                detail_p3,
                size=p3.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        if p2.shape[-2:] != detail_p2.shape[-2:]:
            detail_p2 = F.interpolate(
                detail_p2,
                size=p2.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        gate3 = self.gate_p3(torch.cat([p3, detail_p3], dim=1))
        p3 = self.refine_p3(p3 + gate3 * detail_p3)

        gate2 = self.gate_p2(torch.cat([p2, detail_p2], dim=1))
        p2 = self.refine_p2(p2 + gate2 * detail_p2)

        return p2, p3, p4, p5

class LDoffset(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=15):
        super().__init__()
        p = kernel_size // 2
        self.conv_h = nn.Conv2d(
            in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, p)
        )
        self.conv_v = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(p, 0)
        )

    def forward(self, x):
        return self.conv_h(x) + self.conv_v(x)

class DeformableStripConv(nn.Module):
    def __init__(self, in_ch, out_ch, dckernel_size=3, lkernel_size=15):
        super().__init__()
        self.offset_conv = LDoffset(
            in_ch=in_ch,
            out_ch=2 * dckernel_size * dckernel_size,
            kernel_size=lkernel_size
        )
        self.mask_conv = nn.Conv2d(
            in_ch,
            dckernel_size * dckernel_size,
            kernel_size=3,
            padding=1
        )
        self.conv = DeformConv2d(
            in_ch,
            out_ch,
            kernel_size=dckernel_size,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.relu(self.bn(self.conv(x, offset, mask=mask)))

class DirectionAwarePipelineAggregation(nn.Module):
    def __init__(self, channels, lkernel_size=15, dckernel_size=3):
        super().__init__()
        p = lkernel_size // 2

        self.branch_h = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=(1, lkernel_size),
                padding=(0, p),
                groups=channels
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.branch_v = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=(lkernel_size, 1),
                padding=(p, 0),
                groups=channels
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.branch_dcn = DeformableStripConv(
            channels,
            channels,
            dckernel_size=dckernel_size,
            lkernel_size=lkernel_size
        )

        self.branch_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), 4, kernel_size=1)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b_h = self.branch_h(x)
        b_v = self.branch_v(x)
        b_d = self.branch_dcn(x)
        b_l = self.branch_local(x)

        w = torch.softmax(self.gate(x), dim=1)

        out = (
            w[:, 0:1] * b_h +
            w[:, 1:2] * b_v +
            w[:, 2:3] * b_d +
            w[:, 3:4] * b_l
        )

        return self.fuse(out + x)

    class PipeDecoderBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch, lkernel_size=15, dckernel_size=3):
            super().__init__()

            self.upsample = nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=2,
                stride=2
            )

            self.reduce = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

            self.dapa = DirectionAwarePipelineAggregation(
                out_ch,
                lkernel_size=lkernel_size,
                dckernel_size=dckernel_size
            )

            self.out_conv = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, x, skip):
            x = self.upsample(x)

            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(
                    skip,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=True
                )

            x = torch.cat([x, skip], dim=1)
            x = self.reduce(x)
            x = self.dapa(x)
            x = self.out_conv(x)

            return x

class BoundarySkeletonGuidedRefinement(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(in_ch + 3, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, 1, kernel_size=1)
        )

    def forward(self, feat, mask_coarse, edge_pred, skeleton_pred):
        h, w = feat.shape[-2:]

        edge = F.interpolate(
            edge_pred,
            size=(h, w),
            mode="bilinear",
            align_corners=True
        )

        skeleton = F.interpolate(
            skeleton_pred,
            size=(h, w),
            mode="bilinear",
            align_corners=True
        )

        x = torch.cat(
            [
                feat,
                torch.sigmoid(mask_coarse),
                torch.sigmoid(edge),
                torch.sigmoid(skeleton)
            ],
            dim=1
        )

        residual = self.refine(x)

        return mask_coarse + residual

class PipeDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, lkernel_size=15, dckernel_size=3):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.dapa = DirectionAwarePipelineAggregation(
            out_ch,
            lkernel_size=lkernel_size,
            dckernel_size=dckernel_size
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(
                skip,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        x = self.dapa(x)
        x = self.out_conv(x)

        return x
class CDBNet(nn.Module):
    def __init__(self, dinov3_path, embed_dim=1024, lkernel_size=15, dckernel_size=3):
        super().__init__()

        self.backbone = torch.hub.load(
            r"./dinov3/",
            "dinov3_vitl16",
            source="local",
            weights=dinov3_path
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.cdfpa = CDPFA(1024, 256)

        # Multi-level foundation feature adapter
        self.neck_p5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(embed_dim, 256, kernel_size=1)
        )

        self.neck_p4 = nn.Conv2d(embed_dim, 256, kernel_size=1)

        self.neck_p3 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.neck_p2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Direction-aware decoder
        self.dec4 = PipeDecoderBlock(
            256, 256, 128,
            lkernel_size=lkernel_size,
            dckernel_size=dckernel_size
        )

        self.dec3 = PipeDecoderBlock(
            128, 256, 64,
            lkernel_size=lkernel_size,
            dckernel_size=dckernel_size
        )

        self.dec2 = PipeDecoderBlock(
            64, 256, 32,
            lkernel_size=lkernel_size,
            dckernel_size=dckernel_size
        )

        # Boundary head
        self.edge_head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # Skeleton head
        self.skeleton_head = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        # Coarse mask and refinement
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.mask_coarse_head = nn.Conv2d(16, 1, kernel_size=1)

        self.refine = BoundarySkeletonGuidedRefinement(16)

    def forward(self, x):
        with torch.no_grad():
            layers = self.backbone.get_intermediate_layers(
                x,
                n=[7, 15, 23],
                reshape=True
            )

        f_shallow, f_mid, f_deep = layers

        p2, p3, p4, p5 = self.cdfpa(x, f_shallow, f_mid, f_deep)

        d4 = self.dec4(p5, p4)
        d3 = self.dec3(d4, p3)
        d2 = self.dec2(d3, p2)

        edge_pred = self.edge_head(d2)
        skeleton_pred = self.skeleton_head(d2)

        f_high = self.final_up(d2)
        mask_coarse = self.mask_coarse_head(f_high)

        mask_final = self.refine(
            f_high,
            mask_coarse,
            edge_pred,
            skeleton_pred
        )

        return  mask_final, edge_pred, skeleton_pred #"coarse_mask": mask_coarse,
        #return mask_coarse, edge_pred, skeleton_pred
