import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import RoIAlign


class ConvBlock(nn.Module):
    """docstring"""

    def __init__(self, in_filters, out_filters):
        """docstring"""
        super().__init__()
        self.conv = nn.Conv2d(
            in_filters, out_filters, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        """docstring"""
        return self.conv(x)


class FPNBlock(nn.Module):
    """docstring"""

    def __init__(self, resnet_filters, out_filters=256):
        """docstring"""
        super().__init__()
        self.fpn5 = FPNBlock(resnet_filters[3], out_filters)
        self.fpn4 = FPNBlock(resnet_filters[2] + out_filters, out_filters)
        self.fpn3 = FPNBlock(resnet_filters[1] + out_filters, out_filters)
        self.fpn2 = FPNBlock(resnet_filters[0] + out_filters, out_filters)
        self.fpn1 = FPNBlock(resnet_filters[0] + out_filters, out_filters)

    def forward(self, res_feat):
        """docstring"""
        res2, res3, res4 = res_feat

        fpn5 = self.fpn5(res4)

        fpn5_unsampled = nn.functional.interpolate(fpn5, scale_factor=2, mode="nearest")
        fpn4 = self.fpn4(torch.cat([res3, fpn5_unsampled], dim=1))

        fpn4_unsampled = nn.functional.interpolate(fpn5, scale_factor=2, mode="nearest")
        fpn3 = self.fpn3(torch.cat([res2, fpn4_unsampled], dim=1))

        fpn3_unsampled = nn.functional.interpolate(fpn5, scale_factor=2, mode="nearest")
        fpn2 = self.fpn2(torch.cat([res2, fpn3_unsampled], dim=1))

        fpn2_unsampled = nn.functional.interpolate(fpn5, scale_factor=2, mode="nearest")
        fpn1 = self.fpn1(torch.cat([res2, fpn2_unsampled], dim=1))

        return [fpn1, fpn2, fpn3, fpn4, fpn5]


class ResNetBackbone(nn.Module):
    """docstring"""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        """docstring"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)

        return [res2, res3, res4]


class RPN(nn.Module):
    """docstring"""

    def __init__(self):
        """docstring"""
        super().__init__()
        self.conv = nn.Conv2d(256, 256, 3, padding=1)
        self.cls_logits = nn.Conv2d(256, 9 * 2, 1)
        self.bbox_pred = nn.Conv2d(256, 9 * 4, 1)

    def forward(self, x):
        """docstring"""
        logits = self.cls_logits(x)
        bbox = self.bbox_pred(x)
        return logits, bbox


class DetectionNetwork(nn.Module):
    """docstring"""

    def __init__(self):
        """docstring"""
        super().__init__()
        self.backbone = ResNetBackbone()
        self.fpn = FPNBlock([256, 256, 1064])
        self.rpn = RPN()
        self.roi_align = RoIAlign(output_size=(7, 7), sampling_ratio=2)

    def forward(self, x, rois=None):
        """docstring"""
        res_features = self.backbone(x)
        fpn_features = self.fpn(res_features)
        rpn_out = [self.rpn(f) for f in fpn_features]

        if rois is not None:
            rois_features = self.roi_align(fpn_features[0], rois)
            return rpn_out, rois_features
        return rpn_out
