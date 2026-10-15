"""EfficientNet-B0 classifier plus a wrapper that also outputs class activation maps."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARCH


def build_model(num_classes: int, pretrained: bool = True, arch: str = ARCH) -> nn.Module:
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_rate=0.3)


def classifier_params(model: nn.Module):
    return list(model.get_classifier().parameters())


class ExportWrapper(nn.Module):
    """forward(x) -> (logits [B,K], cam [B,K,7,7]).

    The head is global-average-pool + linear, so the CAM for class k is the linear
    layer's weights applied at every spatial position (a 1x1 conv). The browser
    upsamples it into a heatmap without needing any gradients.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        fc = model.get_classifier()
        self.register_buffer("w", fc.weight.detach().clone()[:, :, None, None])
        self.register_buffer("b", fc.bias.detach().clone())

    def forward(self, x):
        feats = self.model.forward_features(x)
        logits = F.linear(feats.mean(dim=(2, 3)), self.w[:, :, 0, 0], self.b)
        cam = F.conv2d(feats, self.w)
        return logits, cam


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(len(ckpt["classes"]), pretrained=False, arch=ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt
