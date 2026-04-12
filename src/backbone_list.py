from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class _HookedBackbone(nn.Module):
    

    def __init__(self, backbone: nn.Module, layers: list[str]):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self._hooks = []
        self._features: dict[str, torch.Tensor] = {}
        self._patch_size: int | None = None

        for p in self.backbone.parameters():
            p.requires_grad = False

        for layer_name in layers:
            layer = getattr(self.backbone, layer_name)
            self._hooks.append(layer.register_forward_hook(self._make_hook(layer_name)))

    def _make_hook(self, name: str):
        def hook(_module, _inp, output):
            self._features[name] = output
            if name == self.layers[0]:
                self._patch_size = int(output.shape[-1])
        return hook

    @property
    def patch_size(self) -> int:
        if self._patch_size is None:
            raise RuntimeError("patch_size is not available before the first forward pass.")
        return self._patch_size

    @torch.no_grad()
    def get_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._features.clear()
        _ = self.backbone(x)
        return {name: self._features[name] for name in self.layers}

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class WideResNet50Backbone(_HookedBackbone):
    def __init__(self, device: str):
        weights = tvm.Wide_ResNet50_2_Weights.DEFAULT
        backbone = tvm.wide_resnet50_2(weights=weights).to(device).eval()
        super().__init__(backbone, layers=["layer2", "layer3"])


class ResNet18Backbone(_HookedBackbone):
    def __init__(self, device: str):
        weights = tvm.ResNet18_Weights.DEFAULT
        backbone = tvm.resnet18(weights=weights).to(device).eval()
        super().__init__(backbone, layers=["layer2", "layer3"])


backborn_list = {
    "wide_resnet50": WideResNet50Backbone,
    "resnet18": ResNet18Backbone,
}
