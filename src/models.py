"""
  0. DeepSVDDWithNorm   — Ablation baseline: F.normalize + bias=True (original implementation)
  1. DeepSVDD           — Primary model: ResNet-18 encoder + projection head (fixed)
  2. DeepSVDDWithSE     — Enhanced primary: adds SE channel attention (pre-avgpool)
  3. FCDDWithSE         — Custom model: spatial hypersphere (FCDD) + SE attention
  4. ConvAutoencoder    — Baseline 1: from-scratch convolutional AE
  5. ResNet18Classifier — Baseline 2: supervised binary CNN (upper bound)
  6. PatchCoreExtractor — Baseline 3: frozen ResNet-18 feature extractor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm



class _NormalisedProjectionHead(nn.Module):
    #Projection head with bias=True — used in DeepSVDDWithNorm for ablation.

    def __init__(self, in_dim=512, hidden_dims=None, out_dim=64, use_bn=True, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))         
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))       
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepSVDDWithNorm(nn.Module):
    
    #1 F.normalize in forward():
    #2 bias=True in Linear layers:
    #3 use_bn=True default:
       

    def __init__(
        self,
        freeze_up_to: str = 'layer2',
        hidden_dims: list = None,
        out_dim: int = 64,
        use_bn: bool = True,          
        dropout: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]

        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )
        if freeze_up_to is not None:
            freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
            stop = freeze_layers.index(freeze_up_to) + 1 if freeze_up_to in freeze_layers else 0
            for name in freeze_layers[:stop]:
                for param in getattr(backbone, name).parameters():
                    param.requires_grad = False

        self.head = _NormalisedProjectionHead(512, hidden_dims, out_dim, use_bn, dropout)
        self.register_buffer('centre', torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x).flatten(1)
        z     = self.head(feats)
        z     = F.normalize(z, p=2, dim=1)  
        return z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward(x)
        return ((z - self.centre) ** 2).sum(dim=1)

    @torch.no_grad()
    def init_centre(self, loader, device, eps=0.1):
        from tqdm import tqdm
        self.eval()
        all_z = []
        for x, _ in tqdm(loader, desc='init_centre (buggy)'):
            all_z.append(self.forward(x.to(device)).cpu())
        c = torch.cat(all_z, dim=0).mean(dim=0)
        c[(c.abs() < eps) & (c >= 0)] =  eps
        c[(c.abs() < eps) & (c < 0)]  = -eps
        self.centre.copy_(c.to(device))
        print(f"  Centre norm: {c.norm().item():.4f}")



# 1.  Deep SVDD

class ProjectionHead(nn.Module):
    """
    MLP projection head that maps ResNet features into the SVDD hypersphere.

    All Linear layers use bias=False. With bias=True, the network can set
    weights→0 and bias→c, trivially solving the SVDD loss without learning
    any features (hypersphere collapse shortcut).
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dims: list = None,
        out_dim: int = 64,
        use_bn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h, bias=False))  # bias=False: prevents collapse shortcut
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim, bias=False))  # bias=False: see class docstring

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSVDD(nn.Module):
    """
    Deep SVDD: ResNet-18 encoder + ProjectionHead.

    The encoder maps 128×128×3 patches to a 512-dim feature vector.
    The projection head compresses this to `out_dim` dimensions.

    Anomaly score at inference: ||phi(x) - c||^2

    """

    def __init__(
        self,
        freeze_up_to: str = 'layer2',
        hidden_dims: list = None,
        out_dim: int = 64,
        use_bn: bool = False,   # False: BN encourages embedding collapse in SVDD
        dropout: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]

       
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        # Remove avgpool and fc, we do our own global average pool
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,   # output: (B, 512, 1, 1)
        )

        # Freeze layers up to freeze_up_to
        if freeze_up_to is not None:
            self._freeze_encoder(backbone, freeze_up_to)

        
        self.head = ProjectionHead(
            in_dim=512,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            use_bn=use_bn,
            dropout=dropout,
        )

        
        self.register_buffer('centre', torch.zeros(out_dim))

    def _freeze_encoder(self, backbone: nn.Module, freeze_up_to: str) -> None:
        freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
        stop = freeze_layers.index(freeze_up_to) + 1 if freeze_up_to in freeze_layers else 0
        for name in freeze_layers[:stop]:
            for param in getattr(backbone, name).parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x).flatten(1)   # (B, 512)
        z     = self.head(feats)             # (B, out_dim) — raw, unbounded
        return z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward(x)
        return ((z - self.centre) ** 2).sum(dim=1)

    @torch.no_grad()
    def init_centre(self, loader, device: torch.device,
                    eps: float = 0.1) -> None:
        from tqdm import tqdm
        self.eval()
        all_z = []
        for x, _ in tqdm(loader, desc='init_centre'):
            z = self.forward(x.to(device))
            all_z.append(z.cpu())
        c = torch.cat(all_z, dim=0).mean(dim=0)
        # Hypersphere collapse guard
        c[(c.abs() < eps) & (c >= 0)] =  eps
        c[(c.abs() < eps) & (c < 0)]  = -eps
        self.centre.copy_(c.to(device))
        print(f"  Centre initialised — mean norm: {c.norm().item():.4f}")



# 1b. Deep SVDD with SE Attention + L2 Normalisation


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention block (Hu et al., 2018).

    Adapted for road damage detection: re-weights the 512 ResNet-18 feature
    channels so the model attends to texture-relevant channels (cracks,
    surface irregularities) and suppresses uninformative ones (uniform
    asphalt, sky).
    """

    def __init__(self, channels: int = 512, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — spatial feature maps (pre-avgpool)
        # squeeze: global avg pool → (B, C), excite: fc → (B, C), broadcast back
        w = x.mean(dim=[2, 3])                          # (B, C) — squeeze
        w = self.fc(w)                                  # (B, C) — excite
        return x * w.unsqueeze(-1).unsqueeze(-1)        # (B, C, H, W) — scale


class DeepSVDDWithSE(nn.Module):

    def __init__(
        self,
        freeze_up_to: str = 'layer1',
        hidden_dims: list = None,
        out_dim: int = 64,
        dropout: float = 0.0,
        pretrained: bool = True,
        se_reduction: int = 16,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

       
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        # Stop before avgpool so SE operates on spatial feature maps (B, 512, 8, 8)
        # rather than the globally-pooled vector (B, 512) — preserves spatial structure
        # for channel attention before collapsing to a single vector.
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            # avgpool moved after SE — see forward()
        )
        self.avgpool = backbone.avgpool  # (B, 512, 8, 8) → (B, 512, 1, 1)

        if freeze_up_to is not None:
            self._freeze_encoder(backbone, freeze_up_to)

       
        self.se = SEBlock(channels=512, reduction=se_reduction)

       
        self.head = ProjectionHead(
            in_dim=512,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            use_bn=False,      
            dropout=dropout,
        )

        self.register_buffer('centre', torch.zeros(out_dim))

    def _freeze_encoder(self, backbone: nn.Module, freeze_up_to: str) -> None:
        freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
        stop = freeze_layers.index(freeze_up_to) + 1 if freeze_up_to in freeze_layers else 0
        for name in freeze_layers[:stop]:
            for param in getattr(backbone, name).parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)                 # (B, 512, 8, 8) — spatial maps
        feats = self.se(feats)                  # SE attention on spatial maps
        feats = self.avgpool(feats).flatten(1)  # (B, 512) — collapse after attention
        z     = self.head(feats)                # (B, out_dim) — raw, unbounded
        return z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward(x)
        return ((z - self.centre) ** 2).sum(dim=1)

    @torch.no_grad()
    def init_centre(self, loader, device: torch.device,
                    eps: float = 0.1) -> None:
        from tqdm import tqdm
        self.eval()
        all_z = []
        for x, _ in tqdm(loader, desc='init_centre'):
            z = self.forward(x.to(device))
            all_z.append(z.cpu())
        c = torch.cat(all_z, dim=0).mean(dim=0)
        c[(c.abs() < eps) & (c >= 0)] =  eps
        c[(c.abs() < eps) & (c < 0)]  = -eps
        self.centre.copy_(c.to(device))
        print(f"  Centre initialised — mean norm: {c.norm().item():.4f}")


# 1c. FCDD with SE Attention  (Custom Model — Spatial Hypersphere)

class SpatialSEBlock(nn.Module):

    def __init__(self, channels: int = 512, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        w = x.mean(dim=[2, 3])          # (B, C) — global avg pool to get channel stats
        w = self.fc(w)                   # (B, C) — channel attention weights
        return x * w.unsqueeze(-1).unsqueeze(-1)   # (B, C, H, W) — broadcast


class FCDDWithSE(nn.Module):
    """
    Fully Convolutional Data Description with SE channel attention.

    Implements the spatial hypersphere loss from:
        Liznerski et al., "Explainable Deep One-Class Classification", ICLR 2021.

    Key difference from Deep SVDD:
        - Deep SVDD:  encoder → avgpool → (B, 512) → FC head → scalar score
        - FCDD:       encoder (no avgpool) → (B, 512, H, W) → 1×1 conv → heatmap

    The hypersphere loss is applied at every spatial location, so the model
    can detect localised anomalies (e.g. cracks covering 3% of image area)
    that avgpool would dilute to noise.

    Custom addition — SpatialSEBlock:
        Applies SE channel attention to the spatial feature map before the
        1×1 convolution, focusing the network on crack-relevant channels
        while preserving spatial resolution.

    Architecture:
        ResNet-18 (no avgpool) → (B, 512, 8, 8)
                               → SpatialSEBlock → (B, 512, 8, 8)
                               → Conv1x1        → (B, 1, 8, 8)   [distance map]
                               → upsample       → (B, 1, 256, 256) [anomaly heatmap]

    Anomaly score (image-level): max or mean of the upsampled heatmap.
    """

    def __init__(
        self,
        freeze_up_to: str = 'layer1',
        pretrained: bool = True,
        se_reduction: int = 16,
    ):
        super().__init__()

       
        weights  = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        # Stop before avgpool to preserve spatial information
        # For 256×256 input: layer4 output is (B, 512, 8, 8)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            # NO avgpool here
        )

        if freeze_up_to is not None:
            self._freeze_encoder(backbone, freeze_up_to)

        
        self.se = SpatialSEBlock(channels=512, reduction=se_reduction)

        self.conv1x1 = nn.Conv2d(512, 1, kernel_size=1, bias=False)

        self.register_buffer('centre', torch.zeros(1))

    def _freeze_encoder(self, backbone: nn.Module, freeze_up_to: str) -> None:
        freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
        stop = freeze_layers.index(freeze_up_to) + 1 if freeze_up_to in freeze_layers else 0
        for name in freeze_layers[:stop]:
            for param in getattr(backbone, name).parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)          # (B, 512, 8, 8)
        feat = self.se(feat)            # (B, 512, 8, 8) — spatial SE attention
        dist = self.conv1x1(feat)       # (B, 1, 8, 8)  — distance map
        return dist

    def anomaly_map(self, x: torch.Tensor,
                    output_size: tuple = (256, 256)) -> torch.Tensor:
        
        dist = self.forward(x)          # (B, 1, 8, 8)
        return F.interpolate(dist, size=output_size,
                             mode='bilinear', align_corners=False)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        
        dist = self.forward(x)          # (B, 1, 8, 8)
        return dist.pow(2).mean(dim=[1, 2, 3])   # (B,) — mean squared distance

    @torch.no_grad()
    def init_centre(self, loader, device: torch.device,
                    eps: float = 0.1) -> None:
        
        print("  FCDD: centre fixed at 0 (spatial hypersphere centred at origin).")
        print(f"  conv1x1 weight norm: {self.conv1x1.weight.norm().item():.4f}")



# 2.  Convolutional Autoencoder

class ConvEncoder(nn.Module):
    """
    4-block convolutional encoder.

    Architecture (default):
        Input (3, 256, 256)
        → Conv(32) → [BN] → ReLU → Pool    → (32, 128, 128)
        → Conv(64) → [BN] → ReLU → Pool    → (64, 64, 64)
        → Conv(128)→ [BN] → ReLU → Pool    → (128, 32, 32)
        → Conv(256)→ [BN] → ReLU → Pool    → (256, 16, 16)
        → Flatten + Linear                  → (bottleneck_dim,)
    """

    def __init__(
        self,
        bottleneck_dim: int = 256,
        use_bn: bool = True,
        pool: str = 'max',        # 'max' or 'avg'
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        def _block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            pool_layer = nn.MaxPool2d(2) if pool == 'max' else nn.AvgPool2d(2)
            layers.append(pool_layer)
            return nn.Sequential(*layers)

        self.blocks = nn.Sequential(
            _block(3,   32),
            _block(32,  64),
            _block(64,  128),
            _block(128, 256),
        )
        # After 4 pools: spatial size = 256/16 = 16
        self.flatten   = nn.Flatten()
        self.fc        = nn.Linear(256 * 16 * 16, bottleneck_dim)
        self.dropout   = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act_bottleneck = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)          # (B, 256, 16, 16)
        x = self.flatten(x)         # (B, 256*16*16)
        x = self.act_bottleneck(self.fc(x))   # (B, bottleneck_dim)
        x = self.dropout(x)
        return x


class ConvDecoder(nn.Module):
    """
    Symmetric 4-block convolutional decoder.

    Architecture (default):
        Input (bottleneck_dim,)
        → Linear → Reshape              → (256, 16, 16)
        → ConvT(128) → [BN] → ReLU      → (128, 32, 32)
        → ConvT(64)  → [BN] → ReLU      → (64, 64, 64)
        → ConvT(32)  → [BN] → ReLU      → (32, 128, 128)
        → ConvT(3)   → Sigmoid           → (3, 256, 256)
    """

    def __init__(
        self,
        bottleneck_dim: int = 256,
        use_bn: bool = True,
    ):
        super().__init__()
        self.fc      = nn.Linear(bottleneck_dim, 256 * 16 * 16)
        self.reshape = lambda x: x.view(-1, 256, 16, 16)

        def _block(in_c, out_c, final=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)]
            if not final:
                if use_bn:
                    layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)

        self.blocks = nn.Sequential(
            _block(256, 128),
            _block(128, 64),
            _block(64,  32),
            _block(32,  3, final=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = self.reshape(x)    # (B, 256, 16, 16)
        x = self.blocks(x)     # (B, 3, 256, 256)
        return x


class ConvAutoencoder(nn.Module):

    def __init__(
        self,
        bottleneck_dim: int = 256,
        use_bn: bool = True,
        pool: str = 'max',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = ConvEncoder(bottleneck_dim, use_bn, pool, dropout)
        self.decoder = ConvDecoder(bottleneck_dim, use_bn)

    def forward(self, x: torch.Tensor):
       
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction='none').mean(dim=[1, 2, 3])



# 3.  ResNet-18 Binary Classifier


class ResNet18Classifier(nn.Module):

    def __init__(
        self,
        freeze_up_to: str = 'layer2',
        pretrained: bool = True,
    ):
        super().__init__()
        weights   = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone  = tvm.resnet18(weights=weights)

        # Freeze early layers
        if freeze_up_to is not None:
            freeze_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
            stop = freeze_names.index(freeze_up_to) + 1
            for name in freeze_names[:stop]:
                for param in getattr(backbone, name).parameters():
                    param.requires_grad = False

        # Replace final FC with binary head
        in_feats = backbone.fc.in_features   # 512
        backbone.fc = nn.Linear(in_feats, 1)
        self.model  = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# 4.  PatchCore Feature Extractor  (SOTA no gradient training)

class PatchCoreExtractor(nn.Module):
    """
    Frozen ResNet-18 feature extractor for PatchCore.

    Taps intermediate feature maps from selected layers via forward hooks,
    concatenates and spatially average-pools them to produce a single
    feature vector per patch.
    """

    def __init__(
        self,
        feature_layers: list = None,
        pretrained: bool = True,
    ):
        super().__init__()
        if feature_layers is None:
            feature_layers = ['layer2', 'layer3']

        weights  = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)

        # Freeze all weights
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone       = backbone
        self.feature_layers = feature_layers
        self._hooks         = []
        self._features      = {}

        # Register forward hooks
        for layer_name in feature_layers:
            layer = getattr(backbone, layer_name)
            handle = layer.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(handle)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = output   # (B, C, H, W)
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._features.clear()
        with torch.no_grad():
            _ = self.backbone(x)   # triggers hooks

        pooled = []
        for name in self.feature_layers:
            feat = self._features[name]               # (B, C, H, W)
            pooled.append(feat.mean(dim=[2, 3]))      # (B, C)
        return torch.cat(pooled, dim=1)               # (B, sum_C)

    @property
    def feature_dim(self) -> int:
        # layer2: 128ch, layer3: 256ch, layer4: 512ch, layer1: 64ch
        _dims = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        return sum(_dims[l] for l in self.feature_layers)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
