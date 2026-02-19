import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class ECA(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * self.sigmoid(y).expand_as(x)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, learnable=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learnable else p
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class WildlifeHead(nn.Module):
    def __init__(self, in_features=1536, n_classes=8,
                 mlp_hidden=512, dropout=0.4):
        super().__init__()
        self.eca = ECA(kernel_size=5)
        self.gem = GeM(p=3, learnable=True)
        self.norm = nn.LayerNorm(in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_classes)
        )

    def forward(self, x):
        x = self.eca(x)
        x = self.gem(x).flatten(1)
        x = self.norm(x)
        return self.mlp(x)


def _create_base_model() -> nn.Module:
    """
    Builds the ConvNeXt backbone + WildlifeHead without loading any weights.
    Backbone is frozen by default — unfreeze explicitly in train.py when needed.
    """
    model = timm.create_model(
        "convnext_large_in22k",
        pretrained=False,  # never load ImageNet weights here — caller decides
        num_classes=0,
        global_pool=''
    )
    for param in model.parameters():
        param.requires_grad = False

    model.head = WildlifeHead(in_features=1536, n_classes=8,
                              mlp_hidden=512, dropout=0.4)
    return model


def build_model(cloud: bool) -> nn.Module:
    """
    Fresh model with ImageNet-22k pretrained backbone weights.
    Backbone frozen, head randomly initialised and trainable.
    Use this at the start of phase 1.
    """
    model = timm.create_model(
        "convnext_large_in22k",
        pretrained=True,  # load ImageNet-22k weights into backbone
        num_classes=0,
        global_pool=''
    )
    for param in model.parameters():
        param.requires_grad = False

    model.head = WildlifeHead(in_features=1536, n_classes=8,
                              mlp_hidden=512, dropout=0.4)

    model = model.to('cuda', memory_format=torch.channels_last)
    if cloud:
        model = torch.compile(model, mode='max-autotune')
    return model


def load_model(checkpoint_path: str, cloud: bool) -> nn.Module:
    """
    Rebuilds architecture (no pretrained download) and loads saved weights.
    Handles torch.compile prefix '_orig_mod.' automatically.
    Use this for phase 2, submission, and any resumption.
    """
    model = _create_base_model()
    model = model.to('cuda', memory_format=torch.channels_last)

    weights = torch.load(checkpoint_path, map_location='cuda')

    # torch.compile wraps all keys with '_orig_mod.' — strip it if present
    if any(k.startswith('_orig_mod.') for k in weights.keys()):
        weights = {k.replace('_orig_mod.', ''): v for k, v in weights.items()}

    model.load_state_dict(weights)
    print(f"  Loaded weights from {checkpoint_path}")

    # Only compile after loading — compiling changes the key structure
    if cloud:
        model = torch.compile(model, mode='max-autotune')

    return model


def get_trainable_params(model: nn.Module) -> list:
    """Returns only parameters with requires_grad=True for the optimizer."""
    return [p for p in model.parameters() if p.requires_grad]
