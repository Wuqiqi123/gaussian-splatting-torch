"""
View-dependent colour network for Gaussian Splatting.

Two backends, same API:
  - HashColorNetwork   : tiny-cuda-nn  HashGrid + FullyFusedMLP  (fast, high quality)
  - FallbackColorNetwork: pure PyTorch  frequency encoding + MLP  (no extra deps)

Auto-selected at import time; override with USE_TCNN env var:
    USE_TCNN=0 python train.py   # force pure-PyTorch backend
"""

import os
import math
import torch
import torch.nn as nn

# ── backend selection ────────────────────────────────────────────────────────

_USE_TCNN = os.environ.get('USE_TCNN', '1') != '0'
try:
    if _USE_TCNN:
        import tinycudann as tcnn
        _HAS_TCNN = True
    else:
        _HAS_TCNN = False
except ImportError:
    _HAS_TCNN = False

if _HAS_TCNN:
    print("[color_network] backend: tiny-cuda-nn (HashGrid + FullyFusedMLP)")
else:
    print("[color_network] backend: pure PyTorch (frequency encoding + MLP)")


# ═══════════════════════════════════════════════════════════════════════════════
#  tiny-cuda-nn backend
# ═══════════════════════════════════════════════════════════════════════════════

class _TcnnColorNetwork(nn.Module):
    """
    Hash-grid colour network backed by tiny-cuda-nn.

    xyz  →  multi-resolution HashGrid  ─┐
                                         ├─ concat → FullyFusedMLP → RGB
    dir  →  SphericalHarmonics (deg 4) ─┘
    """

    def __init__(
        self,
        n_levels: int = 12,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 1024,
        mlp_hidden_dim: int = 64,
        mlp_n_layers: int = 2,
    ):
        super().__init__()
        per_level_scale = (finest_resolution / base_resolution) ** (1.0 / (n_levels - 1))

        self.pos_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.dir_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "SphericalHarmonics", "degree": 4},
        )

        n_pos_feat = n_levels * n_features_per_level
        n_dir_feat = 4 ** 2   # SH degree 4 → 16

        self.mlp = tcnn.Network(
            n_input_dims=n_pos_feat + n_dir_feat,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": mlp_hidden_dim,
                "n_hidden_layers": mlp_n_layers,
            },
        )

        self.register_buffer("scene_center", torch.zeros(3))
        self.register_buffer("scene_scale",  torch.ones(1))

    @torch.no_grad()
    def set_scene_bounds(self, xyz: torch.Tensor):
        center = xyz.mean(dim=0)
        scale  = (xyz - center).norm(dim=-1).max() * 1.1 + 1e-5
        self.scene_center.copy_(center)
        self.scene_scale.fill_(scale.item())

    def forward(self, xyz: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        xyz_norm = ((xyz - self.scene_center) / self.scene_scale * 0.5 + 0.5).clamp(0.0, 1.0)
        dir_unit = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
        feat = torch.cat([self.pos_encoder(xyz_norm), self.dir_encoder(dir_unit)], dim=-1)
        return self.mlp(feat).float()


# ═══════════════════════════════════════════════════════════════════════════════
#  pure-PyTorch fallback backend
# ═══════════════════════════════════════════════════════════════════════════════

class _FreqEncoding(nn.Module):
    """NeRF-style frequency (Fourier feature) positional encoding."""

    def __init__(self, n_freqs: int, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self):
        return (1 + 2 * self.n_freqs) * 3 if self.include_input else 2 * self.n_freqs * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3)
        x_freq = x[..., None] * self.freqs * math.pi   # (..., 3, n_freqs)
        enc = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)  # (..., 3, 2*n_freqs)
        enc = enc.flatten(-2)                                              # (..., 6*n_freqs)
        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)
        return enc


class _FallbackColorNetwork(nn.Module):
    """
    Pure-PyTorch colour network using frequency encoding + MLP.
    Drops in as a replacement when tiny-cuda-nn is unavailable.
    """

    def __init__(
        self,
        pos_n_freqs: int = 10,
        dir_n_freqs: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.pos_enc = _FreqEncoding(n_freqs=pos_n_freqs, include_input=True)
        self.dir_enc = _FreqEncoding(n_freqs=dir_n_freqs, include_input=True)

        in_dim = self.pos_enc.out_dim + self.dir_enc.out_dim
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, 3), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

        self.register_buffer("scene_center", torch.zeros(3))
        self.register_buffer("scene_scale",  torch.ones(1))

    @torch.no_grad()
    def set_scene_bounds(self, xyz: torch.Tensor):
        center = xyz.mean(dim=0)
        scale  = (xyz - center).norm(dim=-1).max() * 1.1 + 1e-5
        self.scene_center.copy_(center)
        self.scene_scale.fill_(scale.item())

    def forward(self, xyz: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        xyz_norm = (xyz - self.scene_center) / self.scene_scale     # approx [-1, 1]
        dir_unit = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
        feat = torch.cat([self.pos_enc(xyz_norm), self.dir_enc(dir_unit)], dim=-1)
        return self.mlp(feat)


# ═══════════════════════════════════════════════════════════════════════════════
#  Public name — same API regardless of backend
# ═══════════════════════════════════════════════════════════════════════════════

HashColorNetwork = _TcnnColorNetwork if _HAS_TCNN else _FallbackColorNetwork
