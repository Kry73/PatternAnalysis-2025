# Source code of the models

import torch
import torch.nn as nn
import torch.fft
from functools import partial
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath
import math


class Percep(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.0):
        super(Percep, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(nn.Module):
    """Global filter in frequency domain using FFT."""
    def __init__(self, dim, h=14, w=8):
        super(GlobalFilter, self).__init__()
        # Initialize with smaller values for better stability
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        
        x = x.view(B, a, b, C).to(torch.float32)
        
        # FFT with conjugate symmetry (rfft2)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        # Apply learnable filter in frequency domain
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        
        # Inverse FFT
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        
        return x


class Block(nn.Module):
    """Transformer block with GlobalFilter, LayerScale, and Stochastic Depth."""
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8,
                 init_values=1e-4):  # LayerScale initialization
        super(Block, self).__init__()
        
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Percep(in_features=dim, hidden_features=mlp_hidden_dim, 
                          act_layer=act_layer, drop=drop)
        
        # LayerScale: learnable scaling parameters
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # Apply LayerScale to each branch
        x = x + self.drop_path(self.gamma_1 * self.filter(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchyEmbedding(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super(PatchyEmbedding, self).__init__()
        
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class GFNet(nn.Module):
    """
    Improved Global Filter Network with LayerScale and Progressive Stochastic Depth.
    
    Improvements based on:
    - LayerScale (Touvron et al., 2021): Better convergence for deeper models
    - Progressive Stochastic Depth (Huang et al., 2016): Linear drop rate scaling
    - Medical imaging best practices
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=2, 
                 embed_dim=768, depth=12, mlp_ratio=4., drop_rate=0., 
                 drop_path_rate=0.1, norm_layer=None, init_values=1e-4):
        super(GFNet, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim  # For feature extraction
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # Patch embedding
        self.patch_embed = PatchyEmbedding(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Filter dimensions
        h = img_size // patch_size
        w = h // 2 + 1
        
        # IMPROVED: Progressive stochastic depth rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Stack Transformer blocks with LayerScale
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w,
                init_values=init_values  # LayerScale initialization
            )
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Extract features through patch embedding and Transformer blocks."""
        # Patch embedding + positional encoding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Process through blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Normalize
        x = self.norm(x)
        
        # Global average pooling (already optimal)
        x = x.mean(dim=1)
        
        return x

    def forward(self, x):
        """Forward pass through feature extractor and classification head."""
        x = self.forward_features(x)
        x = self.head(x)
        return x