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
    def __init__(self, dim, h=14, w=14):
        super(GlobalFilter, self).__init__()
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
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        weight = torch.view_as_complex(self.complex_weight.to(x.device))
        x = x * weight
        
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        
        return x


class Block(nn.Module):
    """Transformer block with GlobalFilter, LayerScale, and Stochastic Depth."""
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8,
                init_values=1e-4):
        super(Block, self).__init__()
        
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Percep(in_features=dim, hidden_features=mlp_hidden_dim, 
                        act_layer=act_layer, drop=drop)
        
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x, spatial_size=None):
        x = x + self.drop_path(self.gamma_1 * self.filter(self.norm1(x), spatial_size))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchyEmbedding(nn.Module):
    """Image to Patch Embedding with overlapping patches option."""
    def __init__(self, img_size=224, patch_size=16, stride=None, 
                in_chans=1, embed_dim=256):
        super(PatchyEmbedding, self).__init__()
        
        stride = stride or patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        
        # Calculate output dimensions
        self.H = (img_size - patch_size) // stride + 1
        self.W = (img_size - patch_size) // stride + 1
        self.num_patches = self.H * self.W
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                            kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x, (H, W)


class PatchMerging(nn.Module):
    """Merge patches to reduce spatial resolution and increase channels."""
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim = dim
        out_dim = out_dim or dim * 2
        
        # 2x2 reduction: 4*dim -> out_dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
        
        x = x.view(B, H, W, C)
        
        # Extract 2x2 patches: (B, H/2, W/2, 4*C)
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # Concatenate along channel
        x = x.view(B, -1, 4 * C)  # Flatten spatial dims
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x, H // 2, W // 2


class PyramidStage(nn.Module):
    """A pyramid stage with multiple GFNet blocks."""
    def __init__(self, dim, depth, mlp_ratio=3., drop=0., drop_path=0.,
                norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-4):
        super(PyramidStage, self).__init__()
        
        # Build blocks for this stage
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, h=h, w=w, init_values=init_values
            )
            for i in range(depth)
        ])
        
    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, spatial_size=(H, W))
        return x


class PyramidGFNet(nn.Module):
    """
    Pyramid Global Filter Network with hierarchical multi-scale features.
    
    Architecture:
    - Stage 1: High resolution, low channels (e.g., 56x56, dim=64)
    - Stage 2: Medium resolution, medium channels (e.g., 28x28, dim=128)
    - Stage 3: Medium-low resolution, more channels (e.g., 14x14, dim=256)
    - Stage 4: Low resolution, high channels (e.g., 7x7, dim=512)
    
    Args:
        img_size: Input image size
        patch_size: Initial patch size (stage 1)
        in_chans: Number of input channels
        num_classes: Number of classification classes
        embed_dims: Embedding dimensions for each stage [64, 128, 256, 512]
        depths: Number of blocks in each stage [3, 4, 6, 3]
        mlp_ratios: MLP expansion ratios for each stage [4, 4, 4, 4]
        drop_rate: Dropout rate
        drop_path_rate: Stochastic depth rate
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=1, num_classes=2,
                embed_dims=[64, 128, 256, 512], depths=[3, 4, 6, 3],
                mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0.1,
                norm_layer=None, init_values=1e-4):
        super(PyramidGFNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_stages = len(depths)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Stage 1: Patch embedding
        self.patch_embed1 = PatchyEmbedding(
            img_size=img_size, patch_size=patch_size, stride=patch_size,
            in_chans=in_chans, embed_dim=embed_dims[0]
        )
        
        # Positional embedding for stage 1
        self.pos_embed1 = nn.Parameter(
            torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0])
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Build pyramid stages
        cur = 0
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        
        for i in range(self.num_stages):
            # Calculate spatial dimensions for this stage
            h = img_size // (patch_size * (2 ** i))
            w = h // 2 + 1  # For rfft2
            
            # Build stage
            stage = PyramidStage(
                dim=embed_dims[i],
                depth=depths[i],
                mlp_ratio=mlp_ratios[i],
                drop=drop_rate,
                drop_path=dpr[cur:cur + depths[i]],
                norm_layer=norm_layer,
                h=h,
                w=w,
                init_values=init_values
            )
            self.stages.append(stage)
            
            # Patch merging (except for last stage)
            if i < self.num_stages - 1:
                merge = PatchMerging(
                    dim=embed_dims[i],
                    out_dim=embed_dims[i + 1],
                    norm_layer=norm_layer
                )
                self.merges.append(merge)
            
            cur += depths[i]
        
        # Final norm
        self.norm = norm_layer(embed_dims[-1])
        
        # Classification head
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
        # Initialize weights
        trunc_normal_(self.pos_embed1, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Extract hierarchical features from all stages."""
        # Stage 1: Initial patch embedding
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop(x)
        
        # Process through pyramid stages
        features = []
        for i in range(self.num_stages):
            x = self.stages[i](x, H, W)
            features.append(x)
            
            # Merge patches for next stage (if not last)
            if i < self.num_stages - 1:
                x, H, W = self.merges[i](x, H, W)
        
        # Final normalization
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return x, features

    def forward(self, x):
        """Forward pass through pyramid network."""
        x, _ = self.forward_features(x)
        x = self.head(x)
        return x


# Example usage and model variants
def pyramid_gfnet_tiny(img_size=224, num_classes=2, **kwargs):
    """Tiny variant: ~13M parameters"""
    model = PyramidGFNet(
        img_size=img_size,
        patch_size=4,
        embed_dims=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        mlp_ratios=[4, 4, 4, 4],
        num_classes=num_classes,
        **kwargs
    )
    return model


def pyramid_gfnet_small(img_size=224, num_classes=2, **kwargs):
    """Small variant: ~25M parameters"""
    model = PyramidGFNet(
        img_size=img_size,
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        depths=[3, 4, 6, 3],
        mlp_ratios=[4, 4, 4, 4],
        num_classes=num_classes,
        **kwargs
    )
    return model


def pyramid_gfnet_base(img_size=224, num_classes=2, **kwargs):
    """Base variant: ~44M parameters"""
    model = PyramidGFNet(
        img_size=img_size,
        patch_size=4,
        embed_dims=[96, 192, 384, 768],
        depths=[3, 4, 18, 3],
        mlp_ratios=[4, 4, 4, 4],
        num_classes=num_classes,
        **kwargs
    )
    return model