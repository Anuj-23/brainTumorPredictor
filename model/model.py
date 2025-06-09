import torch
import torch.nn as nn
import torchvision.models as models

# === Patch Embedding Layer ===
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)
        self.transpose = lambda x: x.transpose(1, 2)

    def forward(self, x):
        x = self.proj(x)               # Shape: (B, embed_dim, H', W')
        x = self.flatten(x)           # Shape: (B, embed_dim, H'*W')
        x = self.transpose(x)         # Shape: (B, H'*W', embed_dim)
        return x

# === Transformer Encoder Layer ===
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-head self-attention with residual connection
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Feed-forward network with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

# === Self-Optimizing Hybrid Vision Transformer (SOHViT) ===
class SOHViT(nn.Module):
    def __init__(self, num_classes=4, cnn_out_dim=512, embed_dim=256, patch_size=2, depth=4, num_heads=8):
        super().__init__()
        # ResNet-18 Backbone (remove final pooling and FC)
        cnn = models.resnet18(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(cnn.children())[:-2])  # Output: (B, 512, H/32, W/32)

        # Patch embedding after CNN
        self.patch_embed = PatchEmbedding(in_channels=cnn_out_dim, embed_dim=embed_dim, patch_size=patch_size)

        # Positional Embeddings (assumes 7x7 patches after 224x224 input and patch_size=2)
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim))  # (1, num_patches, embed_dim)

        # Transformer Encoder
        self.transformer = nn.Sequential(
            *[TransformerEncoderLayer(embed_dim, num_heads) for _ in range(depth)]
        )

        # Normalization and final classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.cnn_backbone(x)                  # Shape: (B, 512, 7, 7) if input is 224x224
        x = self.patch_embed(x)                   # Shape: (B, 49, 256)
        x = x + self.pos_embed[:, :x.size(1), :]  # Add positional embeddings
        x = self.transformer(x)                   # Transformer encoder
        x = x.mean(dim=1)                         # Global average pooling
        x = self.norm(x)                          # LayerNorm
        x = self.head(x)                          # Classification head
        return x
