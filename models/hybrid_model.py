import torch
import torch.nn as nn
from .backbones import DualFeatureBackbone
from .attention import BidirectionalCrossAttention
from .fusion import CBAM

class ConvNeXtViTHybridModel(nn.Module):
    def __init__(self, num_classes, embed_dim=768, num_heads=8, dropout=0.4):
        super().__init__()
        self.backbone = DualFeatureBackbone()
        self.cnn_proj = nn.Linear(768, embed_dim)
        self.cross_attention = BidirectionalCrossAttention(dim=embed_dim, num_heads=num_heads)
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)

        self.cbam = CBAM(channels=embed_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        cnn_feat, vit_feat = self.backbone(x)
        
        H, W = cnn_feat.shape[2], cnn_feat.shape[3]
        cnn_tokens = self.cnn_proj(cnn_feat.flatten(2).transpose(1, 2))

        if vit_feat.ndim == 3 and vit_feat.shape[1] == 197:
            vit_feat = vit_feat[:, 1:, :] 

        attn_cnn, attn_vit = self.cross_attention(cnn_tokens, vit_feat)

        feat_cnn = attn_cnn.transpose(1, 2).reshape(B, -1, H, W)
        feat_vit = attn_vit.mean(dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        fused = (self.alpha * feat_cnn) + (self.beta * feat_vit)
        refined = self.cbam(fused)
        
        return self.classifier(self.pool(refined).flatten(1))