import torch.nn as nn
import timm

class DualFeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model('convnext_tiny', pretrained=True, features_only=True, out_indices=[3])
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

    def forward(self, x):
        cnn_features = self.cnn(x)[0]
        vit_features = self.vit.forward_features(x)
        return cnn_features, vit_features
