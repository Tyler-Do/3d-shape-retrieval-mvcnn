# models/mvcnn.py
import torch
import torch.nn as nn
from torchvision import models


class MVCNN(nn.Module):
    """
    Multi-View CNN built on top of ResNet-18.

    We share one 2D backbone across all views, then pool the per-view
    features into a single 512-d descriptor per object.

    Shapes (with V views):
        Input:         [B, V, C, H, W]
        After backbone: [B, V, D] with D=feature_dim
        After pooling:  [B, D]
        Logits:         [B, num_classes]
    """

    def __init__(self,
                 num_classes: int,
                 pretrained: bool = True,
                 feature_dim: int = 512,
                 pooling: str = "max"):
        super().__init__()

        # We reuse ResNet-18 and remove its final FC layer.
        if pretrained:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
        else:
            backbone = models.resnet18(weights=None)

        backbone.fc = nn.Identity()  # output is now a 512-d feature
        self.backbone = backbone
        self.feature_dim = feature_dim
        assert pooling in ["max", "mean"]
        self.pooling = pooling

        # Classifier on top of the pooled descriptor.
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """
        x: [B, V, C, H, W]
        returns:
            logits: [B, num_classes]
            pooled: [B, feature_dim]  (embedding used for retrieval)
        """
        B, V, C, H, W = x.shape

        # Merge batch and view dimension so that we can reuse a standard 2D CNN.
        x = x.view(B * V, C, H, W)

        # Pass all views through the shared backbone.
        feats = self.backbone(x)                     # [B*V, D]

        # Reshape back to [B, V, D] to pool across views.
        feats = feats.view(B, V, self.feature_dim)   # [B, V, D]

        if self.pooling == "mean":
            pooled = feats.mean(dim=1)               # [B, D]
        else:
            pooled, _ = feats.max(dim=1)             # [B, D]

        logits = self.fc(pooled)                     # [B, num_classes]
        return logits, pooled
