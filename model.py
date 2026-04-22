"""Hybrid neural network for age-invariant face recognition."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)


BackboneType = Literal['mobilenet_v2', 'resnet18']
ModeType = Literal['hybrid', 'cnn_only', 'geom_only']


def _safe_load_mobilenet(pretrained: bool):
    if not pretrained:
        return mobilenet_v2(weights=None)
    try:
        return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    except Exception as exc:  # pragma: no cover - depends on runtime cache/network.
        print(f'[warning] Could not load pretrained MobileNetV2 weights: {exc}. Falling back to random init.')
        return mobilenet_v2(weights=None)



def _safe_load_resnet18(pretrained: bool):
    if not pretrained:
        return resnet18(weights=None)
    try:
        return resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:  # pragma: no cover - depends on runtime cache/network.
        print(f'[warning] Could not load pretrained ResNet18 weights: {exc}. Falling back to random init.')
        return resnet18(weights=None)


class DeepBackbone(nn.Module):
    """Image branch based on a lightweight pretrained CNN backbone."""

    def __init__(
        self,
        backbone: BackboneType = 'mobilenet_v2',
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

        if backbone == 'mobilenet_v2':
            net = _safe_load_mobilenet(pretrained=pretrained)
            self.features = net.features
            out_channels = 1280
        elif backbone == 'resnet18':
            net = _safe_load_resnet18(pretrained=pretrained)
            self.features = nn.Sequential(*list(net.children())[:-2])
            out_channels = 512
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(out_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        embedding = self.projection(pooled)
        return embedding


class GeometricBranch(nn.Module):
    """MLP branch for low-dimensional landmark geometry features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HybridFaceRecognizer(nn.Module):
    """Fuse deep embeddings and geometric descriptors into a final embedding."""

    def __init__(
        self,
        geometry_input_dim: int,
        backbone: BackboneType = 'mobilenet_v2',
        mode: ModeType = 'hybrid',
        deep_embedding_dim: int = 256,
        geom_hidden_dim: int = 64,
        geom_embedding_dim: int = 64,
        fusion_hidden_dim: int = 256,
        final_embedding_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.backbone = backbone
        self.geometry_input_dim = geometry_input_dim
        self.deep_embedding_dim = deep_embedding_dim
        self.geom_hidden_dim = geom_hidden_dim
        self.geom_embedding_dim = geom_embedding_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.final_embedding_dim = final_embedding_dim
        self.dropout = dropout

        self.deep_branch = DeepBackbone(
            backbone=backbone,
            embedding_dim=deep_embedding_dim,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.geom_branch = GeometricBranch(
            input_dim=geometry_input_dim,
            hidden_dim=geom_hidden_dim,
            output_dim=geom_embedding_dim,
            dropout=dropout / 2,
        )

        if mode == 'hybrid':
            fusion_input_dim = deep_embedding_dim + geom_embedding_dim
        elif mode == 'cnn_only':
            fusion_input_dim = deep_embedding_dim
        elif mode == 'geom_only':
            fusion_input_dim = geom_embedding_dim
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, final_embedding_dim),
        )

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        if self.mode == 'hybrid':
            deep_feat = self.deep_branch(image)
            geom_feat = self.geom_branch(geometry)
            fused = torch.cat([deep_feat, geom_feat], dim=1)
        elif self.mode == 'cnn_only':
            fused = self.deep_branch(image)
        else:
            fused = self.geom_branch(geometry)

        embedding = self.fusion_head(fused)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward_pair(
        self,
        image1: torch.Tensor,
        geom1: torch.Tensor,
        image2: torch.Tensor,
        geom2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute embeddings and cosine similarity for a pair of faces."""

        emb1 = self.forward(image1, geom1)
        emb2 = self.forward(image2, geom2)
        score = F.cosine_similarity(emb1, emb2)
        return emb1, emb2, score

    def freeze_backbone(self) -> None:
        for parameter in self.deep_branch.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.deep_branch.features.parameters():
            parameter.requires_grad = True

    def export_config(self) -> dict:
        return {
            'geometry_input_dim': self.geometry_input_dim,
            'backbone': self.backbone,
            'mode': self.mode,
            'embedding_dim': self.final_embedding_dim,
            'deep_embedding_dim': self.deep_embedding_dim,
            'geom_hidden_dim': self.geom_hidden_dim,
            'geom_embedding_dim': self.geom_embedding_dim,
            'fusion_hidden_dim': self.fusion_hidden_dim,
            'dropout': self.dropout,
        }
