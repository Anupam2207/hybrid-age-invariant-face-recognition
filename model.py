"""Hybrid neural network for age-invariant face recognition."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    mobilenet_v2,
    resnet18,
    resnet50,
)


BackboneType = Literal['mobilenet_v2', 'resnet18', 'resnet50']
ModeType = Literal['hybrid', 'cnn_only', 'geom_only']
FusionType = Literal['concat', 'attention']



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



def _safe_load_resnet50(pretrained: bool):
    if not pretrained:
        return resnet50(weights=None)
    try:
        return resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception as exc:  # pragma: no cover - depends on runtime cache/network.
        print(f'[warning] Could not load pretrained ResNet50 weights: {exc}. Falling back to random init.')
        return resnet50(weights=None)


class DeepBackbone(nn.Module):
    """Image branch based on a pretrained CNN backbone."""

    def __init__(
        self,
        backbone: BackboneType = 'resnet50',
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
        elif backbone == 'resnet50':
            net = _safe_load_resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(net.children())[:-2])
            out_channels = 2048
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(out_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(x)
        pooled = self.pool(feature_map).flatten(1)
        return pooled, feature_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.forward_features(x)
        return self.projection(pooled)


class GeometricBranch(nn.Module):
    """MLP branch for low-dimensional landmark geometry features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
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


class ConcatFusionHead(nn.Module):
    """Concatenation-based feature fusion followed by fully connected layers."""

    def __init__(
        self,
        deep_dim: int,
        geom_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(deep_dim + geom_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, deep_feat: torch.Tensor, geom_feat: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        fused = torch.cat([deep_feat, geom_feat], dim=1)
        return self.network(fused), None


class AttentionFusionHead(nn.Module):
    """Lightweight attention-based fusion over deep and geometric tokens."""

    def __init__(
        self,
        deep_dim: int,
        geom_dim: int,
        hidden_dim: int,
        output_dim: int,
        attention_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.deep_projection = nn.Linear(deep_dim, hidden_dim)
        self.geom_projection = nn.Linear(geom_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, deep_feat: torch.Tensor, geom_feat: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        tokens = torch.stack(
            [
                self.deep_projection(deep_feat),
                self.geom_projection(geom_feat),
            ],
            dim=1,
        )
        attended, attn_weights = self.attention(tokens, tokens, tokens, need_weights=True)
        attended = self.norm(tokens + attended)
        fused = attended.reshape(attended.size(0), -1)
        return self.output(fused), attn_weights


class HybridFaceRecognizer(nn.Module):
    """Fuse deep embeddings and geometric descriptors into a final embedding."""

    def __init__(
        self,
        geometry_input_dim: int,
        backbone: BackboneType = 'resnet50',
        mode: ModeType = 'hybrid',
        deep_embedding_dim: int = 256,
        geom_hidden_dim: int = 128,
        geom_embedding_dim: int = 128,
        fusion_hidden_dim: int = 256,
        final_embedding_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.2,
        fusion_type: FusionType = 'concat',
        attention_heads: int = 4,
        enable_identity_head: bool = False,
        num_identity_classes: int | None = None,
        enable_age_head: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.backbone = backbone
        self.fusion_type = fusion_type
        self.attention_heads = int(attention_heads)
        self.geometry_input_dim = int(geometry_input_dim)
        self.deep_embedding_dim = int(deep_embedding_dim)
        self.geom_hidden_dim = int(geom_hidden_dim)
        self.geom_embedding_dim = int(geom_embedding_dim)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.final_embedding_dim = int(final_embedding_dim)
        self.dropout = float(dropout)
        self.enable_identity_head = bool(enable_identity_head)
        self.enable_age_head = bool(enable_age_head)
        self.num_identity_classes = int(num_identity_classes) if num_identity_classes is not None else None

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
            if fusion_type == 'concat':
                self.fusion_head = ConcatFusionHead(
                    deep_dim=deep_embedding_dim,
                    geom_dim=geom_embedding_dim,
                    hidden_dim=fusion_hidden_dim,
                    output_dim=final_embedding_dim,
                    dropout=dropout,
                )
            elif fusion_type == 'attention':
                self.fusion_head = AttentionFusionHead(
                    deep_dim=deep_embedding_dim,
                    geom_dim=geom_embedding_dim,
                    hidden_dim=fusion_hidden_dim,
                    output_dim=final_embedding_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                )
            else:
                raise ValueError(f'Unsupported fusion_type: {fusion_type}')
        elif mode == 'cnn_only':
            self.fusion_head = nn.Sequential(
                nn.Linear(deep_embedding_dim, fusion_hidden_dim),
                nn.BatchNorm1d(fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, final_embedding_dim),
            )
        elif mode == 'geom_only':
            self.fusion_head = nn.Sequential(
                nn.Linear(geom_embedding_dim, fusion_hidden_dim),
                nn.BatchNorm1d(fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, final_embedding_dim),
            )
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        if self.enable_identity_head and self.num_identity_classes is None:
            raise ValueError('num_identity_classes must be set when enable_identity_head=True.')

        self.identity_classifier = (
            nn.Linear(final_embedding_dim, self.num_identity_classes)
            if self.enable_identity_head and self.num_identity_classes is not None
            else None
        )
        self.age_regressor = (
            nn.Sequential(
                nn.Linear(final_embedding_dim, max(32, final_embedding_dim // 2)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(max(32, final_embedding_dim // 2), 1),
            )
            if self.enable_age_head
            else None
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.deep_branch(image)

    def encode_geometry(self, geometry: torch.Tensor) -> torch.Tensor:
        return self.geom_branch(geometry)

    def extract_deep_features(self, image: torch.Tensor) -> torch.Tensor:
        """Step 5: extract deep features from the backbone branch."""

        return F.normalize(self.encode_image(image), p=2, dim=1)

    def _fuse(self, deep_feat: torch.Tensor, geom_feat: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mode == 'hybrid':
            return self.fusion_head(deep_feat, geom_feat)
        if self.mode == 'cnn_only':
            return self.fusion_head(deep_feat), None
        return self.fusion_head(geom_feat), None

    def forward_features(self, image: torch.Tensor, geometry: torch.Tensor) -> dict:
        if self.mode == 'hybrid':
            deep_feat = self.encode_image(image)
            geom_feat = self.encode_geometry(geometry)
        elif self.mode == 'cnn_only':
            deep_feat = self.encode_image(image)
            geom_feat = None
        else:
            deep_feat = None
            geom_feat = self.encode_geometry(geometry)

        if deep_feat is None:
            fused_raw, attention = self._fuse(torch.empty(0, device=geometry.device), geom_feat)
        elif geom_feat is None:
            fused_raw, attention = self._fuse(deep_feat, torch.empty(0, device=image.device))
        else:
            fused_raw, attention = self._fuse(deep_feat, geom_feat)

        embedding = F.normalize(fused_raw, p=2, dim=1)
        outputs = {
            'embedding': embedding,
            'attention_weights': attention,
        }
        if deep_feat is not None:
            outputs['deep_features'] = deep_feat
        if geom_feat is not None:
            outputs['geometry_features'] = geom_feat
        if self.identity_classifier is not None:
            outputs['identity_logits'] = self.identity_classifier(embedding)
        if self.age_regressor is not None:
            outputs['age_prediction'] = self.age_regressor(embedding).squeeze(1)
        return outputs

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return self.forward_features(image, geometry)['embedding']

    def forward_with_aux(self, image: torch.Tensor, geometry: torch.Tensor) -> dict:
        return self.forward_features(image, geometry)

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
            'fusion_type': self.fusion_type,
            'attention_heads': self.attention_heads,
            'embedding_dim': self.final_embedding_dim,
            'deep_embedding_dim': self.deep_embedding_dim,
            'geom_hidden_dim': self.geom_hidden_dim,
            'geom_embedding_dim': self.geom_embedding_dim,
            'fusion_hidden_dim': self.fusion_hidden_dim,
            'dropout': self.dropout,
            'enable_identity_head': self.enable_identity_head,
            'enable_age_head': self.enable_age_head,
            'num_identity_classes': self.num_identity_classes,
        }
