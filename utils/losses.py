"""Loss functions for metric learning and auxiliary supervision."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineBatchHardTripletLoss(nn.Module):
    """Triplet loss on L2-normalized embeddings with batch hard negative mining.

    The dataset still returns explicit anchor/positive/negative triplets, but the
    loss can replace the provided negative with the hardest different-identity
    candidate available inside the current batch. This is much stronger than
    using only random negatives.
    """

    def __init__(self, margin: float = 0.2, use_batch_hard: bool = True) -> None:
        super().__init__()
        self.margin = float(margin)
        self.use_batch_hard = bool(use_batch_hard)

    def _mine_batch_hard_negatives(
        self,
        anchor_embeddings: torch.Tensor,
        anchor_labels: torch.Tensor,
        provided_negative_embeddings: torch.Tensor,
        provided_negative_labels: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_embeddings = torch.cat(
            [provided_negative_embeddings, positive_embeddings, anchor_embeddings],
            dim=0,
        )
        candidate_labels = torch.cat(
            [provided_negative_labels, anchor_labels, anchor_labels],
            dim=0,
        )

        similarities = torch.matmul(anchor_embeddings, candidate_embeddings.t())
        same_identity_mask = candidate_labels.unsqueeze(0).eq(anchor_labels.unsqueeze(1))
        similarities = similarities.masked_fill(same_identity_mask, -1e4)
        hardest_indices = similarities.argmax(dim=1)
        hardest_similarities = similarities.gather(1, hardest_indices.unsqueeze(1)).squeeze(1)
        hardest_negatives = candidate_embeddings[hardest_indices]
        return hardest_negatives, hardest_similarities

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
        anchor_labels: torch.Tensor,
        negative_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

        if self.use_batch_hard:
            selected_negatives, hard_similarities = self._mine_batch_hard_negatives(
                anchor_embeddings=anchor_embeddings,
                anchor_labels=anchor_labels,
                provided_negative_embeddings=negative_embeddings,
                provided_negative_labels=negative_labels,
                positive_embeddings=positive_embeddings,
            )
        else:
            selected_negatives = negative_embeddings
            hard_similarities = F.cosine_similarity(anchor_embeddings, selected_negatives)

        positive_distances = 1.0 - F.cosine_similarity(anchor_embeddings, positive_embeddings)
        negative_distances = 1.0 - F.cosine_similarity(anchor_embeddings, selected_negatives)

        losses = F.relu(positive_distances - negative_distances + self.margin)
        loss = losses.mean()

        diagnostics = {
            'avg_positive_distance': float(positive_distances.detach().mean().item()),
            'avg_negative_distance': float(negative_distances.detach().mean().item()),
            'avg_hard_negative_similarity': float(hard_similarities.detach().mean().item()),
            'active_triplet_fraction': float((losses.detach() > 0).float().mean().item()),
        }
        return loss, diagnostics
