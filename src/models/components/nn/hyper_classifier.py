from typing import Optional

import numpy as np
import torch

from src.utils.meru_utils.lorentz import pairwise_inner, pairwise_dist

class Hyper_NearestNeighboursClassifier(torch.nn.Module):
    """Nearest neighbours classifier.

    It computes the similarity between the query and the supports using the
    cosine similarity and then applies a softmax to obtain the logits.

    Args:
        tau (float): Temperature for the softmax. Defaults to 1.0.
    """

    def __init__(self, tau: float = 1.0, is_hyper: bool = True, use_softmax: bool = False) -> None:
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.tau = tau
        self.is_hyper = is_hyper
        self.use_softmax = use_softmax

    def forward(
        self, query: torch.Tensor, supports: torch.Tensor, mask: Optional[torch.Tensor] = None,
        curvature: torch.Tensor = 1.0
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query (torch.Tensor): Query tensor.
            supports (torch.Tensor): Supports tensor.
            mask (torch.Tensor, optional): Zero out the similarities for the masked supports.
                Defaults to None.
        """
        if not self.is_hyper:
            query = query / query.norm(dim=-1, keepdim=True) # images #! not needed
            supports = supports / supports.norm(dim=-1, keepdim=True) # words #! not needed

        if supports.dim() == 2:
            supports = supports.unsqueeze(0)

        Q, _ = query.shape
        N, C, _ = supports.shape

        supports = supports.mean(dim=0)
        # supports = supports / supports.norm(dim=-1, keepdim=True)
        if self.is_hyper:
            similarity = pairwise_inner(query, supports, curvature) # self.logit_scale.exp()pairwise_inner
        else:
            similarity = self.logit_scale.exp() * query @ supports.T
        similarity = similarity / self.tau if self.tau != 1.0 else similarity

        if self.use_softmax:
            if mask is not None:
                assert mask.shape[0] == query.shape[0] and mask.shape[1] == supports.shape[0]
                similarity = torch.masked_fill(similarity, mask == 0, float("-inf"))
            logits = similarity.softmax(dim=-1)
        
        else:
            # in MERU they do not apply softmax neither when using CLIPBaseline
            if mask is not None:
                assert mask.shape[0] == query.shape[0] and mask.shape[1] == supports.shape[0]
                similarity = torch.masked_fill(similarity, mask == 0, 0) #! no softmax will be performed, we only mask out the candidates for other images
            logits = similarity

        return logits