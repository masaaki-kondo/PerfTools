import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBranchMLP(nn.Module):
    """
    Multi-branch MLP for performance prediction.

    Separate input branches for different feature types are encoded independently,
    then merged into shared layers. Optionally supports:
    - Weight-sharing across branches (e.g., source and target GPU specs share encoder)
    - Two output heads: a regression head and a softmax breakdown head

    Args:
        branch_dims: List of input dimensions, one per branch.
        shared_branch_indices: Groups of branch indices that share weights.
            e.g. [[2, 3]] makes branches 2 and 3 use the same encoder.
        branch_hidden: Hidden size within each branch.
        shared_hidden: Hidden size for the shared layers after merging.
        n_shared_layers: Number of shared hidden layers.
        regression_outputs: Number of regression outputs.
        breakdown_outputs: Number of softmax breakdown outputs (set 0 to disable).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        branch_dims: list[int],
        shared_branch_indices: list[list[int]] | None = None,
        branch_hidden: int = 64,
        shared_hidden: int = 128,
        n_shared_layers: int = 2,
        regression_outputs: int = 5,
        breakdown_outputs: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_branches = len(branch_dims)
        self.shared_groups = shared_branch_indices or []

        branch_to_encoder: dict[int, int] = {}
        encoders: list[nn.Sequential] = []
        assigned: set[int] = set()

        for group in self.shared_groups:
            dims_in_group = {branch_dims[i] for i in group}
            if len(dims_in_group) != 1:
                raise ValueError(f"Shared branches must have same input dim: {group}")
            encoder_idx = len(encoders)
            encoders.append(self._make_branch(branch_dims[group[0]], branch_hidden, dropout))
            for i in group:
                branch_to_encoder[i] = encoder_idx
                assigned.add(i)

        for i in range(self.n_branches):
            if i not in assigned:
                branch_to_encoder[i] = len(encoders)
                encoders.append(self._make_branch(branch_dims[i], branch_hidden, dropout))

        self.encoders = nn.ModuleList(encoders)
        self.branch_to_encoder = branch_to_encoder

        merged_dim = branch_hidden * self.n_branches

        shared_layers: list[nn.Module] = []
        in_dim = merged_dim
        for _ in range(n_shared_layers):
            shared_layers.append(nn.Linear(in_dim, shared_hidden))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            in_dim = shared_hidden
        self.shared = nn.Sequential(*shared_layers)

        self.regression_head = nn.Linear(shared_hidden, regression_outputs)
        self.breakdown_head = (
            nn.Linear(shared_hidden, breakdown_outputs) if breakdown_outputs > 0 else None
        )

    @staticmethod
    def _make_branch(in_dim: int, hidden: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    # SPEC_DROPOUT (2026-10-03): feature-level dropout on the two spec branches (2 = source_specs,
    # 3 = target_specs) during training only, so no single spec value can act as a GPU identity
    # tag (the host-link number became "this is GB200" once GH200 was held out). Env-driven;
    # 0 = off. This is the class the TRAINERS use (v15_core carries a copy for inference).
    SPEC_DROPOUT = float(os.environ.get("SPEC_DROPOUT", "0") or 0)
    SPEC_MASK_P = float(os.environ.get("SPEC_MASK_P", "0") or 0)
    SPEC_MASK_COLS = [int(c) for c in os.environ.get("SPEC_MASK_COLS", "").split(",") if c.strip()]

    def forward(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(inputs) != self.n_branches:
            raise ValueError(f"Expected {self.n_branches} inputs, got {len(inputs)}")
        if self.training and self.SPEC_DROPOUT > 0:
            inputs = tuple(F.dropout(x, p=self.SPEC_DROPOUT, training=True) if i in (2, 3) else x
                           for i, x in enumerate(inputs))
        # SPEC_MASK_COLS / SPEC_MASK_P (2026-10-04): TARGETED version -- only the listed spec columns
        # (e.g. "-1" = the host-link bandwidth, the last spec feature) are set to 0 (= the training
        # mean after standardisation) per sample with probability P, with NO rescaling of kept
        # values, so the feature stays usable but cannot serve as a GPU identity tag.
        if self.training and self.SPEC_MASK_P > 0 and self.SPEC_MASK_COLS:
            def _mask(x):
                x = x.clone(); hit = torch.rand(x.shape[0], device=x.device) < self.SPEC_MASK_P
                for c in self.SPEC_MASK_COLS: x[hit, c] = 0.0
                return x
            inputs = tuple(_mask(x) if i in (2, 3) else x for i, x in enumerate(inputs))

        branch_outs = [
            self.encoders[self.branch_to_encoder[i]](x) for i, x in enumerate(inputs)
        ]
        merged = torch.cat(branch_outs, dim=-1)
        hidden = self.shared(merged)

        result = {"regression": self.regression_head(hidden)}
        if self.breakdown_head is not None:
            result["breakdown"] = F.softmax(self.breakdown_head(hidden), dim=-1)
        return result
