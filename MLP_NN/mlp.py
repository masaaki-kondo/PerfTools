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

    def forward(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(inputs) != self.n_branches:
            raise ValueError(f"Expected {self.n_branches} inputs, got {len(inputs)}")

        branch_outs = [
            self.encoders[self.branch_to_encoder[i]](x) for i, x in enumerate(inputs)
        ]
        merged = torch.cat(branch_outs, dim=-1)
        hidden = self.shared(merged)

        result = {"regression": self.regression_head(hidden)}
        if self.breakdown_head is not None:
            result["breakdown"] = F.softmax(self.breakdown_head(hidden), dim=-1)
        return result
