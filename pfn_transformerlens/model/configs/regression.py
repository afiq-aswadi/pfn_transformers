"""Configuration for supervised regression PFN models."""

from dataclasses import dataclass
from typing import Literal

import torch

from .base import BasePFNConfig


@dataclass
class SupervisedRegressionPFNConfig(BasePFNConfig):
    """Configuration for supervised regression PFN models.

    Supports two prediction modes:
    - "distribution": Predicts probability distribution over buckets
    - "point": Direct scalar regression (no bucketing)

    For distribution predictions, supports two bucketing strategies:
    - "uniform": Evenly-spaced buckets (requires y_min, y_max)
    - "riemann": Quantile-based buckets (requires riemann_borders)

    Attributes:
        mask_type: Attention mask type ("autoregressive-pfn" or "gpt2").
        prediction_type: Output type ("distribution" or "point").
        bucket_type: Bucketing strategy (only for distribution mode).
        bucket_support: Support type ("unbounded" or "bounded").
        y_min: Minimum value for uniform buckets.
        y_max: Maximum value for uniform buckets.
        riemann_borders: Precomputed borders for riemann buckets.
    """

    mask_type: Literal["autoregressive-pfn", "gpt2"] = "autoregressive-pfn"
    prediction_type: Literal["distribution", "point"] = "distribution"
    bucket_type: Literal["uniform", "riemann"] | None = None
    bucket_support: Literal["unbounded", "bounded"] = "unbounded"
    y_min: float | None = None
    y_max: float | None = None
    riemann_borders: torch.Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.prediction_type == "point":
            self.d_vocab_out = 1
            if (
                self.bucket_type is not None
                or self.y_min is not None
                or self.y_max is not None
            ):
                raise ValueError(
                    "Point prediction mode does not use bucketing. "
                    "Set prediction_type='distribution' to use buckets."
                )
        elif self.prediction_type == "distribution":
            if self.bucket_type is None:
                raise ValueError(
                    "Distribution prediction mode requires bucket_type to be specified. "
                    "Use 'uniform' or 'riemann'."
                )
            if self.bucket_type == "uniform":
                if self.y_min is None or self.y_max is None:
                    raise ValueError(
                        "Uniform bucket type requires both y_min and y_max to be specified."
                    )
                if self.y_min >= self.y_max:
                    raise ValueError(
                        f"y_min ({self.y_min}) must be < y_max ({self.y_max})"
                    )
            elif self.bucket_type == "riemann":
                if self.riemann_borders is None:
                    raise ValueError(
                        "Riemann bucket type requires riemann_borders to be specified. "
                        "Use estimate_riemann_borders() to compute them from data."
                    )
                if self.riemann_borders.ndim != 1:
                    raise ValueError(
                        f"riemann_borders must be 1D tensor, got shape {self.riemann_borders.shape}"
                    )
                expected_len = self.d_vocab + 1
                if len(self.riemann_borders) != expected_len:
                    raise ValueError(
                        f"riemann_borders must have length d_vocab+1 ({expected_len}), "
                        f"got {len(self.riemann_borders)}"
                    )
