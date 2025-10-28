"""Configuration for unsupervised (next-token prediction) PFN models."""

from dataclasses import dataclass
from typing import Literal

import torch

from .base import BasePFNConfig


@dataclass
class UnsupervisedPFNConfig(BasePFNConfig):
    """Configuration for unsupervised next-token prediction models.

    This mode trains a standard GPT-2 style transformer without x/y interleaving
    or special PFN attention masks. Pure sequence modeling for approximating
    posterior predictive distributions p(x*|x_1:n).

    Supports both discrete and continuous sequences with point or distributional predictions.

    Attributes:
        d_vocab: Vocabulary size (discrete) or number of buckets (continuous distribution).
        input_type: Whether inputs are discrete tokens or continuous values.
        prediction_type: Whether to predict points or distributions.
        bucket_type: Bucketing strategy for continuous distribution predictions.
        mask_type: Must be "gpt2" (causal attention only).
        act_fn: Activation function (default: "gelu").

    Valid combinations:
        - discrete + point: rare, predicts single token index
        - discrete + distribution: standard language modeling (d_vocab = vocabulary size)
        - continuous + point: next-value regression (output shape: batch x seq x 1)
        - continuous + distribution: probabilistic continuous modeling (requires bucket_type)
    """

    d_vocab: int = 2
    input_type: Literal["discrete", "continuous"] = "discrete"
    prediction_type: Literal["point", "distribution"] = "distribution"
    bucket_type: Literal["uniform", "riemann"] | None = None
    bucket_support: Literal["unbounded", "bounded"] = "unbounded"
    y_min: float | None = None
    y_max: float | None = None
    riemann_borders: torch.Tensor | None = None
    mask_type: Literal["gpt2"] = "gpt2"
    act_fn: str = "gelu"

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.mask_type != "gpt2":
            raise ValueError(
                f"Unsupervised mode only supports mask_type='gpt2', got '{self.mask_type}'. "
                "Use SupervisedRegressionPFNConfig or ClassificationPFNConfig for "
                "PFN-style attention."
            )

        if self.d_vocab <= 0:
            raise ValueError(f"d_vocab must be positive, got {self.d_vocab}")

        # Validate bucket_type requirements
        if self.input_type == "continuous" and self.prediction_type == "distribution":
            if self.bucket_type is None:
                raise ValueError(
                    "continuous inputs with distribution predictions require bucket_type "
                    "(either 'uniform' or 'riemann')"
                )

            # Validate bucketing parameters (similar to SupervisedRegressionPFNConfig)
            if self.bucket_type == "uniform":
                if self.y_min is None or self.y_max is None:
                    raise ValueError(
                        "Uniform bucketing requires y_min and y_max. "
                        "Set bucket_type='riemann' if using riemann_borders instead."
                    )
                if self.y_min >= self.y_max:
                    raise ValueError(
                        f"y_min ({self.y_min}) must be < y_max ({self.y_max})"
                    )
            elif self.bucket_type == "riemann":
                if self.riemann_borders is None:
                    raise ValueError(
                        "Riemann bucketing requires riemann_borders. "
                        "Set bucket_type='uniform' if using y_min/y_max instead."
                    )
        elif self.bucket_type is not None:
            raise ValueError(
                f"bucket_type should only be set for continuous distribution predictions, "
                f"got input_type={self.input_type}, prediction_type={self.prediction_type}"
            )

        # Set output dimension based on prediction type
        if self.prediction_type == "point":
            self.d_vocab_out = 1
        else:
            self.d_vocab_out = self.d_vocab

        self.input_dim = 1
