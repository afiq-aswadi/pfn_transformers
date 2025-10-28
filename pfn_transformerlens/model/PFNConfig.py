from dataclasses import dataclass
from typing import Optional

import torch
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


@dataclass
class PFNConfig(HookedTransformerConfig):
    """Configuration class for Prior-Fitted Networks (PFNs).

    Extends HookedTransformerConfig to include PFN-specific parameters for
    configuring the architecture and training behavior of Prior-Fitted Networks.

    Attributes:
        input_dim: Dimension of the input features. Default is 16.
        y_min: Minimum value for uniform bucket type. Only needed for uniform bucket type.
        y_max: Maximum value for uniform bucket type. Only needed for uniform bucket type.
        mask_type: Type of attention mask to use. Options: "gpt2", "autoregressive-pfn".
        bucket_type: Type of bucketing strategy. Options: "riemann", "uniform".
        use_pos_emb: Whether to use position embeddings in the model.
        normalization_type: Type of normalization to apply. Default is "LN" (Layer Norm).
        prediction_type: Type of prediction output. Options: "distribution", "point".
        bucket_support: Support type for buckets. Options: "unbounded", "bounded".
        riemann_borders: Precomputed borders for riemann buckets. Can be None for auto-computation.
    """

    input_dim: int = 16  # Default value to avoid dataclass ordering issues
    y_min: Optional[float] = None  # only needed for uniform bucket type
    y_max: Optional[float] = None  # only needed for uniform bucket type
    mask_type: str = "autoregressive-pfn"  # Options: "gpt2", "autoregressive-pfn"
    bucket_type: str = "uniform"  # Options: "riemann", "uniform"
    use_pos_emb: bool = True  # whether to use position embeddings
    # TODO: if pos_emb is false then how to bypass transformerlens n_ctx? easy hack: just post-hoc
    # set W_pos to torch.zeros
    normalization_type: str = "LN"
    prediction_type: str = "distribution"  # Options: "distribution", "point"
    bucket_support: str = "unbounded"  # Options: "unbounded", "bounded"
    riemann_borders: torch.Tensor | None = (
        None  # Precomputed borders for riemann buckets
    )

    def __post_init__(self) -> None:
        """Initialize the configuration after dataclass instantiation.

        Calls the parent class's __post_init__ method if available, and sets
        d_vocab_out to 1 when prediction_type is "point" to ensure proper
        output dimension for point predictions. Also sets bucket_type to None
        for point predictions since bucketing is not needed.
        """
        try:
            super().__post_init__()
        except AttributeError:
            pass

        if self.prediction_type == "point":
            self.d_vocab_out = 1
            self.bucket_type = None
