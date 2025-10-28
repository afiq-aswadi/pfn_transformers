"""Base configuration class for PFN models."""

from dataclasses import dataclass

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


@dataclass
class BasePFNConfig(HookedTransformerConfig):
    """Base configuration for all PFN model types.

    Contains common transformer architecture parameters shared across
    regression, classification, and unsupervised models.

    Attributes:
        input_dim: Dimension of input features.
        use_pos_emb: Whether to use position embeddings.
        normalization_type: Type of normalization (default: "LN").
    """

    input_dim: int = 16
    use_pos_emb: bool = True
    normalization_type: str = "LN"

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except AttributeError:
            pass

        if not self.use_pos_emb:
            pass
