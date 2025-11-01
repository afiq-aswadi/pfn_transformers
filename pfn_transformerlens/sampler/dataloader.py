"""Utilities for constructing PFN dataloaders and sampling prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

from .data_generator import DataGenerator
from .sampler import Sampler

if TYPE_CHECKING:
    from train import TrainingConfig


def build_dataloader(
    data_generator: DataGenerator,
    training_config: TrainingConfig,
) -> DataLoader:
    """Construct a DataLoader over an infinite sampler.

    Args:
        data_generator: DataGenerator instance (supervised or unsupervised).
            Use ProbabilisticGenerator to wrap prior/likelihood if needed.
        training_config: Training configuration with batch_size and seq_len.

    Returns:
        DataLoader that yields (x, y) tuples where x may be None for unsupervised models.
    """
    sampler = Sampler(
        seq_len=training_config.seq_len,
        data_generator=data_generator,
        internal_batch_size=training_config.batch_size,
    )

    # Inline collate function to handle None x values for unsupervised learning
    def collate_fn(batch):
        xs, ys = zip(*batch)
        if xs[0] is None:
            return None, torch.stack(ys)
        else:
            return torch.stack(xs), torch.stack(ys)

    return DataLoader(
        sampler,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


def sample_batch(
    data_generator: DataGenerator,
    batch_size: int,
    seq_len: int,
) -> tuple[Float[torch.Tensor, "batch seq input_dim"] | None, Float[torch.Tensor, "batch seq"]]:
    """Sample a batch of sequences from a data generator.

    Args:
        data_generator: DataGenerator instance (supervised or unsupervised).
        batch_size: Number of sequences to generate.
        seq_len: Length of each sequence.

    Returns:
        Tuple of (x, y) where:
            x: Input features of shape (batch_size, seq_len, input_dim), or None for unsupervised
            y: Target values of shape (batch_size, seq_len)
    """
    xs = []
    ys = []

    for _ in range(batch_size):
        result = data_generator.generate(seq_len)
        if isinstance(result, tuple):
            x, y = result
            xs.append(x)
            ys.append(y)
        else:
            # unsupervised generator returns only y
            ys.append(result)

    if xs:
        # supervised: stack x and y
        return torch.stack(xs), torch.stack(ys)
    else:
        # unsupervised: return None, y
        return None, torch.stack(ys)


__all__ = ["build_dataloader", "sample_batch"]
