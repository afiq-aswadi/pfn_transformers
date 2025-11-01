from __future__ import annotations

import math
from typing import Literal, Optional
from warnings import warn

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from .configs.regression import SupervisedRegressionPFNConfig

BucketType = Literal["uniform", "riemann"]
BucketSupport = Literal["bounded", "unbounded"]

HALF_NORMAL_TARGET_CDF = 0.5
HALF_NORMAL_ICDF_AT_TARGET = float(
    torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
        torch.tensor(HALF_NORMAL_TARGET_CDF),
    ),
)
SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)
LOG_SQRT_TWO_OVER_PI = math.log(SQRT_TWO_OVER_PI)
LOG_HALF = math.log(0.5)


def estimate_riemann_borders(
    ys: Float[Tensor, "..."],
    *,
    num_buckets: int,
    widen_borders_factor: float = 1.0,
) -> Float[Tensor, " num_buckets+1"]:
    # TODO: check!
    # TODO: make this a method of.. idk.. the data generator? and we'd have to assert y is continuous and not discrete
    """
    Estimate quantile-based bucket borders for Riemann bucketization.
    As implemented in https://arxiv.org/abs/2112.10510
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive.")
    flat = ys.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("ys must contain at least one value.")
    clean = flat[~torch.isnan(flat)]
    if clean.numel() <= num_buckets:
        raise ValueError("Need more samples than buckets to estimate borders.")
    remainder = clean.numel() % num_buckets
    if remainder:
        clean = clean[:-remainder]
    if clean.numel() == 0:
        raise ValueError("Not enough samples after equalizing bucket sizes.")
    ys_sorted = clean.sort().values
    bucket_size = clean.numel() // num_buckets
    upper_candidates = ys_sorted[bucket_size::bucket_size]
    lower_candidates = ys_sorted[bucket_size - 1 :: bucket_size][:-1]
    interior = 0.5 * (lower_candidates + upper_candidates)
    dtype = clean.dtype
    device = clean.device
    borders = torch.cat(
        (
            ys_sorted[:1],
            interior,
            ys_sorted[-1:].to(dtype=dtype, device=device),
        ),
    )
    if not math.isclose(widen_borders_factor, 1.0):
        borders = borders * float(widen_borders_factor)
    unique_borders = torch.unique_consecutive(borders)
    if unique_borders.numel() != borders.numel():
        warn("Bucket borders were not unique; removed duplicates.")
    if unique_borders.numel() - 1 != num_buckets:
        raise ValueError(
            "Estimated borders do not define the requested number of buckets.",
        )
    return unique_borders


class Bucketizer(nn.Module):
    """Utility that maps continuous targets to discrete buckets (and back)."""

    borders: Tensor
    bucket_widths: Tensor

    def __init__(
        self,
        *,
        bucket_type: BucketType,
        bucket_support: BucketSupport,
        d_vocab: int,
        y_min: Optional[float],
        y_max: Optional[float],
        borders: Optional[Tensor] = None,
    ):
        super().__init__()
        if d_vocab < 2:
            raise ValueError("d_vocab must be >= 2 for bucketization.")

        self.bucket_type = bucket_type
        self.bucket_support = bucket_support
        self.d_vocab = d_vocab

        if bucket_type == "uniform":
            if y_min is None or y_max is None:
                raise ValueError(
                    "y_min/y_max must be provided for uniform bucketization."
                )
            if y_min >= y_max:
                raise ValueError(f"Require y_min < y_max (got {y_min} >= {y_max}).")
            borders = torch.linspace(float(y_min), float(y_max), steps=d_vocab + 1)
        elif bucket_type == "riemann":
            if borders is None:
                raise ValueError(
                    "Riemann bucketization requires precomputed borders.",
                )

        assert borders.ndim == 1, "Bucket borders must be 1D."
        assert len(borders) == d_vocab + 1, (
            f"Expected {d_vocab + 1} borders for {d_vocab} buckets (got {len(borders)})."
        )

        widths = borders[1:] - borders[:-1]
        assert torch.all(widths > 0), "Bucket borders must be strictly increasing."

        self.register_buffer("borders", borders.contiguous(), persistent=False)
        self.register_buffer("bucket_widths", widths, persistent=False)

        if bucket_support == "unbounded":
            if torch.isclose(widths[0], torch.tensor(0.0)) or torch.isclose(
                widths[-1],
                torch.tensor(0.0),
            ):
                raise ValueError(
                    "Side bucket widths must be > 0 for unbounded support."
                )

    @classmethod
    def from_config(cls, config: SupervisedRegressionPFNConfig) -> "Bucketizer":
        return cls(
            bucket_type=config.bucket_type,
            bucket_support=config.bucket_support,
            d_vocab=config.d_vocab,
            y_min=config.y_min,
            y_max=config.y_max,
            borders=config.riemann_borders,
        )

    @property
    def num_buckets(self) -> int:
        return self.d_vocab

    def bucketize(self, y: Float[Tensor, "..."]) -> Int[Tensor, "..."]:
        """Map continuous values to discrete bucket indices.

        Args:
            y: Continuous values to bucketize.

        Returns:
            Bucket indices in range [0, num_buckets-1].
        """
        borders = self.borders.to(device=y.device, dtype=y.dtype)
        indices = torch.searchsorted(borders, y, side="left") - 1
        return indices.clamp(min=0, max=self.num_buckets - 1)

    def decode(
        self,
        bucket_indices: Int[Tensor, "..."],
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> Float[Tensor, "..."]:
        """Convert bucket indices back to continuous values.

        Args:
            bucket_indices: Bucket indices to decode.
            dtype: Optional dtype for output tensor.

        Returns:
            Continuous values corresponding to bucket representatives.
        """
        reps = self.bucket_representatives()
        reps = reps.to(device=bucket_indices.device)
        if dtype is not None:
            reps = reps.to(dtype=dtype)
        bucket_indices = bucket_indices.clamp(min=0, max=self.num_buckets - 1)
        return reps[bucket_indices]

    def bucket_representatives(self) -> Float[Tensor, "num_buckets"]:
        """Get representative values for each bucket.

        For bounded support, uses bucket midpoints. For unbounded support,
        uses interior midpoints for edge buckets (since tails are only for extrapolation).

        Returns:
            Representative values for each bucket.
        """
        mids = self.borders[:-1] + 0.5 * self.bucket_widths
        return mids

    def log_bucket_densities(
        self,
        logits: Float[Tensor, "... num_buckets"],
    ) -> Float[Tensor, "... num_buckets"]:
        """Convert logits into log densities by accounting for bucket widths."""
        log_probs = torch.log_softmax(logits, dim=-1)
        log_widths = torch.log(self.bucket_widths).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        return log_probs - log_widths

    def log_density_at_values(
        self,
        logits: Float[Tensor, "... num_buckets"],
        values: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        """Return log density f(y) evaluated at arbitrary continuous targets."""
        y = values.to(device=logits.device, dtype=logits.dtype)
        log_probs = torch.log_softmax(logits, dim=-1)
        bucket_indices = self.bucketize(y).long()
        gathered = log_probs.gather(-1, bucket_indices.unsqueeze(-1)).squeeze(-1)
        widths = self.bucket_widths.to(device=y.device, dtype=y.dtype)

        # Assert widths are positive
        assert torch.all(widths > 0), "Bucket widths must be positive"

        if self.bucket_support == "bounded":
            selected_widths = widths[bucket_indices]
            return gathered - torch.log(selected_widths)

        borders = self.borders.to(device=y.device, dtype=y.dtype)
        selected_widths = widths[bucket_indices]
        result = torch.empty_like(gathered)
        num_buckets = self.num_buckets

        interior_mask = (bucket_indices > 0) & (bucket_indices < num_buckets - 1)
        if interior_mask.any():
            result[interior_mask] = gathered[interior_mask] - torch.log(
                selected_widths[interior_mask],
            )

        # Left bucket: split into interior and tail
        left_bucket_mask = bucket_indices == 0
        if left_bucket_mask.any():
            # Interior part: borders[0] <= y < borders[1]
            left_interior_mask = left_bucket_mask & (y >= borders[0])
            if left_interior_mask.any():
                result[left_interior_mask] = (
                    gathered[left_interior_mask]
                    + LOG_HALF
                    - torch.log(selected_widths[left_interior_mask])
                )

            # Tail part: y < borders[0]
            left_tail_mask = left_bucket_mask & (y < borders[0])
            if left_tail_mask.any():
                left_scale = self._halfnormal_scale(self.bucket_widths[0]).to(
                    device=y.device,
                    dtype=y.dtype,
                )
                distances = (borders[0] - y[left_tail_mask]).clamp_min(0.0)
                log_pdf = self._half_normal_log_pdf(distances, left_scale)
                result[left_tail_mask] = gathered[left_tail_mask] + LOG_HALF + log_pdf

        # Right bucket: split into interior and tail
        right_bucket_mask = bucket_indices == num_buckets - 1
        if right_bucket_mask.any():
            # Interior part: borders[-2] <= y < borders[-1]
            right_interior_mask = right_bucket_mask & (y < borders[-1])
            if right_interior_mask.any():
                result[right_interior_mask] = (
                    gathered[right_interior_mask]
                    + LOG_HALF
                    - torch.log(selected_widths[right_interior_mask])
                )

            # Tail part: y >= borders[-1]
            right_tail_mask = right_bucket_mask & (y >= borders[-1])
            if right_tail_mask.any():
                right_scale = self._halfnormal_scale(self.bucket_widths[-1]).to(
                    device=y.device,
                    dtype=y.dtype,
                )
                distances = (y[right_tail_mask] - borders[-1]).clamp_min(0.0)
                log_pdf = self._half_normal_log_pdf(distances, right_scale)
                result[right_tail_mask] = gathered[right_tail_mask] + LOG_HALF + log_pdf

        # Assert all values are finite
        assert torch.all(torch.isfinite(result)), "Log densities must be finite"

        return result

    def _left_tail_mean(self) -> Tensor:
        """Compute mean of left tail distribution for unbounded support."""
        scale = self._halfnormal_scale(self.bucket_widths[0])
        mean_offset = scale * SQRT_TWO_OVER_PI
        return self.borders[0] - mean_offset

    def _right_tail_mean(self) -> Tensor:
        """Compute mean of right tail distribution for unbounded support."""
        scale = self._halfnormal_scale(self.bucket_widths[-1])
        mean_offset = scale * SQRT_TWO_OVER_PI
        return self.borders[-1] + mean_offset

    @staticmethod
    def _halfnormal_scale(width: Tensor) -> Tensor:
        """Compute scale parameter for half-normal distribution from bucket width."""
        return width / HALF_NORMAL_ICDF_AT_TARGET

    @staticmethod
    def _half_normal_log_pdf(distances: Tensor, scale: Tensor) -> Tensor:
        """Compute log PDF of half-normal distribution."""
        return (
            LOG_SQRT_TWO_OVER_PI
            - torch.log(scale)
            - 0.5 * torch.square(distances / scale)
        )
