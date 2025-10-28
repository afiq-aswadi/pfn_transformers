from __future__ import annotations

import pytest
import torch

from pfn_transformerlens.model.bucketizer import (
    HALF_NORMAL_ICDF_AT_TARGET,
    SQRT_TWO_OVER_PI,
    Bucketizer,
)


def build_uniform_bucketizer(bucket_support: str) -> Bucketizer:
    return Bucketizer(
        bucket_type="uniform",
        bucket_support=bucket_support,  # type: ignore[arg-type]
        d_vocab=4,
        y_min=-2.0,
        y_max=2.0,
    )


def build_riemann_bucketizer(
    borders: list[float],
    bucket_support: str = "bounded",
) -> Bucketizer:
    return Bucketizer(
        bucket_type="riemann",
        bucket_support=bucket_support,  # type: ignore[arg-type]
        d_vocab=len(borders) - 1,
        y_min=None,
        y_max=None,
        borders=torch.tensor(borders),
    )


def test_uniform_bucket_mapping_midpoints() -> None:
    bucketizer = build_uniform_bucketizer("bounded")
    y = torch.tensor([-2.0, -0.9, 0.0, 1.7])
    indices = bucketizer.bucketize(y)
    assert indices.tolist() == [0, 1, 1, 3]

    reps = bucketizer.bucket_representatives()
    expected = torch.tensor([-1.5, -0.5, 0.5, 1.5])
    assert torch.allclose(reps, expected)

    decoded = bucketizer.decode(indices)
    assert torch.allclose(decoded, expected[[0, 1, 1, 3]])


def test_unbounded_bucket_tails_adjust_midpoints() -> None:
    bucketizer = build_uniform_bucketizer("unbounded")
    reps = bucketizer.bucket_representatives()
    widths = bucketizer.bucket_widths

    scale = widths[0] / HALF_NORMAL_ICDF_AT_TARGET
    left_expected = bucketizer.borders[1] - scale * SQRT_TWO_OVER_PI
    scale = widths[-1] / HALF_NORMAL_ICDF_AT_TARGET
    right_expected = bucketizer.borders[-2] + scale * SQRT_TWO_OVER_PI

    assert torch.isclose(reps[0], left_expected)
    assert torch.isclose(reps[-1], right_expected)


def test_bucketize_clamps_out_of_range_values() -> None:
    bucketizer = build_uniform_bucketizer("bounded")
    y = torch.tensor([-10.0, -2.001, -1.0, 0.0, 2.0, 10.0])
    indices = bucketizer.bucketize(y)
    assert indices.tolist() == [0, 0, 0, 1, 3, 3]


def test_riemann_bucketizer_respects_custom_borders() -> None:
    borders = [-3.0, -1.5, -0.2, 1.0, 4.0, 6.0]
    bucketizer = build_riemann_bucketizer(borders)
    widths = bucketizer.bucket_widths
    assert torch.allclose(
        widths,
        torch.tensor([1.5, 1.3, 1.2, 3.0, 2.0]),
    )

    y = torch.tensor([-2.0, -0.2, 0.0, 10.0])
    indices = bucketizer.bucketize(y)
    assert indices.tolist() == [0, 1, 2, 4]

    decoded = bucketizer.decode(indices)
    mids = bucketizer.bucket_representatives()
    assert torch.allclose(decoded, mids[indices])


def test_riemann_bucketizer_requires_borders() -> None:
    with pytest.raises(ValueError):
        Bucketizer(
            bucket_type="riemann",
            bucket_support="bounded",  # type: ignore[arg-type]
            d_vocab=3,
            y_min=None,
            y_max=None,
            borders=None,
        )


def test_log_bucket_densities_scale_softmax() -> None:
    bucketizer = build_uniform_bucketizer("bounded")
    logits = torch.zeros(2, bucketizer.num_buckets)
    log_densities = bucketizer.log_bucket_densities(logits)
    expected = torch.log_softmax(logits, dim=-1) - torch.log(bucketizer.bucket_widths)
    assert torch.allclose(log_densities, expected)


def test_log_density_at_values_matches_bucket_lookup_for_bounded() -> None:
    bucketizer = build_uniform_bucketizer("bounded")
    logits = torch.randn(4, bucketizer.num_buckets)
    y = torch.tensor([-1.7, -0.2, 0.4, 1.6])

    log_density = bucketizer.log_density_at_values(logits, y)
    log_bucket = bucketizer.log_bucket_densities(logits)
    gathered = log_bucket.gather(-1, bucketizer.bucketize(y).unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(log_density, gathered)


def test_log_density_at_values_uses_half_normal_tails_for_unbounded() -> None:
    bucketizer = build_uniform_bucketizer("unbounded")
    logits = torch.zeros(4, bucketizer.num_buckets)
    y = torch.tensor(
        [
            bucketizer.borders[0].item() - 0.5,
            bucketizer.borders[0].item() + 0.25,
            0.5,
            bucketizer.borders[-1].item() + 0.75,
        ],
        dtype=torch.float32,
    )

    log_density = bucketizer.log_density_at_values(logits, y)
    log_probs = torch.log_softmax(logits, dim=-1)

    left_scale = bucketizer._halfnormal_scale(bucketizer.bucket_widths[0]).to(
        log_probs.dtype
    )
    right_scale = bucketizer._halfnormal_scale(bucketizer.bucket_widths[-1]).to(
        log_probs.dtype
    )

    left_distance_tail = (bucketizer.borders[1] - y[0]).to(log_probs.dtype)
    left_distance_inside = (bucketizer.borders[1] - y[1]).to(log_probs.dtype)
    right_distance_tail = (y[3] - bucketizer.borders[-2]).to(log_probs.dtype)

    mid_bucket = int(bucketizer.bucketize(y[2:3]).item())
    mid_width = bucketizer.bucket_widths[mid_bucket].to(log_probs.dtype)

    expected = torch.stack(
        [
            log_probs[0, 0]
            + Bucketizer._half_normal_log_pdf(
                left_distance_tail,
                left_scale,
            ),
            log_probs[1, 0]
            + Bucketizer._half_normal_log_pdf(
                left_distance_inside,
                left_scale,
            ),
            log_probs[2, mid_bucket] - torch.log(mid_width),
            log_probs[3, bucketizer.num_buckets - 1]
            + Bucketizer._half_normal_log_pdf(
                right_distance_tail,
                right_scale,
            ),
        ],
    )

    assert torch.allclose(log_density, expected)
