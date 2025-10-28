"""General training loop for Prior-Fitted Network (PFN).

This module provides a reusable training pipeline where callers can provide:
- Prior distribution
- Likelihood distribution
- PFN model configuration (PFNConfig)
- Training configuration (TrainingConfig)

It returns a trained PFNModel and handles checkpointing, logging, and basic
metrics (cross-entropy loss).

For training w/ multiple GPUs, just use simple-gpu-scheduler (https://pypi.org/project/simple-gpu-scheduler/)

"""
# TODO: finetuning?
# TODO: better naming scheme... or at least better way to specify where checkpoints are saved... (check: how do people do it in huggingface/wandb?)
# TODO: training w/ multiple GPUs, uploading to wandb, logging training via wandb etc
# TODO: easy way to train config sweeps + over gpus?

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from tqdm import tqdm

from pfn_transformerlens.checkpointing import (
    CheckpointMetadata,
    _get_git_hash,
    save_checkpoint,
)
from pfn_transformerlens.model.configs import (
    BasePFNConfig,
    ClassificationPFNConfig,
    SupervisedRegressionPFNConfig,
    UnsupervisedPFNConfig,
)
from pfn_transformerlens.model.PFN import BasePFN, PFNModel, UnsupervisedPFN
from pfn_transformerlens.sampler.data_generator import DataGenerator
from pfn_transformerlens.sampler.dataloader import build_dataloader, sample_batch


@dataclass
class TrainingConfig:
    """Training configuration for PFN.

    Attributes:
        batch_size: DataLoader batch size.
        seq_len: Sequence length (x,y are interleaved internally by the model).
        num_steps: Number of optimization steps.
        learning_rate: Base learning rate.
        weight_decay: Weight decay for AdamW.
        use_warmup: Whether to use linear LR warmup.
        warmup_steps: Linear LR warmup steps (only used if use_warmup=True).
        use_grad_clip: Whether to apply gradient clipping.
        grad_clip: Gradient clipping max norm (only used if use_grad_clip=True).
        log_every: Steps between progress logs (avg over window).
        log_distributional_mse: Whether to compute an approximate MSE metric when
            training density models (adds an argmax over buckets each step).
        save_checkpoint: Whether to save checkpoints during training.
        save_every: Steps between checkpoints.
        checkpoint_dir: Directory to save checkpoints.
        eval_every: Steps between evaluations (None = no evaluation).
        eval_batches: Number of batches to average for eval metrics.
        device: "auto", "cuda", "mps", or "cpu".
        use_wandb: Whether to enable wandb logging.
        wandb_project: Wandb project name (falls back to WANDB_PROJECT env var).
        wandb_entity: Wandb entity/username (falls back to WANDB_ENTITY env var).
        wandb_run_name: Optional run name (wandb auto-generates if None).
        wandb_log_model: Whether to upload checkpoints as wandb artifacts.
        wandb_tags: Optional list of tags for the wandb run.
        wandb_notes: Optional notes/description for the wandb run.
    """

    batch_size: int = 32
    seq_len: int = 64
    num_steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_warmup: bool = True
    warmup_steps: int = 500
    use_grad_clip: bool = True
    grad_clip: float = 1.0
    log_every: int = 100
    log_distributional_mse: bool = False
    save_checkpoint: bool = True
    save_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    eval_every: Optional[int] = None
    eval_batches: int = 10
    device: str = "auto"
    use_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_log_model: bool = True
    wandb_tags: list[str] | None = None
    wandb_notes: str | None = None

    def get_device(self) -> str:
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device


class WandbLogger:
    """Lightweight wandb logging wrapper.

    Handles wandb initialization, metric logging, and checkpoint uploads.
    No-op when use_wandb=False. Lazy imports wandb only when enabled.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        model_config: BasePFNConfig,
        data_config: Any = None,
    ) -> None:
        self.enabled = training_config.use_wandb
        self.run_id: str | None = None
        self.run_name: str | None = None
        self.run_url: str | None = None

        # Store configs for artifact metadata
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config

        if not self.enabled:
            return

        # lazy import wandb only if enabled
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is required for logging. Install with: uv sync --extra wandb"
            )

        # read env vars as fallbacks
        project = training_config.wandb_project or os.getenv("WANDB_PROJECT")
        entity = training_config.wandb_entity or os.getenv("WANDB_ENTITY")

        # build config dict
        wandb_config = {
            "model": model_config.__dict__,
            "training": training_config.__dict__,
        }
        if data_config is not None:
            from dataclasses import asdict, is_dataclass

            if is_dataclass(data_config):
                wandb_config["data"] = asdict(data_config)
            else:
                raise TypeError("data_config must be a dataclass")

        # initialize wandb run
        self.wandb.init(
            project=project,
            entity=entity,
            name=training_config.wandb_run_name,
            tags=training_config.wandb_tags,
            notes=training_config.wandb_notes,
            config=wandb_config,
        )

        # Store run info for checkpoint metadata
        if self.wandb.run is not None:
            self.run_id = self.wandb.run.id
            self.run_name = self.wandb.run.name
            self.run_url = self.wandb.run.url

        self.log_model = training_config.wandb_log_model

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to wandb."""
        if not self.enabled:
            return
        self.wandb.log(metrics, step=step)

    def log_checkpoint(
        self,
        checkpoint_path: str | Path,
        step: int,
        metadata: CheckpointMetadata | None = None,
    ) -> None:
        """Upload checkpoint as wandb artifact with run.id-based naming."""
        if not self.enabled or not self.log_model:
            return

        artifact_metadata = {
            "step": step,
            "run_name": self.wandb.run.name,
            "run_id": self.wandb.run.id,
            "run_url": self.wandb.run.url,
        }
        if metadata is not None:
            artifact_metadata.update(
                {
                    "timestamp": metadata.timestamp,
                    "git_hash": metadata.git_hash,
                    "wandb_run_id": metadata.wandb_run_id,
                    "wandb_run_name": metadata.wandb_run_name,
                    "wandb_run_url": metadata.wandb_run_url,
                }
            )

        # Add full configs to artifact metadata for config-based loading
        artifact_metadata["model_config"] = self.model_config.__dict__
        artifact_metadata["training_config"] = self.training_config.__dict__
        if self.data_config is not None:
            from dataclasses import asdict, is_dataclass

            if is_dataclass(self.data_config):
                artifact_metadata["data_config"] = asdict(self.data_config)
            else:
                artifact_metadata["data_config"] = None

        # Use run.id for unique artifact naming
        artifact = self.wandb.Artifact(
            name=f"checkpoint-{self.wandb.run.id}-step-{step}",
            type="model",
            description=f"Checkpoint at step {step}",
            metadata=artifact_metadata,
        )
        artifact.add_file(str(checkpoint_path))
        self.wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Cleanup wandb run."""
        if not self.enabled:
            return
        self.wandb.finish()


def compute_loss(
    model: BasePFN,
    x: Float[torch.Tensor, "batch seq input_dim"] | None,
    y: Float[torch.Tensor, "batch seq"],
    *,
    log_distributional_mse: bool = False,
) -> Tuple[Float[torch.Tensor, ""], dict[str, float]]:
    """Compute PFN loss and metrics.

    Supports distributional NLL, pointwise MSE, classification CE, or unsupervised CE.

    Args:
        model: PFN model (supervised or unsupervised).
        x: Input features for supervised models. Must be None for unsupervised models.
        y: Target values or observations.
        log_distributional_mse: Whether to compute MSE for distributional predictions.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    if isinstance(model, UnsupervisedPFN):
        assert x is None, "x must be None for unsupervised models"
        logits = model(y)
    else:
        assert x is not None, "x must be provided for supervised models"
        logits = model(x, y)

    metrics: dict[str, float] = {}

    if isinstance(model.config, UnsupervisedPFNConfig):
        # Ensure y is on the same device as logits (HookedTransformer may move to different device)
        y = y.to(logits.device)

        if (
            model.config.input_type == "discrete"
            and model.config.prediction_type == "distribution"
        ):
            # Discrete next-token prediction with cross-entropy
            logits_flat = logits[:, :-1, :].reshape(-1, model.config.d_vocab)
            targets_flat = y[:, 1:].long().reshape(-1)
            loss = nn.functional.cross_entropy(
                logits_flat, targets_flat, reduction="mean"
            )
            metrics["loss"] = float(loss.item())

            with torch.no_grad():
                preds = logits[:, :-1, :].argmax(dim=-1)
                acc = (preds == y[:, 1:].long()).float().mean()
            metrics["accuracy"] = float(acc.item())

            return loss, metrics

        elif (
            model.config.input_type == "continuous"
            and model.config.prediction_type == "point"
        ):
            # Continuous next-value prediction with MSE
            preds = logits[:, :-1, :].squeeze(-1)
            targets = y[:, 1:]
            mse = nn.functional.mse_loss(preds, targets, reduction="mean")
            metrics["loss"] = float(mse.item())
            metrics["mse"] = float(mse.item())
            return mse, metrics

        elif (
            model.config.input_type == "continuous"
            and model.config.prediction_type == "distribution"
        ):
            # Continuous distribution prediction with NLL
            logits_shifted = logits[:, :-1, :]
            targets_shifted = y[:, 1:]

            bucketizer = model.bucketizer
            log_density_at_targets = bucketizer.log_density_at_values(
                logits_shifted, targets_shifted
            )
            loss = -log_density_at_targets.mean()
            metrics["loss"] = float(loss.item())

            if log_distributional_mse:
                with torch.no_grad():
                    log_densities = bucketizer.log_bucket_densities(logits_shifted)
                    pred_buckets = log_densities.argmax(dim=-1)
                    pred_y = model.get_y_values(pred_buckets)
                    mse = nn.functional.mse_loss(
                        pred_y, targets_shifted, reduction="mean"
                    )
                metrics["mse"] = float(mse.item())

            return loss, metrics

        else:
            # Discrete + point prediction (rare but valid)
            preds = logits[:, :-1, :].squeeze(-1)
            targets = y[:, 1:].float()
            mse = nn.functional.mse_loss(preds, targets, reduction="mean")
            metrics["loss"] = float(mse.item())
            metrics["mse"] = float(mse.item())
            return mse, metrics

    if isinstance(model.config, ClassificationPFNConfig):
        logits_flat = logits.view(-1, model.config.num_classes)
        targets_flat = y.long().view(-1)
        loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction="mean")
        metrics["loss"] = float(loss.item())

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == y.long()).float().mean()
        metrics["accuracy"] = float(acc.item())

        return loss, metrics

    if (
        isinstance(model.config, SupervisedRegressionPFNConfig)
        and model.config.prediction_type == "point"
    ):
        preds = logits.squeeze(-1)
        mse = nn.functional.mse_loss(preds, y, reduction="mean")
        metrics["loss"] = float(mse.item())
        metrics["mse"] = float(mse.item())
        return mse, metrics

    bucketizer = model.bucketizer
    log_density_at_targets = bucketizer.log_density_at_values(logits, y)
    loss = -log_density_at_targets.mean()
    metrics["loss"] = float(loss.item())

    if log_distributional_mse:
        with torch.no_grad():
            log_densities = bucketizer.log_bucket_densities(logits)
            pred_buckets = log_densities.argmax(dim=-1)
            pred_y = model.get_y_values(pred_buckets)
            mse = nn.functional.mse_loss(pred_y, y, reduction="mean")
        metrics["mse"] = float(mse.item())

    return loss, metrics


# TODO: is this really the best way to do this, feels redundant tbh.
def lr_with_warmup(step: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def evaluate_model(
    model: BasePFN,
    data_generator: DataGenerator,
    eval_batches: int,
    seq_len: int,
    batch_size: int,
    device: str,
    log_distributional_mse: bool = False,
) -> dict[str, float]:
    """Evaluate model on validation data.

    Args:
        model: PFN model to evaluate.
        data_generator: DataGenerator for validation data.
        eval_batches: Number of batches to average over.
        seq_len: Sequence length for evaluation.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.
        log_distributional_mse: Whether to compute MSE for distributional predictions.

    Returns:
        Dictionary of averaged evaluation metrics.
    """
    model.eval()
    total_metrics: dict[str, float] = {}

    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = sample_batch(data_generator, batch_size, seq_len)
            if x is not None:
                x = x.to(device)
            y = y.to(device)

            _, metrics = compute_loss(
                model, x, y, log_distributional_mse=log_distributional_mse
            )

            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value

    # Average metrics
    avg_metrics = {key: value / eval_batches for key, value in total_metrics.items()}

    model.train()
    return avg_metrics


def train(
    data_generator: DataGenerator,
    model_config: BasePFNConfig,
    training_config: TrainingConfig,
    *,
    resume_from: Optional[str] = None,
    eval_data_generator: Optional[DataGenerator] = None,
    data_config: Any = None,
) -> BasePFN:
    """Run a general PFN training loop.

    Args:
        data_generator: DataGenerator instance that produces training sequences.
            Use ProbabilisticGenerator to wrap prior/likelihood if needed.
        model_config: BasePFNConfig for the transformer backbone.
        training_config: Optimization + runtime configuration.
        resume_from: Optional path to a checkpoint to resume from.
        eval_data_generator: Optional separate DataGenerator for validation.
            If None and eval_every is set, uses training generator for eval.
        data_config: Optional dataclass with data config (e.g., num_tasks, input_dim).
            Logged to wandb for config-based model loading with RunNameScheme.
            Kept as dataclass throughout, only converted to dict at wandb boundary.

    Returns:
        Trained PFN model (BasePFN subclass).
    """
    device = training_config.get_device()

    # IO setup
    ckpt_dir = Path(training_config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # wandb logging
    logger = WandbLogger(training_config, model_config, data_config)

    # Model + data
    model = PFNModel(model_config)
    model = model.to(device)

    dataloader = build_dataloader(data_generator, training_config)

    data_iter = iter(dataloader)

    # TODO: Support other optimizers via TrainingConfig (e.g., optimizer_kwargs dict)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    start_step = 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_step = int(state.get("step", 0))

    model.train()
    pbar = tqdm(range(start_step, training_config.num_steps), desc="Training")
    run_loss = 0.0
    run_mse = 0.0
    run_acc = 0.0
    mse_count = 0
    acc_count = 0

    for step in pbar:
        x, y = next(data_iter)

        # validate data matches config type (only check first batch)
        if step == start_step:
            if isinstance(model.config, UnsupervisedPFNConfig):
                assert x is None, (
                    "unsupervised models must receive x=None from dataloader, "
                    f"got x with shape {x.shape if x is not None else None}"
                )
                assert y.ndim == 2, (
                    f"y must be 2D (batch, seq) for unsupervised, got shape {y.shape}"
                )

                if (
                    model.config.input_type == "discrete"
                    and model.config.prediction_type == "distribution"
                ):
                    assert y.dtype in (torch.long, torch.int32, torch.int64), (
                        f"discrete unsupervised models require integer y (token indices), got dtype {y.dtype}"
                    )
                    assert y.min() >= 0 and y.max() < model.config.d_vocab, (
                        f"y values must be in [0, {model.config.d_vocab}), "
                        f"got range [{y.min().item()}, {y.max().item()}]"
                    )
            else:
                # supervised models (classification or regression)
                assert x is not None, (
                    "supervised models require x from dataloader, got x=None. "
                    "use UnsupervisedPFNConfig for models without x"
                )
                assert x.ndim == 3, (
                    f"x must be 3D (batch, seq, input_dim), got shape {x.shape}"
                )
                assert y.ndim == 2, f"y must be 2D (batch, seq), got shape {y.shape}"
                assert x.shape[0] == y.shape[0], (
                    f"x and y batch sizes must match, got {x.shape[0]} and {y.shape[0]}"
                )
                assert x.shape[1] == y.shape[1], (
                    f"x and y sequence lengths must match, got {x.shape[1]} and {y.shape[1]}"
                )
                assert x.shape[2] == model.config.input_dim, (
                    f"x input_dim must match config.input_dim, "
                    f"got x.shape[2]={x.shape[2]} but config.input_dim={model.config.input_dim}"
                )

                if isinstance(model.config, ClassificationPFNConfig):
                    assert y.dtype in (torch.long, torch.int32, torch.int64), (
                        f"classification models require integer y (class indices), got dtype {y.dtype}"
                    )
                    assert y.min() >= 0 and y.max() < model.config.num_classes, (
                        f"y values must be in [0, {model.config.num_classes}), "
                        f"got range [{y.min().item()}, {y.max().item()}]"
                    )

        if x is not None:
            x = x.to(device)
        y = y.to(device)

        # LR schedule
        if training_config.use_warmup:
            lr = lr_with_warmup(
                step, training_config.warmup_steps, training_config.learning_rate
            )
            for g in optimizer.param_groups:
                g["lr"] = lr
        else:
            lr = training_config.learning_rate

        # Forward + backward
        loss, metrics = compute_loss(
            model,
            x,
            y,
            log_distributional_mse=training_config.log_distributional_mse,
        )

        # log loss every step
        logger.log({"loss": metrics["loss"]}, step)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if training_config.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.grad_clip
            )
        optimizer.step()

        # Metrics window
        run_loss += metrics["loss"]
        mse_value = metrics.get("mse")
        if mse_value is not None:
            run_mse += mse_value
            mse_count += 1
        acc_value = metrics.get("accuracy")
        if acc_value is not None:
            run_acc += acc_value
            acc_count += 1

        if (step + 1) % training_config.log_every == 0:
            avg_loss = run_loss / training_config.log_every
            postfix = {
                "loss": f"{avg_loss:.4f}",  # TODO: specify loss function
                "lr": f"{lr:.6f}",
            }

            # prepare wandb metrics
            wandb_metrics = {
                "train/loss": avg_loss,
                "train/lr": lr,
            }

            if mse_count > 0:
                avg_mse = run_mse / mse_count
                postfix["mse"] = f"{avg_mse:.4f}"
                wandb_metrics["train/mse"] = avg_mse

            if acc_count > 0:
                avg_acc = run_acc / acc_count
                postfix["acc"] = f"{avg_acc:.4f}"
                wandb_metrics["train/accuracy"] = avg_acc

            pbar.set_postfix(postfix)
            logger.log(wandb_metrics, step + 1)

            run_loss = 0.0
            run_mse = 0.0
            run_acc = 0.0
            mse_count = 0
            acc_count = 0

        if (
            training_config.save_checkpoint
            and (step + 1) % training_config.save_every == 0
        ):
            from datetime import datetime

            ckpt_path = ckpt_dir / f"checkpoint_step_{step + 1}.pt"

            # Create metadata
            metadata = CheckpointMetadata(
                timestamp=datetime.now().isoformat(),
                wandb_run_id=logger.run_id if logger.enabled else None,
                wandb_run_name=logger.run_name if logger.enabled else None,
                wandb_run_url=logger.run_url if logger.enabled else None,
                git_hash=_get_git_hash(),
            )

            # Save checkpoint with metadata
            save_checkpoint(
                checkpoint_path=ckpt_path,
                step=step + 1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                model_config=model_config,
                training_config=training_config,
                metadata=metadata,
            )
            pbar.write(f"Saved checkpoint to {ckpt_path}")
            logger.log_checkpoint(ckpt_path, step + 1, metadata)

        # Evaluation
        if (
            training_config.eval_every is not None
            and (step + 1) % training_config.eval_every == 0
        ):
            eval_gen = (
                eval_data_generator
                if eval_data_generator is not None
                else data_generator
            )
            eval_metrics = evaluate_model(
                model=model,
                data_generator=eval_gen,
                eval_batches=training_config.eval_batches,
                seq_len=training_config.seq_len,
                batch_size=training_config.batch_size,
                device=device,
                log_distributional_mse=training_config.log_distributional_mse,
            )

            # Format and display eval metrics
            eval_str = "Eval: " + ", ".join(
                f"{key}={value:.4f}" for key, value in eval_metrics.items()
            )
            pbar.write(eval_str)

            # log to wandb with eval/ prefix
            wandb_eval_metrics = {
                f"eval/{key}": value for key, value in eval_metrics.items()
            }
            logger.log(wandb_eval_metrics, step + 1)

    logger.finish()
    return model


__all__ = [
    "TrainingConfig",
    "WandbLogger",
    "compute_loss",
    "lr_with_warmup",
    "evaluate_model",
    "train",
]
