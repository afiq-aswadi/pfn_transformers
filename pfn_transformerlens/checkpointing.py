"""Checkpoint save/load utilities. Decoupled from training logic."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import torch

from pfn_transformerlens.model.PFN import BasePFN, PFNModel
from pfn_transformerlens.model.configs.base import BasePFNConfig


@dataclass
class CheckpointMetadata:
    """Structured metadata for checkpoints."""

    timestamp: str
    wandb_run_id: str | None
    wandb_run_name: str | None
    wandb_run_url: str | None
    git_hash: str | None


def _get_git_hash() -> str | None:
    """Get current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _get_device(device_hint: str) -> str:
    """
    Resolve device hint to actual device string.

    Args:
        device_hint: "auto", "cuda", "mps", or "cpu"

    Returns:
        Device string for torch
    """
    if device_hint == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_hint


def save_checkpoint(
    checkpoint_path: Path,
    step: int,
    model_state: dict,
    optimizer_state: dict,
    model_config: BasePFNConfig,
    training_config,  # TrainingConfig - avoid circular import
    metadata: CheckpointMetadata,
    scheduler_state: dict | None = None,
) -> None:
    """
    Save checkpoint with v2 format.

    Args:
        checkpoint_path: Path to save checkpoint
        step: Current training step
        model_state: model.state_dict()
        optimizer_state: optimizer.state_dict()
        model_config: Model configuration
        training_config: Training configuration
        metadata: Checkpoint metadata with wandb/git info
        scheduler_state: Optional scheduler.state_dict()
    """
    checkpoint = {
        "checkpoint_version": 2,
        "step": step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "model_config": model_config,
        "training_config": training_config,
        "metadata": {
            "timestamp": metadata.timestamp,
            "wandb_run_id": metadata.wandb_run_id,
            "wandb_run_name": metadata.wandb_run_name,
            "wandb_run_url": metadata.wandb_run_url,
            "git_hash": metadata.git_hash,
        },
    }
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path | str,
    device: str = "auto",
    load_optimizer: bool = False,
) -> tuple[BasePFN, dict | None, CheckpointMetadata]:
    """
    Load checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on. "auto" = cuda > mps > cpu
        load_optimizer: If True, return optimizer state dict

    Returns:
        Tuple of (model, optimizer_state_dict | None, metadata)
        - model: Loaded BasePFN in eval mode on specified device
        - optimizer_state: State dict if load_optimizer=True, else None
        - metadata: CheckpointMetadata with wandb/git info

    Example:
        >>> model, opt_state, metadata = load_checkpoint(
        ...     "checkpoints/checkpoint_step_5000.pt",
        ...     device="cuda",
        ...     load_optimizer=True
        ... )
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract metadata
    metadata_dict = ckpt["metadata"]
    metadata = CheckpointMetadata(
        timestamp=metadata_dict.get("timestamp", ""),
        wandb_run_id=metadata_dict.get("wandb_run_id"),
        wandb_run_name=metadata_dict.get("wandb_run_name"),
        wandb_run_url=metadata_dict.get("wandb_run_url"),
        git_hash=metadata_dict.get("git_hash"),
    )

    # Load model
    config = ckpt["model_config"]
    device_str = _get_device(device)
    model = PFNModel(config).to(device_str)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    optimizer_state = ckpt["optimizer_state_dict"] if load_optimizer else None

    return model, optimizer_state, metadata
