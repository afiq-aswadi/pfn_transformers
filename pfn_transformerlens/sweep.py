"""Simple wandb sweep script for GPU instances.

Usage:
    # 1. Create sweep and get ID
    uv run python sweep.py create

    # 2. Run agents on different GPUs (in separate terminals/tmux)
    CUDA_VISIBLE_DEVICES=0 uv run python sweep.py run --sweep-id <id>
    CUDA_VISIBLE_DEVICES=1 uv run python sweep.py run --sweep-id <id>
    CUDA_VISIBLE_DEVICES=2 uv run python sweep.py run --sweep-id <id>

    # Or run single agent (will keep pulling configs until done)
    CUDA_VISIBLE_DEVICES=0 uv run python sweep.py run --sweep-id <id> --count 10

    # wandb agent CLI always expects entity/project/sweep_id
    wandb agent <entity>/<project>/<sweep_id>
"""

from dataclasses import dataclass
import os
import subprocess

import torch
import tyro

from pfn_transformerlens.model.configs import SupervisedRegressionPFNConfig
from pfn_transformerlens.sampler.data_generator import DeterministicFunctionGenerator
from pfn_transformerlens.sampler.dataloader import sample_batch
from pfn_transformerlens.sampler.prior_likelihood import (
    DiscreteTaskDistribution,
    PriorDistribution,
)
from pfn_transformerlens.train import TrainingConfig, train


def get_available_gpus() -> list[int]:
    """
    Get available CUDA GPU IDs.

    Returns:
        List of GPU IDs, empty list if no CUDA available
    """
    try:
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except (ImportError, RuntimeError):
        return []


def parse_gpu_spec(gpus: str | list[int] | None) -> list[int]:
    """
    Parse GPU specification.

    Args:
        gpus: "all", "0,1,2", [0,1,2], or None (defaults to all)

    Returns:
        List of GPU IDs

    Raises:
        ValueError: If no CUDA devices available or invalid GPU IDs
    """
    available = get_available_gpus()

    if not available:
        raise ValueError(
            "No CUDA devices found. "
            "Ensure CUDA is available and torch was built with CUDA support."
        )

    if gpus is None or gpus == "all":
        return available

    if isinstance(gpus, str):
        # Parse "0,1,2" format
        gpu_ids = [int(x.strip()) for x in gpus.split(",")]
    else:
        gpu_ids = list(gpus)

    # Validate
    invalid = [g for g in gpu_ids if g not in available]
    if invalid:
        raise ValueError(f"Invalid GPU IDs: {invalid}. Available: {available}")

    return gpu_ids


def run_sweep_agents(
    sweep_id: str,
    gpus: str | list[int] | None = None,
    project: str | None = None,
    entity: str | None = None,
) -> None:
    """
    Launch wandb agents on specified GPUs.

    Each GPU gets one agent process. Agents pull configs from wandb
    and handle scheduling automatically.

    Args:
        sweep_id: wandb sweep ID
        gpus: GPU specification ("all", "0,1", [0,1], or None for all)
        project: wandb project (defaults to WANDB_PROJECT env)
        entity: wandb entity (defaults to WANDB_ENTITY env)

    Example:
        >>> run_sweep_agents("abc123", gpus="0,1,2")
    """
    gpu_ids = parse_gpu_spec(gpus)

    project = project or os.getenv("WANDB_PROJECT")
    entity = entity or os.getenv("WANDB_ENTITY")
    if not project or not entity:
        raise ValueError(
            "Must provide project/entity or set WANDB_PROJECT/WANDB_ENTITY"
        )

    print(f"Launching {len(gpu_ids)} wandb agents on GPUs: {gpu_ids}")

    processes = []
    for gpu_id in gpu_ids:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        proc = subprocess.Popen(
            ["wandb", "agent", f"{entity}/{project}/{sweep_id}"],
            env=env,
        )
        processes.append(proc)
        print(f"  Agent on GPU {gpu_id}: PID {proc.pid}")

    # Wait for all
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nTerminating agents...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()


@dataclass
class SweepConfig:
    """Wandb sweep configuration."""

    method: str = "grid"
    metric_name: str = "final_test_mse"
    metric_goal: str = "minimize"

    # parameters to sweep
    num_tasks: tuple[int, ...] = (2, 5, 10, 20)
    learning_rate: tuple[float, ...] | None = None
    d_model: tuple[int, ...] | None = None

    def to_wandb_config(self) -> dict:
        """Convert to wandb sweep config format."""
        parameters = {"num_tasks": {"values": list(self.num_tasks)}}

        if self.learning_rate:
            parameters["learning_rate"] = {"values": list(self.learning_rate)}
        if self.d_model:
            parameters["d_model"] = {"values": list(self.d_model)}

        return {
            "method": self.method,
            "metric": {"name": self.metric_name, "goal": self.metric_goal},
            "parameters": parameters,
        }


@dataclass
class DataConfig:
    """Data generator configuration."""

    num_tasks: int
    task_type: str = "linear_regression"
    input_dim: int = 1
    noise_std: float | None = None
    x_distribution: str = "Normal(0, 1)"


def create_data_generator(
    num_tasks: int,
) -> tuple[DeterministicFunctionGenerator, DataConfig]:
    """Create data generator for linear regression."""
    tasks = torch.randn(num_tasks)
    prior = PriorDistribution(DiscreteTaskDistribution(tasks))

    def linear_function(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (w * x).squeeze(-1)

    x_dist = torch.distributions.Normal(0.0, 1.0)

    generator = DeterministicFunctionGenerator(
        prior=prior.base_distribution,
        function=linear_function,
        input_dim=1,
        noise_std=None,
        x_distribution=x_dist,
    )

    config = DataConfig(num_tasks=num_tasks)
    return generator, config


def train_one_run():
    """Train one model (called by wandb agent)."""
    import os
    import wandb

    os.environ["WANDB_CONSOLE"] = "off"
    _ = wandb.init(settings=wandb.Settings(_disable_stats=True, _disable_meta=True))

    # get sweep parameters
    num_tasks = wandb.config.num_tasks
    learning_rate = wandb.config.get("learning_rate", 1e-3)
    d_model = wandb.config.get("d_model", 64)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "auto")
    print(f"\n{'=' * 60}")
    print(f"Run: num_tasks={num_tasks}, lr={learning_rate}, d_model={d_model}")
    print(f"GPU: {gpu_id}")
    print(f"{'=' * 60}\n")

    # create data
    data_generator, data_config = create_data_generator(num_tasks)

    # model config
    model_config = SupervisedRegressionPFNConfig(
        d_vocab=1,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        d_head=16,
        d_mlp=256,
        n_ctx=64,
        act_fn="gelu",
        use_pos_emb=True,
        prediction_type="point",
        bucket_type=None,
        input_dim=1,
        mask_type="gpt2",
    )

    # training config
    training_config = TrainingConfig(
        batch_size=8,
        seq_len=32,
        num_steps=200,
        learning_rate=learning_rate,
        log_every=50,
        save_checkpoint=False,
        eval_every=None,
        device="auto",
        use_wandb=False,
    )

    # log all configs
    wandb.config.update(
        {
            "model": model_config.__dict__,
            "training": training_config.__dict__,
            "data": data_config.__dict__,
            "gpu_id": gpu_id,
        },
        allow_val_change=True,
    )

    # train
    model = train(
        data_generator=data_generator,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
    )

    # evaluate
    print("\nEvaluating...")
    model.eval()
    x_test, y_test = sample_batch(
        data_generator, batch_size=32, seq_len=training_config.seq_len
    )

    device = next(model.parameters()).device
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        pred = model(x_test, y_test)
        final_mse = torch.nn.functional.mse_loss(pred.squeeze(-1), y_test).item()

    print(f"Final MSE: {final_mse:.4f}")
    wandb.log({"final_test_mse": final_mse})
    wandb.finish()


@dataclass
class CreateArgs:
    """Arguments for creating a sweep."""

    project: str = "pfn-sweeps"


@dataclass
class RunArgs:
    """Arguments for running a sweep agent."""

    sweep_id: str
    count: int | None = None
    """Number of runs for this agent (None = run until sweep done)."""


def create(args: CreateArgs):
    """Create a new wandb sweep."""
    import wandb

    # use default sweep config (edit SweepConfig class to customize)
    sweep_config = SweepConfig()

    print("Creating sweep...")
    print(f"Project: {args.project}")
    print(f"Parameters: {sweep_config}")
    print()

    sweep_id = wandb.sweep(
        sweep_config.to_wandb_config(),
        project=args.project,
    )

    print("=" * 60)
    print("SWEEP CREATED!")
    print("=" * 60)
    print(f"Sweep ID: {sweep_id}")
    print(f"URL: https://wandb.ai/<entity>/{args.project}/sweeps/{sweep_id}")
    print()
    print("Run agents with:")
    print(f"  CUDA_VISIBLE_DEVICES=0 uv run python sweep.py run --sweep-id {sweep_id}")
    print(f"  CUDA_VISIBLE_DEVICES=1 uv run python sweep.py run --sweep-id {sweep_id}")
    print()


def run(args: RunArgs):
    """Run a wandb sweep agent."""
    import wandb

    print("=" * 60)
    print(f"Starting agent for sweep: {args.sweep_id}")
    print(f"Count: {args.count or 'unlimited'}")
    print("=" * 60)
    print()

    wandb.agent(args.sweep_id, function=train_one_run, count=args.count)

    print()
    print("=" * 60)
    print("AGENT FINISHED!")
    print("=" * 60)


@dataclass
class RunParallelArgs:
    """Arguments for running sweep agents on multiple GPUs."""

    sweep_id: str
    gpus: str = "all"
    """GPU specification: "all", "0,1,2", etc."""
    project: str | None = None
    """Wandb project (defaults to WANDB_PROJECT env)."""
    entity: str | None = None
    """Wandb entity (defaults to WANDB_ENTITY env)."""


def run_parallel(args: RunParallelArgs):
    """Run wandb agents on multiple GPUs in parallel."""
    print("=" * 60)
    print("RUNNING PARALLEL SWEEP AGENTS")
    print("=" * 60)
    print(f"Sweep ID: {args.sweep_id}")
    print(f"GPUs: {args.gpus}")
    resolved_project = args.project or os.getenv("WANDB_PROJECT", "default")
    resolved_entity = args.entity or os.getenv("WANDB_ENTITY", "<entity unset>")
    print(f"Project: {resolved_project}")
    print(f"Entity: {resolved_entity}")
    print("=" * 60)
    print()

    run_sweep_agents(
        args.sweep_id,
        gpus=args.gpus,
        project=args.project,
        entity=args.entity,
    )


if __name__ == "__main__":
    tyro.cli(
        {
            "create": create,
            "run": run,
            "run-parallel": run_parallel,
        }
    )
