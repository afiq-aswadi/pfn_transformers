# PFN TransformerLens

Library for training Prior-Fitted Networks (PFN) with transformer-lens.

## Installation

### Install from Git

To use this package in another project:

```bash
# Basic installation
uv add git+https://github.com/afiq-aswadi/pfn_transformers.git

# With W&B support
uv add "git+https://github.com/afiq-aswadi/pfn_transformers.git[wandb]"
```

Or add to your `pyproject.toml`:

```toml
[project]
dependencies = [
    "pfn-transformerlens @ git+https://github.com/afiq-aswadi/pfn_transformers.git",
]
```

### Local Development

Clone the repository and install dependencies:

```bash
git clone https://github.com/afiq-aswadi/pfn_transformers.git
cd pfn_transformers

# Basic installation
uv sync

# With W&B support
uv sync --extra wandb
```

## Usage

### Training

#### Using DeterministicFunctionGenerator (function-based tasks)

```python
import torch
from pfn_transformerlens.model.configs.regression import SupervisedRegressionPFNConfig
from pfn_transformerlens.train import train, TrainingConfig
from pfn_transformerlens.sampler.data_generator import DeterministicFunctionGenerator

# Define task function (e.g., linear regression)
def linear_function(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w * x).sum(dim=-1)

# Setup data generator
data_gen = DeterministicFunctionGenerator(
    prior=torch.distributions.Normal(0.0, 1.0),  # distribution over function parameters
    function=linear_function,
    input_dim=10,
    noise_std=0.1,  # None for noiseless
    x_distribution=torch.distributions.Normal(0.0, 1.0)  # optional, defaults to N(0,1)
)

# Configure model
model_cfg = SupervisedRegressionPFNConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_dim=10
)

# Configure training
train_cfg = TrainingConfig(
    batch_size=32,
    seq_len=64,
    num_steps=10000,
    learning_rate=1e-4,
    use_wandb=True,
    wandb_project="my-project",  # or set WANDB_PROJECT env var
    wandb_entity="my-team",      # or set WANDB_ENTITY env var
    save_checkpoint=True,
    checkpoint_dir="checkpoints"
)

# Train
model = train(data_gen, model_cfg, train_cfg)
```

#### Using SupervisedProbabilisticGenerator (Bayesian workflow)

```python
from pfn_transformerlens.sampler.data_generator import SupervisedProbabilisticGenerator
from pfn_transformerlens.sampler.prior_likelihood import (
    PriorDistribution,
    LikelihoodDistribution,
    DiscreteTaskDistribution
)

# Define discrete tasks
tasks = torch.randn(1024)  # 1024 different task parameters
prior = PriorDistribution(DiscreteTaskDistribution(tasks))

# Define likelihood parameterizer
def normal_parameterizer(theta: torch.Tensor, x: torch.Tensor) -> dict:
    return {
        "loc": x.squeeze(-1) * theta,
        "scale": torch.ones_like(x.squeeze(-1)) * 0.1
    }

likelihood = LikelihoodDistribution(
    base_distribution=torch.distributions.Normal(0.0, 1.0),
    parameterizer=normal_parameterizer,
    input_dim=1
)

# Create generator
data_gen = SupervisedProbabilisticGenerator(
    prior=prior,
    likelihood=likelihood,
    x_distribution=torch.distributions.Normal(0.0, 1.0)  # optional
)
```

#### Other available generators

- `UnsupervisedProbabilisticGenerator` - for unsupervised learning (generates y only)
- `FixedDatasetGenerator` - sample from static dataset

### Sampling Data from Generators

Generators provide two ways to sample data:

#### Single sequence generation (use `.generate()` method)

```python
# Generate a single sequence
x, y = data_gen.generate(seq_len=64)
# x shape: (64, input_dim), y shape: (64,)
```

#### Batch generation (use standalone `sample_batch` function)

**Important**: Generators do NOT have a `.sample_batch()` method. Use the standalone function from the dataloader module:

```python
from pfn_transformerlens.sampler.dataloader import sample_batch

# Generate a batch of sequences
x_batch, y_batch = sample_batch(data_gen, batch_size=32, seq_len=64)
# x_batch shape: (32, 64, input_dim), y_batch shape: (32, 64)

# For unsupervised generators, x_batch will be None
unsupervised_gen = UnsupervisedProbabilisticGenerator(prior, likelihood)
x_batch, y_batch = sample_batch(unsupervised_gen, batch_size=32, seq_len=64)
# x_batch is None, y_batch shape: (32, 64)
```

#### Using dataloaders in training

The `train()` function handles batching automatically. You don't need to call `sample_batch` manually:

```python
# The train function uses build_dataloader internally
model = train(data_gen, model_cfg, train_cfg)
```

### Loading Models from Checkpoints

Load from local checkpoint:

```python
from pfn_transformerlens.checkpointing import load_checkpoint

model, optimizer_state, metadata = load_checkpoint(
    "checkpoints/checkpoint_step_5000.pt",
    device="cuda"
)

print(f"Loaded model trained at: {metadata.timestamp}")
print(f"Git hash: {metadata.git_hash}")
```

### Loading Models from W&B Artifacts

**Note**: Requires `uv sync --extra wandb` to install wandb.

Set environment variables for convenience:

```bash
export WANDB_PROJECT="my-project"
export WANDB_ENTITY="my-team"
```

Load by run ID (recommended):

```python
from pfn_transformerlens import wandb_utils

model, _, metadata = wandb_utils.load_from_pretrained(
    "abc123",  # wandb run ID
    checkpoint_step=5000,
    device="cuda"
)
```

Load latest checkpoint:

```python
model, _, metadata = wandb_utils.load_from_pretrained("abc123")
```

Load with optimizer state for resuming training:

```python
model, opt_state, metadata = wandb_utils.load_from_pretrained(
    "abc123",
    checkpoint_step=5000,
    load_optimizer=True
)

# Resume training
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(opt_state)
```

#### Loading by config parameters

Search and load models by matching config values:

```python
from pfn_transformerlens.wandb_utils import load_by_config, RunNameScheme

# Define which config fields to match on
scheme = RunNameScheme(
    model_fields=("n_layers", "d_model"),
    training_fields=("learning_rate",),
    data_fields=("input_dim",)
)

# Load model with specific config
model, _, metadata = load_by_config(
    scheme=scheme,
    n_layers=4,
    d_model=128,
    learning_rate=1e-4,
    checkpoint_step="latest",  # or "earliest" or specific step number
    device="cuda"
)
```

Create scheme from template configs:

```python
# Use actual config objects as templates
scheme = RunNameScheme.from_templates(
    model=SupervisedRegressionPFNConfig(),
    training=TrainingConfig(),
    data={"input_dim": 10}
)

# Only filters on fields defined in scheme
model, _, _ = load_by_config(
    scheme=scheme,
    n_layers=6,
    d_model=256
)
```

List available models:

```python
models = wandb_utils.list_available_models(
    project="my-project",
    entity="my-team",
    tags=["production"]
)

for model_info in models:
    print(f"{model_info.run_name}: {model_info.checkpoint_step} steps")
    print(f"  Config: {model_info.model_config}")
```

#### Other wandb utilities

List all checkpoints for a run:

```python
checkpoints = wandb_utils.list_checkpoints("abc123")
for ckpt in checkpoints:
    print(f"Step {ckpt.step}: {ckpt.artifact_name}")
```

Get checkpoint metadata without downloading:

```python
metadata = wandb_utils.get_checkpoint_metadata("abc123", checkpoint_step=5000)
print(f"Trained at: {metadata.timestamp}")
print(f"Git hash: {metadata.git_hash}")
```

Create structured run names:

```python
from pfn_transformerlens.wandb_utils import create_run_name, RunNameScheme

scheme = RunNameScheme(
    model_fields=("n_layers", "d_model"),
    training_fields=("learning_rate",)
)

run_name = create_run_name(
    base="pfn",
    model_config=model_cfg,
    training_config=train_cfg,
    scheme=scheme
)
# Result: "pfn_n4_d128_lr0.0001"
```

### Running Sweeps

Create a sweep configuration:

```python
from pfn_transformerlens.sweep import SweepConfig, run_sweep_agents
import wandb

# Define sweep
sweep_config = SweepConfig(
    method="grid",
    num_tasks=(32, 64, 128),
    learning_rate=(1e-4, 1e-3),
    d_model=(64, 128, 256)
)

# Create sweep
sweep_id = wandb.sweep(
    sweep_config.to_wandb_config(),
    project="my-project"
)

print(f"Sweep ID: {sweep_id}")
```

Run agents on multiple GPUs:

```python
# Run on all available GPUs
run_sweep_agents(sweep_id, gpus="all", project="my-project")

# Run on specific GPUs
run_sweep_agents(sweep_id, gpus="0,1,2", project="my-project")

# Or with list
run_sweep_agents(sweep_id, gpus=[0,1,2], project="my-project")
```

Alternatively, use the sweep.py script directly:

```bash
# Create sweep
python pfn_transformerlens/sweep.py create

# Run agents in parallel on GPUs 0,1,2
python pfn_transformerlens/sweep.py run-parallel --sweep-id abc123 --gpus "0,1,2"

# Or run single agent (manual GPU assignment)
CUDA_VISIBLE_DEVICES=0 python pfn_transformerlens/sweep.py run --sweep-id abc123
```

## Development

### Code Quality Checks

After making changes, run these checks:

```bash
# Format and lint
ruff check --fix . && ruff format .

# Type check
uvx ty check

# Tests
uv run pytest
```