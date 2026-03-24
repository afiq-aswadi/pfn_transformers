# Retrospective
This library is primarily intended for my own research with PFNs, and is meant to be a library to train classes of transformers on synthetic data generators. I intended to use this project as a way to learn high-level software engineering skills. This was also around when agentic coding tools (Codex, Claude Code) became more widespread, so I wanted to use this library as a way to develop my own workflow when it comes to writing research code.

Overall I'm pretty happy with what I learned, and I feel I got what I wanted out of developing this library. Some things I learned include:
- Using git issues and pull requests. When I want to work on a feature that seems to have a wide scope, I draft an issue with CC. I then use the issue to plan (and implement) the implementation. I have Codex review setup to review any pull request.
- Basic test-driven development: for particularly technical implementations I had CC implement tests to check if implementations were correct.
- General git/version control, basic OOP.
- Using linters and type checkers (though I'm unsure if I'll continue using these, especially in collaborative projects)

Some major mistakes I made and learned from:
- Scope creep was a major issue for me. I may have been overambitious with what I wanted this library to be able to do, which led to unecessary abstractions. The YAGNI principle probably holds here. 
- Certain parts of the codebase became a black box for me, especially when I started dealing with libraries I wasn't familiar with. I fell into a bad habit of copy and pasting issues raised by Codex review, expecting CC to fix the problem without actually looking at the code myself. This meant I didn't have a deep understanding of the codebase, and this made it difficult for me to iterate. I should note that I think it's fine to have certain parts of a codebase be a black box (e.g:scripts to set up training sweeps from a yaml file, or plotting scripts), but I should definitely be more aware of what parts of the codebase I want to deeply understand.  
- Not thinking about the simplest possible solution. For instance, I tried setting up weird infra to download/upload checkpoints from wandb to run experiments, when setting up cloud storage probably would have been a simpler solution.  

Some things I can do better:
- Actually reading code AI writes and trying to understand what's going on (and using this as a learning tool to learn libraries I'm not familiar with!). 
- Being more opinionated with how code should look like. It's not just about writing code that runs. Though this will probably take time for me to develop that 'taste' of what good code looks like.

*The documentation below was written by AI.*

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

### Structured W&B Run Names

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
