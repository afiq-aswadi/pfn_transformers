# pfn_transformers: AI-Optimized Documentation

**Version:** 0.1.0
**Purpose:** Training Prior-Fitted Networks (PFNs) for in-context Bayesian inference

---

## Quick Start

```python
# Clean, intuitive imports
from pfn_transformers import (
    train, TrainingConfig,
    RegressionConfig,
    DeterministicGenerator,
    sample_batch,
)

# That's it! Everything you need in one line.
```

---

## What is pfn_transformers?

A PyTorch library for training **Prior-Fitted Networks (PFNs)** - transformers that perform **in-context Bayesian inference** without parameter updates.

**Use cases:**
- Supervised learning: regression, classification
- Unsupervised learning: next-token prediction
- Probabilistic predictions with uncertainty
- Few-shot learning via in-context examples

**Key features:**
- Clean API with intuitive naming
- Flexible data generation (Bayesian, deterministic, datasets)
- Distribution or point predictions
- Full W&B integration for reproducibility
- Built on TransformerLens for interpretability

---

## Installation

```bash
# From git
pip install git+https://github.com/yourusername/pfn_transformers.git

# Local + W&B
uv sync --extra wandb
```

---

## Complete Workflow

### Step 1: Data Generator

Choose how to generate training sequences:

#### Option A: Deterministic Functions

```python
from pfn_transformers import DeterministicGenerator
import torch

# Define function
def my_function(x, params):
    w, b = params
    return (w * x).sum(dim=-1) + b

# Define parameter prior
prior = torch.distributions.Independent(
    torch.distributions.Normal(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 0.5])
    ), 1
)

# Create generator
data_gen = DeterministicGenerator(
    prior=prior,
    function=my_function,
    input_dim=5,
    noise_std=0.1,  # or None for noiseless
)

x, y = data_gen.generate(seq_len=64)
# x: [64, 5], y: [64]
```

#### Option B: Bayesian Prior + Likelihood

```python
from pfn_transformers import BayesianGenerator
from pfn_transformers.bayes import Prior, Likelihood
import torch

# Prior over parameters
prior = Prior(torch.distributions.Normal(0.0, 1.0))

# Likelihood p(y | x, theta)
def parameterizer(theta, x):
    mean = (theta * x).sum(dim=-1)
    return {"loc": mean, "scale": 0.1}

likelihood = Likelihood(
    distribution_class=torch.distributions.Normal,
    parameterizer=parameterizer,
    input_dim=10
)

data_gen = BayesianGenerator(prior=prior, likelihood=likelihood)
x, y = data_gen.generate(seq_len=64)
```

#### Option C: Fixed Dataset

```python
from pfn_transformers import DatasetGenerator
import torch

x_data = torch.randn(1000, 10)  # [N, input_dim]
y_data = torch.randn(1000)       # [N]

data_gen = DatasetGenerator(
    x_data=x_data,
    y_data=y_data,
    sequential=False  # False: random sampling, True: consecutive
)
```

#### Option D: Unsupervised (no x values)

```python
from pfn_transformers import UnsupervisedBayesian
from pfn_transformers.bayes import Prior, Likelihood

prior = Prior(torch.distributions.Normal(0.0, 1.0))

def parameterizer(theta, x):
    # x is dummy [seq, 1]
    seq_len = x.shape[0]
    return {"loc": theta.expand(seq_len), "scale": 0.5}

likelihood = Likelihood(
    distribution_class=torch.distributions.Normal,
    parameterizer=parameterizer,
    input_dim=1  # dummy
)

data_gen = UnsupervisedBayesian(prior=prior, likelihood=likelihood)
y = data_gen.generate(seq_len=64)  # [64]
```

---

### Step 2: Model Configuration

#### Regression (Continuous Outputs)

```python
from pfn_transformers import RegressionConfig

# Point predictions (MSE loss)
config = RegressionConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_dim=10,
    prediction_type="point",
)

# Distribution predictions with uniform bucketing
config = RegressionConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_dim=10,
    prediction_type="distribution",
    bucket_type="uniform",
    d_vocab=100,  # number of buckets
    y_min=-5.0,
    y_max=5.0,
)

# Distribution with Riemann bucketing (quantile-based)
from pfn_transformers import estimate_riemann_borders

borders = estimate_riemann_borders(
    torch.randn(10000),  # sample data
    num_buckets=100
)

config = RegressionConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_dim=10,
    prediction_type="distribution",
    bucket_type="riemann",
    d_vocab=100,
    riemann_borders=borders,
)
```

**Key parameters:**
- `mask_type`: `"autoregressive-pfn"` (PFN attention) or `"gpt2"` (standard causal)
- `bucket_support`: `"unbounded"` (uses padding) or `"bounded"` (hard boundaries)

#### Classification (Discrete Outputs)

```python
from pfn_transformers import ClassificationConfig

config = ClassificationConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_dim=10,
    num_classes=5,
)
```

#### Unsupervised (Next-Token Prediction)

```python
from pfn_transformers import UnsupervisedConfig

# Discrete tokens (language modeling)
config = UnsupervisedConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_type="discrete",
    prediction_type="distribution",
    d_vocab=1000,  # vocabulary size
)

# Continuous with point predictions
config = UnsupervisedConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_type="continuous",
    prediction_type="point",
)

# Continuous with distribution predictions
config = UnsupervisedConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_type="continuous",
    prediction_type="distribution",
    bucket_type="uniform",
    d_vocab=100,
    y_min=-5.0,
    y_max=5.0,
)
```

**Note:** Unsupervised configs MUST use `mask_type="gpt2"` (enforced).

---

### Step 3: Training

```python
from pfn_transformers import train, TrainingConfig

training_config = TrainingConfig(
    # Data
    batch_size=32,
    seq_len=64,

    # Optimization
    num_steps=10000,
    learning_rate=1e-4,
    weight_decay=1e-5,

    # LR warmup
    use_warmup=True,
    warmup_steps=500,

    # Gradient clipping
    use_grad_clip=True,
    grad_clip=1.0,

    # Logging
    log_every=100,

    # Checkpoints
    save_checkpoint=True,
    save_every=1000,
    checkpoint_dir="checkpoints",

    # Device
    device="auto",  # cuda > mps > cpu
)

# Train!
model = train(data_gen, model_config, training_config)
```

**Optional: Evaluation during training**

```python
training_config = TrainingConfig(
    # ... other params ...
    eval_every=500,
    eval_batches=20,
)

model = train(
    data_gen,
    model_config,
    training_config,
    eval_data_generator=eval_data_gen  # optional separate eval data
)
```

---

### Step 4: W&B Integration

#### Basic Usage

```python
from pfn_transformers import train, TrainingConfig

training_config = TrainingConfig(
    # ... other params ...
    use_wandb=True,
    wandb_project="my-project",  # or set WANDB_PROJECT env var
    wandb_entity="myteam",       # or set WANDB_ENTITY env var
    wandb_run_name="experiment-1",
    wandb_tags=["regression", "v1"],
    wandb_log_model=True,  # upload checkpoints
)

model = train(data_gen, model_config, training_config)
```

**What gets logged:**
- Loss every step
- Metrics (MSE, accuracy) every `log_every` steps
- Checkpoints as artifacts: `checkpoint-{run_id}-step-{step}`
- Full configs in artifact metadata

#### Structured Run Names (for easy retrieval)

```python
from pfn_transformers.wandb import RunNameScheme, create_run_name
from dataclasses import dataclass

# Define data config
@dataclass
class MyDataConfig:
    num_tasks: int = 100
    input_dim: int = 10

data_config = MyDataConfig()

# Define naming scheme
scheme = RunNameScheme.from_templates(
    model={'n_layers': None, 'd_model': None},
    data={'num_tasks': None, 'input_dim': None}
)

# Create structured name
run_name = create_run_name(
    base="my-experiment",
    model_config=model_config,
    data_config=data_config,
    scheme=scheme
)
# Result: "my-experiment-model-n_layers4-d_model128-data-num_tasks100-input_dim10"

# Train with structured naming
training_config = TrainingConfig(
    use_wandb=True,
    wandb_run_name=run_name,
)

# IMPORTANT: pass data_config to train() for config-based loading
model = train(data_gen, model_config, training_config, data_config=data_config)
```

#### Loading Models from W&B

**By Run ID (recommended):**

```python
from pfn_transformers.wandb import load_from_pretrained

# Load latest checkpoint
model, _, metadata = load_from_pretrained(
    run_identifier="abc123",
    project="my-project",
)

# Load specific step
model, _, metadata = load_from_pretrained(
    run_identifier="abc123",
    checkpoint_step=5000,
    project="my-project",
)

print(f"Loaded from: {metadata.wandb_run_url}")
```

**By Config Matching:**

```python
from pfn_transformers.wandb import load_by_config, RunNameScheme

scheme = RunNameScheme.from_templates(
    model={'n_layers': None, 'd_model': None},
    data={'num_tasks': None}
)

model, _, metadata = load_by_config(
    scheme=scheme,
    n_layers=4,
    d_model=128,
    num_tasks=100,
    checkpoint_step="latest",  # or "earliest" or int
    project="my-project",
)
```

#### Browse Available Models

```python
from pfn_transformers.wandb import list_available_models, list_checkpoints

# List all models
models = list_available_models(
    project="my-project",
    tags=["production"]  # optional
)

for m in models:
    print(f"{m.run_name} (step {m.checkpoint_step})")
    print(f"  URL: {m.run_url}")

# List checkpoints for a run
checkpoints = list_checkpoints("abc123", project="my-project")
for c in checkpoints:
    print(f"Step {c.step}: {c.artifact_name}")
```

---

## Utilities

### Sample Batch

```python
from pfn_transformers import sample_batch

# Easy batching
x, y = sample_batch(data_gen, batch_size=32, seq_len=64)
# x: [32, 64, input_dim] or None (unsupervised)
# y: [32, 64]
```

### Estimate Riemann Borders

```python
from pfn_transformers import estimate_riemann_borders
import torch

# Sample from your data distribution
sample_data = torch.randn(10000)

# Estimate quantile-based bucket borders
borders = estimate_riemann_borders(sample_data, num_buckets=100)
# borders: [101] tensor with bucket boundaries
```

### Checkpointing

```python
from pfn_transformers import checkpointing

# Save
checkpointing.save_checkpoint(
    checkpoint_path="model.pt",
    step=10000,
    model_state=model.state_dict(),
    model_config=model_config,
    training_config=training_config,
)

# Load
model, opt_state, metadata = checkpointing.load_checkpoint(
    checkpoint_path="model.pt",
    device="auto",
    load_optimizer=False,
)
```

---

## Complete Examples

### Example 1: Regression with Clean Imports

```python
from pfn_transformers import (
    train, TrainingConfig,
    RegressionConfig,
    DeterministicGenerator,
)
import torch

# Data generator
def linear_fn(x, w):
    return (w * x).sum(dim=-1)

data_gen = DeterministicGenerator(
    prior=torch.distributions.Normal(0.0, 1.0),
    function=linear_fn,
    input_dim=10,
    noise_std=0.1,
)

# Model config
model_config = RegressionConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_dim=10,
    prediction_type="distribution",
    bucket_type="uniform",
    d_vocab=100,
    y_min=-5.0,
    y_max=5.0,
)

# Training config
training_config = TrainingConfig(
    batch_size=32,
    seq_len=64,
    num_steps=10000,
    use_wandb=True,
    wandb_project="pfn-experiments",
)

# Train
model = train(data_gen, model_config, training_config)

# Inference
device = next(model.parameters()).device
x_test = torch.randn(1, 20, 10).to(device)
y_test = torch.randn(1, 20).to(device)

with torch.no_grad():
    logits = model(x_test, y_test)
    log_densities = model.bucketizer.log_bucket_densities(logits)
    pred_buckets = log_densities.argmax(dim=-1)
    pred_y = model.get_y_values(pred_buckets)
```

### Example 2: Classification

```python
from pfn_transformers import (
    train, TrainingConfig,
    ClassificationConfig,
    DatasetGenerator,
)
import torch

# Data
x_data = torch.randn(1000, 10)
y_data = torch.randint(0, 3, (1000,))

data_gen = DatasetGenerator(x_data, y_data)

# Config
config = ClassificationConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_dim=10,
    num_classes=3,
)

# Train
model = train(
    data_gen,
    config,
    TrainingConfig(batch_size=32, seq_len=32, num_steps=5000),
)

# Inference
x_test = torch.randn(1, 10, 10).to(device)
y_test = torch.randint(0, 3, (1, 10)).to(device)

with torch.no_grad():
    logits = model(x_test, y_test)
    predictions = logits.argmax(dim=-1)
```

### Example 3: Config-Based W&B Loading

```python
from pfn_transformers.wandb import RunNameScheme, create_run_name, load_by_config
from pfn_transformers import train, TrainingConfig, RegressionConfig
from dataclasses import dataclass

# Data config
@dataclass
class DataConfig:
    num_tasks: int = 50
    input_dim: int = 10

data_config = DataConfig()

# Naming scheme
scheme = RunNameScheme.from_templates(
    model={'n_layers': None, 'd_model': None},
    data={'num_tasks': None, 'input_dim': None}
)

# Create config
model_config = RegressionConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_head=32,
    input_dim=10,
    prediction_type="point",
)

# Train with structured name
run_name = create_run_name(
    base="my-exp",
    model_config=model_config,
    data_config=data_config,
    scheme=scheme,
)

model = train(
    data_gen,
    model_config,
    TrainingConfig(use_wandb=True, wandb_run_name=run_name),
    data_config=data_config,  # pass to train()!
)

# Later: load by config
model_loaded, _, meta = load_by_config(
    scheme=scheme,
    n_layers=4,
    d_model=128,
    num_tasks=50,
    input_dim=10,
    checkpoint_step="latest",
    project="my-project",
)
```

---

## API Reference

### Top-Level Imports

```python
from pfn_transformers import (
    # Training
    train,
    TrainingConfig,
    WandbLogger,

    # Model
    PFN,  # factory function

    # Configs
    RegressionConfig,
    ClassificationConfig,
    UnsupervisedConfig,

    # Generators
    DeterministicGenerator,
    BayesianGenerator,
    DatasetGenerator,
    UnsupervisedBayesian,

    # Utilities
    sample_batch,
    estimate_riemann_borders,

    # Submodules
    checkpointing,
    wandb,
    generators,
    configs,
    bayes,
)
```

### Submodule Imports (Alternative)

```python
# Generators with short names
from pfn_transformers.generators import (
    Deterministic,
    Bayesian,
    Dataset,
    UnsupervisedBayesian,
)

# Configs with short names
from pfn_transformers.configs import (
    Regression,
    Classification,
    Unsupervised,
)

# Bayesian utilities
from pfn_transformers.bayes import (
    Prior,
    Likelihood,
)
```

### Function Signatures

#### Training

```python
train(
    data_generator: DataGenerator,
    model_config: RegressionConfig | ClassificationConfig | UnsupervisedConfig,
    training_config: TrainingConfig,
    *,
    resume_from: str | None = None,
    eval_data_generator: DataGenerator | None = None,
    data_config: Any = None,  # dataclass for W&B logging
) -> model
```

#### Model Factory

```python
PFN(config) -> model
```

Creates appropriate model based on config type.

#### Sample Batch

```python
sample_batch(
    data_generator: DataGenerator,
    batch_size: int,
    seq_len: int,
) -> tuple[Tensor | None, Tensor]
```

Returns `(x, y)` where x is None for unsupervised.

#### Estimate Riemann Borders

```python
estimate_riemann_borders(
    sample_data: Tensor,  # [N]
    num_buckets: int,
) -> Tensor  # [num_buckets + 1]
```

#### W&B Loading

```python
# By run ID
load_from_pretrained(
    run_identifier: str,
    checkpoint_step: int | None = None,
    project: str | None = None,
    entity: str | None = None,
    device: str = "auto",
    load_optimizer: bool = False,
) -> tuple[model, dict | None, metadata]

# By config
load_by_config(
    scheme: RunNameScheme,
    checkpoint_step: int | "latest" | "earliest" = "latest",
    device: str = "auto",
    project: str | None = None,
    entity: str | None = None,
    **config_filters,  # e.g., n_layers=4, d_model=128
) -> tuple[model, dict | None, metadata]
```

---

## Configuration Parameters

### RegressionConfig

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Model dimension |
| `n_layers` | int | Number of layers |
| `n_heads` | int | Attention heads |
| `d_head` | int | Head dimension |
| `input_dim` | int | Input feature dimension |
| `prediction_type` | `"distribution"` \| `"point"` | Prediction mode |
| `mask_type` | `"autoregressive-pfn"` \| `"gpt2"` | Attention mask |
| `bucket_type` | `"uniform"` \| `"riemann"` \| None | Bucketing (for distribution) |
| `bucket_support` | `"unbounded"` \| `"bounded"` | Support type |
| `d_vocab` | int | Number of buckets |
| `y_min` | float | Min value (uniform) |
| `y_max` | float | Max value (uniform) |
| `riemann_borders` | Tensor | Borders (riemann) |

### ClassificationConfig

Same as RegressionConfig, plus:

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_classes` | int | Number of classes |

### UnsupervisedConfig

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_type` | `"discrete"` \| `"continuous"` | Input type |
| `prediction_type` | `"distribution"` \| `"point"` | Prediction mode |
| `d_vocab` | int | Vocab size or buckets |
| `mask_type` | `"gpt2"` | Must be gpt2 |
| Other | ... | Same as RegressionConfig |

### TrainingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 32 | Batch size |
| `seq_len` | int | 64 | Sequence length |
| `num_steps` | int | 10000 | Training steps |
| `learning_rate` | float | 1e-4 | Learning rate |
| `weight_decay` | float | 1e-5 | Weight decay |
| `use_warmup` | bool | True | LR warmup |
| `warmup_steps` | int | 500 | Warmup steps |
| `use_grad_clip` | bool | True | Gradient clipping |
| `grad_clip` | float | 1.0 | Max gradient norm |
| `log_every` | int | 100 | Logging interval |
| `save_checkpoint` | bool | True | Save checkpoints |
| `save_every` | int | 1000 | Checkpoint interval |
| `checkpoint_dir` | str | "checkpoints" | Directory |
| `eval_every` | int \| None | None | Eval interval |
| `eval_batches` | int | 10 | Batches per eval |
| `device` | str | "auto" | Device |
| `use_wandb` | bool | False | Enable W&B |
| `wandb_project` | str \| None | None | W&B project |
| `wandb_entity` | str \| None | None | W&B entity |
| `wandb_run_name` | str \| None | None | Run name |
| `wandb_log_model` | bool | True | Upload checkpoints |
| `wandb_tags` | list[str] \| None | None | Tags |
| `wandb_notes` | str \| None | None | Notes |

---

## Best Practices

### Device Placement

Always move inputs to model device:

```python
device = next(model.parameters()).device
x = x.to(device)
y = y.to(device)
logits = model(x, y)
```

### W&B Environment Variables

```bash
export WANDB_PROJECT="my-project"
export WANDB_ENTITY="myteam"
```

Then omit from TrainingConfig:

```python
TrainingConfig(use_wandb=True)  # uses env vars
```

### Data Config for Reproducibility

Always define and pass data_config:

```python
from dataclasses import dataclass

@dataclass
class DataConfig:
    num_tasks: int
    input_dim: int

model = train(data_gen, model_config, training_config, data_config=DataConfig(...))
```

---

## Development

```bash
# Format and lint
ruff check --fix .
ruff format .

# Type check
uvx ty check

# Tests
uv run pytest

# Specific test
uv run pytest tests/test_train.py -xvs
```

---

## Package Structure

```
pfn_transformerlens/
├── __init__.py              # Top-level exports
├── generators.py            # Generator shortcuts
├── configs.py               # Config shortcuts
├── bayes.py                 # Bayesian utilities
├── train.py                 # Training loop
├── checkpointing.py         # Checkpoint utilities
├── wandb_utils.py           # W&B utilities
├── model/
│   ├── PFN.py              # Model implementations
│   ├── PFNMasks.py         # Attention masks
│   ├── bucketizer.py       # Bucketing
│   └── configs/            # Config classes
└── sampler/
    ├── data_generator.py   # Generator implementations
    ├── prior_likelihood.py # Bayesian classes
    └── dataloader.py       # Batching utilities
```

---

## Common Patterns

### Basic Training Workflow

```python
from pfn_transformers import (
    train, TrainingConfig, RegressionConfig, DeterministicGenerator
)

# 1. Data
data_gen = DeterministicGenerator(...)

# 2. Config
model_config = RegressionConfig(...)
training_config = TrainingConfig(...)

# 3. Train
model = train(data_gen, model_config, training_config)

# 4. Inference
with torch.no_grad():
    logits = model(x.to(device), y.to(device))
```

### W&B Workflow

```python
from pfn_transformers import train, TrainingConfig
from pfn_transformers.wandb import RunNameScheme, create_run_name, load_by_config
from dataclasses import dataclass

# 1. Define data config
@dataclass
class DataConfig:
    num_tasks: int = 100

# 2. Define scheme
scheme = RunNameScheme.from_templates(
    model={'n_layers': None},
    data={'num_tasks': None}
)

# 3. Create run name
run_name = create_run_name(
    base="exp",
    model_config=model_config,
    data_config=DataConfig(),
    scheme=scheme,
)

# 4. Train
model = train(
    data_gen,
    model_config,
    TrainingConfig(use_wandb=True, wandb_run_name=run_name),
    data_config=DataConfig(),
)

# 5. Load later
model, _, _ = load_by_config(
    scheme=scheme,
    n_layers=4,
    num_tasks=100,
    project="my-project",
)
```

---

## License

MIT License

---

## Support

Issues: https://github.com/yourusername/pfn_transformers/issues
Docs: https://pfn-transformers.readthedocs.io
Paper: https://arxiv.org/abs/2112.10510
