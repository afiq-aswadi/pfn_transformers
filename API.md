TODO: check if correct, don't think the way they explain masking is right

# API Documentation

This codebase trains TransformerLens transformers on Bayesian inference tasks, supporting both supervised and unsupervised learning through flexible data generation strategies.

## Core Concept

The main goal is to train transformers to perform in-context Bayesian inference. Given a sequence of observations, the model learns to approximate posterior distributions or make predictions for new data points. This works for:

- **Regression**: Predict continuous values y given inputs x
- **Classification**: Predict discrete class labels
- **Supervised**: Learn from (x, y) pairs
- **Unsupervised**: Learn from sequences of y values only

## Data Generation Strategies

### 1. Prior-Likelihood Specification

Use `ProbabilisticGenerator` to specify Bayesian models through prior and likelihood distributions:

```python
from sampler.prior_likelihood import PriorDistribution, LikelihoodDistribution
from sampler.data_generator import ProbabilisticGenerator

# Define prior over task parameters θ
prior = PriorDistribution(DiscreteTaskDistribution(tasks))

# Define likelihood p(y|x,θ) 
likelihood = LikelihoodDistribution(...)

# Create generator
generator = ProbabilisticGenerator(prior=prior, likelihood=likelihood)
```

### 2. Direct Function Generators

For deterministic functions with optional noise:

```python
from sampler.data_generator import DeterministicFunctionGenerator

def linear_function(x, w):
    return w * x

generator = DeterministicFunctionGenerator(
    prior=prior,
    function=linear_function,
    input_dim=1,
    noise_std=0.1  # or None for noiseless
)
```

### 3. Fixed Dataset Generators

For learning from static datasets:

```python
from sampler.data_generator import FixedDatasetGenerator

generator = FixedDatasetGenerator(
    x_data=x_tensor,  # shape: (N, input_dim)
    y_data=y_tensor,  # shape: (N,)
    sequential=True   # sample consecutive subsequences
)
```

## Bucketing for Continuous Settings

For continuous regression with distributional predictions, use bucketing to discretize the output space:

### Uniform Bucketing
Evenly-spaced buckets over a bounded interval:

```python
config = SupervisedRegressionPFNConfig(
    prediction_type="distribution",
    bucket_type="uniform",
    bucket_support="bounded",
    y_min=-5.0,
    y_max=5.0,
    d_vocab=100  # number of buckets
)
```

### Riemann Bucketing
Quantile-based buckets estimated from data (see [arxiv:2112.10510](https://arxiv.org/abs/2112.10510)):

```python
from model.bucketizer import estimate_riemann_borders

# Estimate borders from training data
borders = estimate_riemann_borders(y_data, num_buckets=100)

config = SupervisedRegressionPFNConfig(
    prediction_type="distribution", 
    bucket_type="riemann",
    riemann_borders=borders,
    d_vocab=100
)
```

### Support Types
- `"bounded"`: Finite interval [y_min, y_max]
- `"unbounded"`: Infinite support with appropriate boundary handling

## Attention Mask Types

PFN models support two attention mask strategies that control how tokens attend to each other:

### autoregressive-pfn (PFN-style masking)

Custom attention mask designed for in-context Bayesian inference with supervised learning. This mask:

- **Interleaves x and y tokens**: Sequences are arranged as (x₁, y₁, x₂, y₂, ..., xₙ, yₙ)
- **Enables proper conditioning**: 
  - Each xₜ can attend to all previous (x, y) pairs: x₁, y₁, ..., xₜ₋₁, yₜ₋₁
  - Each yₜ can attend to current xₜ and all previous pairs: x₁, y₁, ..., xₜ₋₁, yₜ₋₁, xₜ
- **Preserves causality**: Future information never leaks into past predictions
- **Best for**: Supervised regression and classification with (x, y) pairs

This masking pattern ensures the model learns proper Bayesian inference: when predicting yₜ, it has access to the current input xₜ and all previous examples, mimicking how a Bayesian posterior incorporates new evidence.

### gpt2 (Standard causal masking)

Standard transformer causal attention where each token can only attend to previous tokens (including itself):

- **No interleaving**: Sequences remain as-is
- **Simple causality**: Token at position t attends to positions ≤ t
- **Best for**: Unsupervised next-token prediction models
- **Required for**: `UnsupervisedPFNConfig` (no x/y pairs to interleave)

Use this for pure sequence modeling tasks like density estimation or next-value prediction without explicit input features.

### Choosing a Mask Type

```python
# For supervised learning (regression/classification)
config = SupervisedRegressionPFNConfig(
    mask_type="autoregressive-pfn",  # default, recommended
    # or mask_type="gpt2"  # simpler but less structured
    ...
)

# For unsupervised learning (always gpt2)
config = UnsupervisedPFNConfig(
    mask_type="gpt2",  # only option, enforced by config
    ...
)
```

## Key Entry Points

### Training
```python
from train import train, load_model, TrainingConfig
from model.configs.regression import SupervisedRegressionPFNConfig

# Configure model
model_config = SupervisedRegressionPFNConfig(
    d_model=64,
    n_layers=2,
    input_dim=1,
    prediction_type="distribution"
)

# Configure training
training_config = TrainingConfig(
    batch_size=32,
    seq_len=64,
    num_steps=10000
)

# Train
model = train(
    data_generator=generator,
    model_config=model_config,
    training_config=training_config
)

# Or load from checkpoint
model = load_model("checkpoints/checkpoint_step_10000.pt")
```

### Model Configurations

- `SupervisedRegressionPFNConfig`: Regression with (x,y) pairs
- `ClassificationPFNConfig`: Classification tasks  
- `UnsupervisedPFNConfig`: Next-token prediction on y sequences

### Data Generators

All generators implement either:
- `SupervisedDataGenerator`: Returns `(x, y)` tuples
- `UnsupervisedDataGenerator`: Returns `y` sequences only

## Prototype Examples

See the `prototypes/` directory for complete working examples:

- `noiseless_linear_regression.py`: Simple linear regression with discrete tasks
- `exponential_gamma_unsupervised.py`: Unsupervised continuous modeling
- `generalized_linear_models.py`: More complex regression examples
- `beta_bernoulli_unsupervised.py`: Discrete unsupervised modeling

## Architecture

- **Model**: `model/PFN.py` - TransformerLens-based PFN implementations
- **Sampling**: `sampler/` - Data generation and prior/likelihood distributions  
- **Training**: `train.py` - Training loop and configuration
- **Bucketing**: `model/bucketizer.py` - Continuous-to-discrete mapping utilities
