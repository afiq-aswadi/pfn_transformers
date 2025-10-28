"""PFN TransformerLens: Prior-Fitted Networks with transformer-lens."""

# Core training
from pfn_transformerlens.train import train, TrainingConfig, WandbLogger

# Model factory
from pfn_transformerlens.model.PFN import PFNModel as PFN

# Configs (short names)
from pfn_transformerlens.model.configs.regression import (
    SupervisedRegressionPFNConfig as RegressionConfig,
)
from pfn_transformerlens.model.configs.classification import (
    ClassificationPFNConfig as ClassificationConfig,
)
from pfn_transformerlens.model.configs.unsupervised import (
    UnsupervisedPFNConfig as UnsupervisedConfig,
)

# Generators (short names)
from pfn_transformerlens.sampler.data_generator import (
    DeterministicFunctionGenerator as DeterministicGenerator,
    FixedDatasetGenerator as DatasetGenerator,
    SupervisedProbabilisticGenerator as BayesianGenerator,
    UnsupervisedProbabilisticGenerator as UnsupervisedBayesian,
)

# Utilities
from pfn_transformerlens.sampler.dataloader import sample_batch
from pfn_transformerlens.model.bucketizer import estimate_riemann_borders

# Submodules
from pfn_transformerlens import checkpointing, wandb_utils
from pfn_transformerlens import generators, configs, bayes

# Rename wandb_utils to wandb for cleaner imports
wandb = wandb_utils

__version__ = "0.1.0"

__all__ = [
    # Training
    "train",
    "TrainingConfig",
    "WandbLogger",
    # Model
    "PFN",
    # Configs
    "RegressionConfig",
    "ClassificationConfig",
    "UnsupervisedConfig",
    # Generators
    "DeterministicGenerator",
    "BayesianGenerator",
    "DatasetGenerator",
    "UnsupervisedBayesian",
    # Utilities
    "sample_batch",
    "estimate_riemann_borders",
    # Submodules
    "checkpointing",
    "wandb",
    "wandb_utils",
    "generators",
    "configs",
    "bayes",
]
