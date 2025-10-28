"""PFN configuration classes for different task types."""

from .base import BasePFNConfig
from .classification import ClassificationPFNConfig
from .regression import SupervisedRegressionPFNConfig
from .unsupervised import UnsupervisedPFNConfig

# Backward compatibility alias
PFNConfig = SupervisedRegressionPFNConfig

__all__ = [
    "BasePFNConfig",
    "SupervisedRegressionPFNConfig",
    "ClassificationPFNConfig",
    "UnsupervisedPFNConfig",
    "PFNConfig",
]
