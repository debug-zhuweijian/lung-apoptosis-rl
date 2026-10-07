"""Training-data API; original prepared files remain immutable."""

from .prepare import prepare_training
from .reader import TrainingData

__all__ = ["TrainingData", "prepare_training"]
