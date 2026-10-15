"""Training-data API; source-native readers do not require training dependencies."""

from importlib import import_module

__all__ = ["TrainingData", "prepare_training"]


def __getattr__(name):
    # Keep the public API; load Parquet/training dependencies only when requested.
    modules = {"TrainingData": ".reader", "prepare_training": ".prepare"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
