# Modified by AI Audio Studio (TadashiJLU), 2026: keep inference imports
# side-effect free while preserving Matcha's training utility exports lazily.
from importlib import import_module


_LAZY_EXPORTS = {
    "instantiate_callbacks": ("matcha.utils.instantiators", "instantiate_callbacks"),
    "instantiate_loggers": ("matcha.utils.instantiators", "instantiate_loggers"),
    "log_hyperparameters": ("matcha.utils.logging_utils", "log_hyperparameters"),
    "get_pylogger": ("matcha.utils.pylogger", "get_pylogger"),
    "enforce_tags": ("matcha.utils.rich_utils", "enforce_tags"),
    "print_config_tree": ("matcha.utils.rich_utils", "print_config_tree"),
    "extras": ("matcha.utils.utils", "extras"),
    "get_metric_value": ("matcha.utils.utils", "get_metric_value"),
    "task_wrapper": ("matcha.utils.utils", "task_wrapper"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
