"""
Backward-compatible wandb integration import surface.

Historically, U-Time scripts imported wandb helpers from `utime.utils.wandb_logger`.
The implementation now lives under `utime.utils.wandb`.
"""

from utime.utils.wandb import (  # noqa: F401
    WANDB_AVAILABLE,
    WandbMetricsLogger,
    WandbModelCheckpoint,
    create_wandb_callbacks,
    finish_wandb_run,
    init_wandb_run,
    is_wandb_available,
    is_wandb_enabled,
    log_confusion_matrix,
    log_dataset_info,
    log_prediction_summary,
    log_validation_metrics,
    wandb,
)

__all__ = [
    "WANDB_AVAILABLE",
    "WandbMetricsLogger",
    "WandbModelCheckpoint",
    "wandb",
    "is_wandb_available",
    "is_wandb_enabled",
    "init_wandb_run",
    "finish_wandb_run",
    "create_wandb_callbacks",
    "log_dataset_info",
    "log_prediction_summary",
    "log_confusion_matrix",
    "log_validation_metrics",
]

