"""
wandb callback factory for U-Time.

This module is safe to import even when `wandb` is not installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from .utils import (
    WandbMetricsLogger,
    WandbModelCheckpoint,
    is_wandb_available,
    wandb,
)

logger = logging.getLogger(__name__)

def create_wandb_callbacks(
    wandb_config: Dict[str, Any],
    hparams: Any,
    model_dir: str = "model"
) -> List:
    """
    Factory function to create all wandb callbacks based on configuration.
    
    Uses callback-specific config from wandb.callbacks section.
    Returns list of initialized wandb callbacks (official + custom).
    Handles graceful degradation if wandb is unavailable.
    
    Args:
        wandb_config: Wandb configuration dictionary from hparams (including callbacks section)
        hparams: Complete YAMLHParams object
        model_dir: Directory for saving model checkpoints
        
    Returns:
        List of initialized callback objects (empty list if wandb unavailable)
    """
    if not is_wandb_available():
        logger.info("wandb not available - skipping wandb callbacks")
        return []
    
    if not wandb.run:
        logger.warning("wandb run not initialized - cannot create callbacks")
        return []
    
    callbacks = []
    
    try:
        # Get callback-specific config
        callback_config = wandb_config.get('callbacks', {})
        
        # 1. Add WandbMetricsLogger for standard metrics
        log_freq = callback_config.get('log_freq', 'epoch')
        if isinstance(log_freq, int) and log_freq > 0:
            log_freq = 'epoch'  # For compatibility with Keras callback
        
        if WandbMetricsLogger is None:
            raise RuntimeError("wandb is available but WandbMetricsLogger could not be imported")
        callbacks.append(WandbMetricsLogger(log_freq=log_freq))
        logger.info(f"Added WandbMetricsLogger (log_freq={log_freq})")
        
        # 2. Add WandbModelCheckpoint if model logging is enabled
        if callback_config.get('log_model', False):
            checkpoint_path = os.path.join(model_dir, "wandb_checkpoint")
            if WandbModelCheckpoint is None:
                raise RuntimeError("wandb is available but WandbModelCheckpoint could not be imported")
            callbacks.append(
                WandbModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_dice',
                    mode='max',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
            )
            logger.info(f"Added WandbModelCheckpoint (path={checkpoint_path})")
        
        # 3. Add custom U-Time WandbCallback
        from utime.callbacks import WandbCallback
        callbacks.append(WandbCallback(config=callback_config, hparams=hparams))
        logger.info("Added custom U-Time WandbCallback")
        
    except Exception as e:
        logger.error(f"Error creating wandb callbacks: {e}")
        return []
    
    return callbacks