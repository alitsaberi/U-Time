"""
Weights & Biases (wandb) integration utilities for U-Time/U-Sleep.

Safe to import even when `wandb` is not installed; all functionality degrades
gracefully when unavailable/disabled.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if wandb is available
try:
    import wandb
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    WandbMetricsLogger = None
    WandbModelCheckpoint = None


def is_wandb_available() -> bool:
    """
    Check if wandb is installed and available for import.
    
    Returns:
        bool: True if wandb is available, False otherwise
    """
    return WANDB_AVAILABLE


def is_wandb_enabled(wandb_config: Dict[str, Any]) -> bool:
    """
    Return whether wandb is enabled from an effective wandb config dict.

    Expected input is the merged config produced by `merge_wandb_args_with_hparams`
    (hparams + env vars + CLI args), so all precedence rules have already been applied.
    """
    try:
        return bool((wandb_config or {}).get("enabled", False))
    except Exception:
        return False


def init_wandb_run(
    config: Dict[str, Any],
    job_type: str
) -> Optional[Any]:
    """
    Initialize or resume a wandb run from an effective config dict.
    
    Behavior:
    - If wandb is not enabled (`config['enabled']` is falsy), returns None.
    - If wandb is enabled but not installed/available, returns None.
    - If a run id is present in the config (`run_id`/`id`), resumes that run.
      Otherwise, creates a new run.
    
    Args:
        config: Effective wandb config dict (already merged from hparams/env/CLI)
        job_type: W&B job type string (e.g. "training", "evaluation", "prediction")
    
    Returns:
        wandb.run object if successful, None otherwise
    """
    final_config = (config or {}).copy()
    if not is_wandb_enabled(final_config):
        logger.info("wandb is not enabled - skipping wandb run")
        return None

    if not is_wandb_available():
        logger.warning("wandb is not installed. Run 'pip install wandb' to enable experiment tracking.")
        return None
    
    try:
        run_id = final_config.get("run_id")
        resume = final_config.get("resume", "allow")

        # Extract wandb.init parameters (global settings only)
        init_params = {
            "project": final_config.get("project", "u-time"),
            "entity": final_config.get("entity"),
            "name": final_config.get("name") or f"{job_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "group": final_config.get("group"),
            "tags": final_config.get("tags", []),
            "notes": final_config.get("notes"),
            "job_type": job_type,
        }

        if run_id:
            init_params["id"] = run_id
            init_params["resume"] = resume
            # Avoid project mismatch issues when resuming (common source of errors)
            # unless explicitly forced in config.
            if not final_config.get("force_project_on_resume", False):
                init_params.pop("project", None)
        
        # Remove None values
        init_params = {k: v for k, v in init_params.items() if v is not None}
        
        # Initialize wandb run
        run = wandb.init(**init_params)
        
        if run:
            logger.info(f"Initialized wandb run: {run.name} (ID: {run.id})")
            logger.info(f"View run at: {run.url}")
        
        return run
        
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        logger.info("Continuing without wandb logging...")
        return None

def finish_wandb_run() -> None:
    """
    Properly finish the current wandb run.
    """
    if is_wandb_available() and wandb.run:
        try:
            wandb.finish()
            logger.info("Finished wandb run")
        except Exception as e:
            logger.warning(f"Error finishing wandb run: {e}")

def _str2bool(v):
    """
    Parse common truthy/falsey strings to bool.

    Allows: true/false, 1/0, yes/no, on/off (case-insensitive).
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    if not isinstance(v, str):
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got {type(v)}")
    s = v.strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{v}'. Use one of: true/false, 1/0, yes/no."
    )


def _parse_env_tags(value: str) -> List[str]:
    # Accept comma-separated and/or whitespace-separated values
    parts = re.split(r"[,\s]+", (value or "").strip())
    return [p for p in parts if p]


def add_wandb_arguments(parser):
    """
    Add Weights & Biases (wandb) arguments to an ArgumentParser.
    
    This function adds common wandb CLI arguments that are shared across
    train, evaluate, and predict scripts. These arguments can override
    the settings in hparams.yaml.
    
    Args:
        parser: ArgumentParser object to add arguments to
        
    Returns:
        ArgumentParser with wandb arguments added
    """
    # Global wandb arguments (work for all scripts)
    # NOTE: default=None so scripts can still enable wandb via hparams/env vars
    # when the user does not specify the --wandb flag.
    parser.add_argument(
        "--wandb",
        nargs="?",
        const=True,
        default=None,
        type=_str2bool,
        help="Enable/disable W&B experiment tracking. If provided without a value, enables wandb. "
             "You may also pass an explicit boolean value (e.g. --wandb=true, --wandb false). "
             "When omitted, defaults to hparams/env configuration.",
    )
    # Accept both dash and underscore variants for compatibility/convenience
    parser.add_argument("--wandb-run-id", "--wandb_run_id", dest="wandb_run_id", type=str, default=None,
                        help="W&B run ID to resume. If provided, resumes the existing run for logging. "
                             "If not provided and --wandb is set, creates a new run.")
    parser.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", type=str, default=None,
                        help="W&B project name (overrides hparams.yaml)")
    parser.add_argument("--wandb-entity", "--wandb_entity", dest="wandb_entity", type=str, default=None,
                        help="W&B entity/team name (overrides hparams.yaml)")
    parser.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", type=str, default=None,
                        help="W&B run name (overrides hparams.yaml; auto-generated if not provided)")
    parser.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", type=str, default=None,
                        help="W&B group name for organizing related runs (overrides hparams.yaml)")
    parser.add_argument("--wandb-tags", "--wandb_tags", dest="wandb_tags", nargs='*', type=str, default=None,
                        help="W&B tags for the run, space-separated (overrides hparams.yaml)")
    parser.add_argument("--wandb-save-artifact", "--wandb_save_artifact", dest="wandb_save_artifact", action="store_true", default=False)
    
    return parser


def create_wandb_config(hparams, args):
    """
    Create an effective wandb config from hparams, env vars, and CLI args.
    
    Priority order (lowest -> highest):
    - hparams.yaml (`hparams['wandb']`)
    - environment variables (`WANDB_*`)
    - CLI args (`--wandb-*`)
    
    Args:
        hparams: YAMLHParams object containing configuration
        args: Parsed command-line arguments (Namespace object)
        
    Returns:
        tuple: (wandb_config, env_overrides, cli_overrides) - wandb config, environment overrides, and CLI overrides
    """
    base: Dict[str, Any] = hparams.get("wandb", {}).copy() if getattr(hparams, "get", None) and hparams.get("wandb") else {}

    # Environment overrides (common W&B env vars)
    env_overrides: Dict[str, Any] = {}
    if os.environ.get("WANDB_PROJECT"):
        env_overrides["project"] = os.environ["WANDB_PROJECT"]
    if os.environ.get("WANDB_ENTITY"):
        env_overrides["entity"] = os.environ["WANDB_ENTITY"]

    # WANDB_MODE influences enablement; keep it in config for transparency.
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode:
        env_overrides["mode"] = wandb_mode
        if wandb_mode.strip().lower() == "disabled":
            env_overrides["enabled"] = False

    # CLI overrides
    cli_overrides: Dict[str, Any] = {}
    if hasattr(args, "wandb") and args.wandb is not None:
        cli_overrides["enabled"] = bool(args.wandb)
    # If a run id is provided, we treat this as an explicit request to use wandb
    # unless the user explicitly disabled it via --wandb=false.
    if getattr(args, "wandb_run_id", None):
        cli_overrides["run_id"] = args.wandb_run_id
        if "enabled" not in cli_overrides:
            cli_overrides["enabled"] = True
    if getattr(args, "wandb_project", None):
        cli_overrides["project"] = args.wandb_project
    if getattr(args, "wandb_entity", None):
        cli_overrides["entity"] = args.wandb_entity
    if getattr(args, "wandb_name", None):
        cli_overrides["name"] = args.wandb_name
    if getattr(args, "wandb_group", None):
        cli_overrides["group"] = args.wandb_group
    if getattr(args, "wandb_tags", None):
        cli_overrides["tags"] = args.wandb_tags
    if getattr(args, "wandb_notes", None):
        cli_overrides["notes"] = args.wandb_notes

    # Convenience flags used by downstream scripts
    if getattr(args, "wandb_save_artifact", False):
        cli_overrides["save_artifact"] = True

    # Merge all sources in order: base <- env <- cli
    wandb_config: Dict[str, Any] = {}
    wandb_config.update(base)
    wandb_config.update(env_overrides)
    wandb_config.update(cli_overrides)

    return wandb_config, env_overrides, cli_overrides
