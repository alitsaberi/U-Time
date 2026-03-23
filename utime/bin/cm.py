"""
Script to compute confusion matrices from one or more pairs of true/pred .npz
files of labels
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from scipy import stats
from utime.evaluation import concatenate_true_pred_pairs
from utime.evaluation import (f1_scores_from_cm, precision_scores_from_cm,
                              recall_scores_from_cm)
from utime.evaluation.metrics import class_wise_kappa
from utime.evaluation.dataframe import add_to_eval_df, log_eval_df, with_grand_mean_col
from utime.evaluation.plotting import plot_and_save_cm, plot_and_save_hypnogram
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)

DEFAULT_CMAP = "Blues"
TRUE_SUFFIX = "_TRUE"
PRED_SUFFIX = "_PRED"

# Define standard mappings for different classification scenarios
MAPPINGS = {
    2: {  # Binary: Wake vs Sleep
        0: "Wake",
        1: "Sleep"
    },
    3: {  # 3-class: Wake, NREM, REM
        0: "Wake",
        1: "NREM",
        2: "REM"
    },
    4: {  # 4-class: Wake, Light, Deep, REM
        0: "Wake",
        1: "Light",
        2: "Deep",
        3: "REM"
    },
    5: {  # 5-class: Wake, Light, Deep, REM, Unknown
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM"
    }
}


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Output a confusion matrix computed '
                                        'over one or more true/pred .npz '
                                        'files.')
    parser.add_argument("--out-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--true_pattern", type=str,
                        default="--norm",
                        help='Glob-like pattern to one or more .npz files '
                             'storing the true labels')
    parser.add_argument("--pred_pattern", type=str,
                        default="split*/predictions/test_data/dataset_1/files/*/pred.npz",
                        help='Glob-like pattern to one or more .npz files '
                             'storing the true labels')
    parser.add_argument("--normalized", action="store_true",
                        help="Normalize the CM to show fraction of total trues")
    parser.add_argument("--show_pairs", action="store_true",
                        help="Show the paired files (for debugging)")
    parser.add_argument("--group_classes", type=str, default=None,
                        help="Specify how to group classes. A comma-separated list of class mappings"
                             " in format 'source1:target1,source2:target2'")
    parser.add_argument("--round", type=int, default=3,
                        help="Round float numbers, only applicable "
                             "with --normalized.")
    parser.add_argument("--wake_trim_min", type=int, required=False,
                        help="Only evaluate on within wake_trim_min of wake "
                             "before and after sleep, as determined by true "
                             "labels")
    parser.add_argument("--period_length_sec", type=int, default=30,
                        help="Used with --wake_trim_min to determine number of"
                             " periods to trim")
    parser.add_argument("--ignore_classes", type=str, nargs="+", default=None,
                        help="Space-separated list of classes or class:label mappings to ignore.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing log files.')
    parser.add_argument("--log_file", type=str, default="evaluate.log",
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is 'evaluate.log'")
    parser.add_argument("--plot_cm", action="store_true",
                        help="Save the confusion matrix to a file.")
    parser.add_argument("--cmap", type=str, default=DEFAULT_CMAP,
                        help="Matplotlib colormap to use for the confusion matrix.")
    parser.add_argument("--apply_argmax", action="store_true",
                        help="Apply argmax to predictions before calculating metrics. "
                             "Useful when predictions are probability distributions.")
    parser.add_argument("--aggregate_window", type=int, default=None,
                        help="Aggregate predictions over time windows of specified size. "
                             "Applied to predictions only (not true labels). "
                             "If predictions are already argmaxed (apply_argmax not set), uses mode. "
                             "Otherwise uses mean (on probabilities before argmax).")
    parser.add_argument("--per_study_plots", action="store_true",
                        help="Save per-study plots to [out_dir]/<study_id>/: hypnogram.png (with hypnodensity when "
                             "predictions are probabilities) and cm.png.")
    parser.add_argument("--write_eval", action="store_true",
                        help="Write per-study Dice (F1) and class-wise kappa to "
                             "evaluation_dice.csv/.txt and evaluation_kappa.csv/.txt with grand mean.")
    return parser


def _parse_ignore_classes(raw):
    """Parse --ignore_classes values. Each can be 'N' (ignore class N) or 'N:label' (mapping for hypnogram).
    Returns (ignore_classes: List[int] or None, class_mapping: Dict[int, str] or None).
    """
    if raw is None:
        return None

    mapping = {}
    for s in raw:
        if ":" in s:
            parts = s.split(":", 1)
            cls = int(parts[0])
            mapping[cls] = parts[1].strip()
        else:
            mapping[int(s)] = f"Class {s}"
    return mapping


def _get_eval_df_from_ids(study_ids, labels):
    """Build evaluation DataFrame with columns=study_ids, index=mean + labels."""
    return pd.DataFrame(columns=study_ids, index=["mean"] + labels)


def _get_wake_trim_slice(true: np.ndarray, wake_trim_min: int, period_length_sec: int) -> Tuple[int, int]:
    """Return (start, end) slice indices for wake trimming."""
    trim = int((60 / period_length_sec) * wake_trim_min)
    inds = np.where(true != 0)[0]
    start = max(0, inds[0] - trim)
    end = inds[-1] + trim
    return start, end


def wake_trim(true: np.ndarray, pred: np.ndarray, wake_trim_min: int, period_length_sec: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim the true/pred to remove long stretches of 'wake' in either end.
    Trims to a maximum of 'wake_trim_min' of uninterrupted 'wake' in either
    end, determined by the >TRUE< labels.

    args:
        true:             (np.ndarray) True labels
        pred:             (np.ndarray) Predicted labels
        wake_trim_min:    (int)  Maximum number of minutes of uninterrupted wake
                                 sleep stage (integer value '0') to allow
                                 according to TRUE values.
        period_length_sec (int)  The length in seconds of 1 period/epoch/segment

    Returns:
        Tuple of trimmed true and predicted arrays
    """
    start, end = _get_wake_trim_slice(true, wake_trim_min, period_length_sec)
    return true[start:end], pred[start:end]


def _get_true_pred_pairs(true_pattern: str, pred_pattern: str) -> List[Tuple[str, str]]:
    true_files = sorted(glob(true_pattern))
    pred_files = sorted(glob(pred_pattern))
    
    if not true_files:
        raise OSError("Did not find any 'true' files matching "
                      "pattern {}".format(true_pattern))
    if not pred_files:
        raise OSError("Did not find any 'true' files matching "
                      "pattern {}".format(pred_pattern))
    if len(true_files) != len(pred_files):
        raise OSError("Did not find a matching number "
                      "of true and pred files ({} and {})"
                      "".format(len(true_files), len(pred_files)))
    if len(true_files) != len(set(true_files)):
        raise ValueError("Two or more identical file names in the set "
                         "of 'true' files. Cannot uniquely match true/pred "
                         "files")
    if len(pred_files) != len(set(pred_files)):
        raise ValueError("Two or more identical file names in the set "
                         "of 'pred' files. Cannot uniquely match true/pred "
                         "files")
    
    return list(zip(true_files, pred_files))


def _get_numpy_arrays(pairs: List[Tuple[str, str]]) -> Dict[str, List[np.ndarray]]:
    np_pairs = {}
    for true_file, pred_file in pairs:
        study_id = Path(true_file).stem.replace(TRUE_SUFFIX, "")
        np_pairs[study_id] = [np.load(f)["arr_0"] if os.path.splitext(f)[-1] == ".npz" else np.load(f) for f in [true_file, pred_file]]
    return np_pairs


def _aggregate_predictions_time(pred, window_size, use_mode):
    pred = pred.reshape(-1, window_size, pred.shape[-1])
    return stats.mode(pred, axis=1)[0].flatten() if use_mode else np.mean(pred, axis=1)

    
def glob_to_metrics_df(true_pattern: str,
                       pred_pattern: str,
                       out_dir: Path,
                       wake_trim_min: int = None,
                       ignore_class_mapping: dict = None,
                       group_classes: str = None,
                       normalized: bool = False,
                       round: int = 3,
                       period_length_sec: int = 30,
                       show_pairs: bool = False,
                       plot_cm: bool = False,
                       cmap: str = DEFAULT_CMAP,
                       apply_argmax: bool = False,
                       aggregate_window: int = None,
                       per_study_plots: bool = False,
                       write_eval: bool = False):
    """
    Run the script according to the provided arguments.

    Args:
        true_pattern: Glob pattern for true label files
        pred_pattern: Glob pattern for prediction files
        out_dir: Output directory for results
        wake_trim_min: Minutes of wake to trim from start/end
        ignore_class_mapping: Dict mapping class int to label (from class:label in --ignore_classes)
        group_classes: How to group classes. A comma-separated list of class mappings
                       in format 'source1:target1,source2:target2'
        normalized: Whether to normalize confusion matrix
        round: Number of decimals to round to
        period_length_sec: Length of each period in seconds
        show_pairs: Whether to show file pairs
        plot_cm: Whether to plot confusion matrix
        cmap: Matplotlib colormap for confusion matrix
        apply_argmax: Whether to apply argmax to predictions (for probability distributions)
        aggregate_window: Size of time window for aggregating predictions (applied to predictions only)
        per_study_plots: Whether to save per-study hypnogram and confusion matrix plots
        write_eval: Whether to write evaluation_dice/kappa CSV and TXT with per-study and grand mean
    """
    
    logger.info("Looking for files...")
    pairs = _get_true_pred_pairs(true_pattern, pred_pattern)
    
    if show_pairs:
        logger.info("PAIRS:\n{}".format(pairs))

    # Load the pairs
    logger.info("Loading {} pairs...".format(len(pairs)))
    np_pairs = _get_numpy_arrays(pairs)
    all_classes = set()

    for study_id, (true, pred) in np_pairs.items():

        logger.info(f"Processing study {study_id}...")

        # Apply time-wise aggregation to predictions if requested
        if aggregate_window is not None:
            logger.info(f"Aggregating predictions over time windows of size {aggregate_window}...")
            pred = _aggregate_predictions_time(pred, aggregate_window, use_mode=not apply_argmax)
            
            if len(true) != len(pred):
                raise ValueError(f"True labels and aggregated predictions have different lengths: {len(true)} != {len(pred)}")

        # Capture pred_probs before argmax for hypnodensity (only when no group_classes)
        pred_probs = None
        if pred.ndim > 1:
            pred_probs = np.asarray(pred, dtype=np.float64).copy()

        # Apply argmax to predictions if requested
        if apply_argmax:
            logger.info("Applying argmax to predictions...")
            if pred.ndim == 1:
                raise ValueError("Predictions are already argmaxed, but apply_argmax flag is set")

            pred = np.argmax(pred, axis=-1)

        # Trim the true/pred to remove long stretches of 'wake' in either end
        if wake_trim_min:
            logger.info("OBS: Wake trimming of {} minutes (period length {} sec)"
                        "".format(wake_trim_min, period_length_sec))
            true, pred = wake_trim(true, pred, wake_trim_min, period_length_sec)

        if group_classes:
            try:
                group_map = {}
                for pair in group_classes.split(","):
                    source, target = map(int, pair.split(":"))
                    group_map[source] = target
                # First pass: Map everything to temporary values to avoid conflicts
                # Create temp mapping using negative numbers
                temp_map = {source: -(i + 1) for i, (source, _) in enumerate(group_map.items())}
                for source, temp in temp_map.items():
                    true = np.where(true == source, temp, true)
                    pred = np.where(pred == source, temp, pred)
                
                # Second pass: Map to final values
                final_map = {temp_map[source]: target for source, target in group_map.items()}
                for temp, target in final_map.items():
                    true = np.where(true == temp, target, true)
                    pred = np.where(pred == temp, target, pred)
                
                # Update labels with all unique target values from the mapping
                classes = sorted(list(set(group_map.values())))
                logger.info(f"Applied custom class grouping. New classes: {classes}")
                pred_probs = None  # Hypnodensity not supported with group_classes
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid group_classes format. Expected 'source:target,...' (e.g. '2:1,3:1'). Got: {group_classes}") from e


        # Detect unique classes in the data
        true_classes = sorted(list(set(np.unique(true))))
        pred_classes = sorted(list(set(np.unique(pred))))
        logger.info(f"Classes in true labels: {true_classes}")
        logger.info(f"Classes in predictions: {pred_classes}")

        classes = sorted(pred_classes)

        if per_study_plots:
            mapping = MAPPINGS[len(classes)].copy()
            pair_out = out_dir / study_id
            pair_out.mkdir(parents=True, exist_ok=True)
            plot_and_save_hypnogram(
                str(pair_out / "hypnogram.png"),
                pred,
                y_true=true,
                id_=study_id,
                class_mapping=mapping,
                unmapped_class_mapping=ignore_class_mapping,
                y_pred_probs=pred_probs,
            )

        if ignore_class_mapping:

            ignore_classes = list(ignore_class_mapping.keys())

            classes = sorted(list(set(classes) - set(ignore_classes)))
            logger.info(f"Ignoring class(es) {ignore_classes}. Remaining classes: {classes}")
            
            # Also filter the data
            keep_mask = ~(np.isin(true, ignore_classes) | np.isin(pred, ignore_classes))
            true = true[keep_mask]
            pred = pred[keep_mask]

        all_classes.update(classes)
        np_pairs[study_id] = (true, pred)

    all_classes = sorted(all_classes)
    num_classes = len(all_classes)
    out_dir = Path(out_dir)
    mapping = MAPPINGS[num_classes]

    labels = [mapping[i] for i in all_classes]

    # Per-study evaluation and optional plots (Dice, kappa, hypnograms, per-study CMs)
    if write_eval:
        dice_eval_df = _get_eval_df_from_ids(np_pairs.keys(), labels)
        kappa_eval_df = _get_eval_df_from_ids(np_pairs.keys(), labels)
        
        for study_id, (true, pred) in np_pairs.items():
            
            dice_pr_class = f1_score(
                y_true=true,
                y_pred=pred,
                average=None,
                zero_division=1,
            )
            add_to_eval_df(dice_eval_df, study_id, values=dice_pr_class)

            kappa_pr_class = class_wise_kappa(
                true, pred, n_classes=num_classes
            )
            add_to_eval_df(kappa_eval_df, study_id, values=kappa_pr_class)

            if per_study_plots:
                pair_out = out_dir / study_id
                pair_out.mkdir(parents=True, exist_ok=True)
                plot_and_save_cm(
                    str(pair_out / "cm.png"),
                    pred,
                    true,
                    num_classes,
                    id_=study_id,
                    normalized=True,
                    cmap=cmap,
                    class_labels=labels,
                )

        if write_eval:
            dice_eval_df = with_grand_mean_col(dice_eval_df)
            log_eval_df(
                dice_eval_df.T,
                out_csv_file=str(out_dir / "evaluation_dice.csv"),
                out_txt_file=str(out_dir / "evaluation_dice.txt"),
                round=round,
                txt="EVALUATION DICE SCORES",
            )
            kappa_eval_df = with_grand_mean_col(kappa_eval_df)
            log_eval_df(
                kappa_eval_df.T,
                out_csv_file=str(out_dir / "evaluation_kappa.csv"),
                out_txt_file=str(out_dir / "evaluation_kappa.txt"),
                round=round,
                txt="EVALUATION KAPPA SCORES",
            )

    # Concatenate data
    true, pred = map(lambda x: x.astype(np.uint8).reshape(-1, 1), concatenate_true_pred_pairs(pairs=list(np_pairs.values())))

    # Print macro metrics
    global_scores_str = (
        f"Accuracy: {np.round((true == pred).mean(), round)}\n"
        f"Macro F1: {np.round(f1_score(true, pred, average='macro'), round)}\n"
        f"Micro F1: {np.round(f1_score(true, pred, average='micro'), round)}\n"
        f"Kappa:    {np.round(cohen_kappa_score(true, pred), round)}"
    )
    logger.info(f"Unweighted global scores:\n{global_scores_str}")

    cm = confusion_matrix(true, pred)

    # Print stage-wise metrics
    f1 = f1_scores_from_cm(cm)
    prec = precision_scores_from_cm(cm)
    recall = recall_scores_from_cm(cm)
    metrics = pd.DataFrame({
        "F1": f1,
        "Precision": prec,
        "Recall/Sens.": recall
    }, index=labels)
    metrics = metrics.T
    metrics["mean"] = metrics.mean(axis=1)
    metrics_str = str(np.round(metrics.T, round))
    logger.info("Metrics:\n" + metrics_str + "\n")

    if normalized:
        cm = cm.astype(np.float64)
        cm /= cm.sum(axis=1, keepdims=True)

    # Pretty print
    cm = pd.DataFrame(data=cm,
                      index=["True {}".format(label) for label in labels],
                      columns=["Pred {}".format(label) for label in labels])
    p = "raw" if not normalized else "normed"
    logger.info(f"{p} Confusion Matrix:\n" + str(cm.round(round)) + "\n")

    if plot_cm:
        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, f"cm_{p}.png")
        plot_and_save_cm(plot_path, pred, true, len(labels), normalized=normalized, title=global_scores_str, cmap=cmap, class_labels=labels)
        logger.info(f"Saved confusion matrix plot to: {plot_path}")

    return metrics


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")

    if args.aggregate_window is not None and args.aggregate_window <= 1:
        raise ValueError("Aggregate window must be greater than 1")

    ignore_class_mapping = _parse_ignore_classes(args.ignore_classes)
    glob_to_metrics_df(
        true_pattern=args.true_pattern,
        pred_pattern=args.pred_pattern,
        out_dir=args.out_dir,
        wake_trim_min=args.wake_trim_min,
        ignore_class_mapping=ignore_class_mapping,
        group_classes=args.group_classes,
        normalized=args.normalized,
        round=args.round,
        period_length_sec=args.period_length_sec,
        show_pairs=args.show_pairs,
        plot_cm=args.plot_cm,
        cmap=args.cmap,
        apply_argmax=args.apply_argmax,
        aggregate_window=args.aggregate_window,
        per_study_plots=args.per_study_plots,
        write_eval=args.write_eval,
    )


if __name__ == "__main__":
    entry_func()
