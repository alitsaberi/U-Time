"""
Script to compute confusion matrices from one or more pairs of true/pred .npz
files of labels
"""

import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from utime.evaluation import concatenate_true_pred_pairs
from utime.evaluation import (f1_scores_from_cm, precision_scores_from_cm,
                              recall_scores_from_cm)
from utime.evaluation.plotting import plot_and_save_cm
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)

DEFAULT_CMAP = "Blues"


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Output a confusion matrix computed '
                                        'over one or more true/pred .npz '
                                        'files.')
    parser.add_argument("--out-dir", type=Path, default=Path("."))
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
                        help="Specify how to group classes. Options: 'wake_sleep' (groups all sleep stages), "
                             "'non_rem' (groups N1, N2, N3), or a comma-separated list of class mappings "
                             "in format 'source1:target1,source2:target2'")
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
    parser.add_argument("--ignore_classes", type=int, nargs="+", default=None,
                        help="Optional space separated list of class integers to ignore.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing log files.')
    parser.add_argument("--log_file", type=str, default=None,
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is None (no log file)")
    parser.add_argument("--plot-cm", action="store_true",
                        help="Save the confusion matrix to a file.")
    parser.add_argument("--cmap", type=str, default=DEFAULT_CMAP,
                        help="Matplotlib colormap to use for the confusion matrix.")
    return parser


def wake_trim(pairs, wake_trim_min, period_length_sec):
    """
    Trim the pred/true pairs to remove long stretches of 'wake' in either end.
    Trims to a maximum of 'wake_trim_min' of uninterrupted 'wake' in either
    end, determined by the >TRUE< labels.

    args:
        pairs:            (list) A list of (true, prediction) pairs to trim
        wake_trim_min:    (int)  Maximum number of minutes of uninterrupted wake
                                 sleep stage (integer value '0') to allow
                                 according to TRUE values.
        period_length_sec (int)  The length in seconds of 1 period/epoch/segment

    Returns:
        List of trimmed (true, prediction) pairs
    """
    trim = int((60/period_length_sec) * wake_trim_min)
    trimmed_pairs = []
    for true, pred in pairs:
        inds = np.where(true != 0)[0]
        start = max(0, inds[0]-trim)
        end = inds[-1]+trim
        trimmed_pairs.append([
            true[start:end], pred[start:end]
        ])
    return trimmed_pairs


def trim(p1, p2):
    """
    Trims a pair of label arrays (true/pred normally) to equal length by
    removing elements from the tail of the longest array.
    This assumes that the arrays are aligned to the first element.
    """
    diff = len(p1) - len(p2)
    if diff > 0:
        p1 = p1[:len(p2)]
    else:
        p2 = p2[:len(p1)]
    return p1, p2


def glob_to_metrics_df(true_pattern: str,
                       pred_pattern: str,
                       out_dir: Path,
                       wake_trim_min: int = None,
                       ignore_classes: list = None,
                       group_classes: str = None,
                       normalized: bool = False,
                       round: int = 3,
                       period_length_sec: int = 30,
                       show_pairs: bool = False,
                       plot_cm: bool = False,
                       cmap: str = DEFAULT_CMAP):
    """
    Run the script according to the provided arguments.

    Args:
        true_pattern: Glob pattern for true label files
        pred_pattern: Glob pattern for prediction files
        out_dir: Output directory for results
        wake_trim_min: Minutes of wake to trim from start/end
        ignore_classes: List of class indices to ignore
        group_classes: How to group classes. Options:
            - 'wake_sleep': Groups all sleep stages into one
            - 'non_rem': Groups N1, N2, N3 into one
            - 'source:target,...': Custom grouping (e.g. '2:1,3:1')
        normalized: Whether to normalize confusion matrix
        round: Number of decimals to round to
        period_length_sec: Length of each period in seconds
        show_pairs: Whether to show file pairs
        plot_cm: Whether to plot confusion matrix
        cmap: Matplotlib colormap for confusion matrix
    """
    logger.info("Looking for files...")
    true = sorted(glob(true_pattern))
    pred = sorted(glob(pred_pattern))
    if not true:
        raise OSError("Did not find any 'true' files matching "
                      "pattern {}".format(true_pattern))
    if not pred:
        raise OSError("Did not find any 'true' files matching "
                      "pattern {}".format(pred_pattern))
    if len(true) != len(pred):
        raise OSError("Did not find a matching number "
                      "of true and pred files ({} and {})"
                      "".format(len(true), len(pred)))
    if len(true) != len(set(true)):
        raise ValueError("Two or more identical file names in the set "
                         "of 'true' files. Cannot uniquely match true/pred "
                         "files")
    if len(pred) != len(set(pred)):
        raise ValueError("Two or more identical file names in the set "
                         "of 'pred' files. Cannot uniquely match true/pred "
                         "files")
    
    pairs = list(zip(true, pred))
    if show_pairs:
        logger.info("PAIRS:\n{}".format(pairs))
    # Load the pairs
    logger.info("Loading {} pairs...".format(len(pairs)))
    def load_array(files):
        return [np.load(f)["arr_0"] if os.path.splitext(f)[-1] == ".npz" else np.load(f) for f in files]
    np_pairs = list(map(load_array, pairs))
    for i, (p1, p2) in enumerate(np_pairs):
        if len(p1) != len(p2):
            logger.warning(f"Not equal lengths: {pairs[i]} {f'{len(p1)}/{len(p2)}'}. Trimming...")
            np_pairs[i] = trim(p1, p2)
    if wake_trim_min:
        logger.info("OBS: Wake trimming of {} minutes (period length {} sec)"
                    "".format(wake_trim_min, period_length_sec))
        np_pairs = wake_trim(np_pairs, wake_trim_min, period_length_sec)
    # Load and concatenate data first
    true, pred = map(lambda x: x.astype(np.uint8).reshape(-1, 1), concatenate_true_pred_pairs(pairs=np_pairs))
    
    # Detect unique classes in the data
    true_classes = sorted(list(set(np.unique(true))))
    pred_classes = sorted(list(set(np.unique(pred))))
    logger.info(f"Classes in true labels: {true_classes}")
    logger.info(f"Classes in predictions: {pred_classes}")
    
    # Get all unique classes and handle ignore_classes first
    all_classes = sorted(pred_classes)
    if ignore_classes:
        all_classes = sorted(list(set(all_classes) - set(ignore_classes)))
        logger.info(f"Ignoring class(es) {ignore_classes}. Remaining classes: {all_classes}")
        
        # Also filter the data
        keep_mask = ~np.isin(true, ignore_classes)
        true = true[keep_mask]
        pred = pred[keep_mask]
    
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
        }
    }
    
    # Determine number of classes from the filtered data
    num_classes = len(all_classes)
    
    # Create mapping based on number of classes
    if num_classes in MAPPINGS:
        # Map the actual class values to the standard mapping
        # Sort classes to ensure consistent mapping (lowest value = Wake, etc)
        sorted_classes = sorted(all_classes)
        mapping = {actual: MAPPINGS[num_classes][expected] 
                  for expected, actual in enumerate(sorted_classes)}
        logger.info(f"Using {num_classes}-class mapping: {mapping}")
    else:
        logger.warning(f"Unexpected number of classes ({num_classes}). "
                      f"Expected one of: {list(MAPPINGS.keys())}. "
                      f"Using generic class names.")
        mapping = {i: f"Class_{i}" for i in all_classes}
    
    labels = all_classes  # Keep original class order

    if group_classes:
        if group_classes == "wake_sleep":
            # Group all sleep stages (1-4) into one class (1)
            ones = np.ones_like(true)
            true = np.where(np.isin(true, [1, 2, 3, 4]), ones, true)
            pred = np.where(np.isin(pred, [1, 2, 3, 4]), ones, pred)
            for i in [2, 3, 4]:
                if i in labels:
                    labels.remove(i)
            mapping[1] = "Sleep"
            for i in [2, 3, 4]:
                if i in mapping:
                    del mapping[i]
            logger.info(f"Grouping into Wake/Sleep. New labels: {labels} / {[mapping[i] for i in labels]}")
        elif group_classes == "non_rem":
            # Group all NREM stages (1-3) into one class (1)
            ones = np.ones_like(true)
            true = np.where(np.isin(true, [1, 2, 3]), ones, true)
            pred = np.where(np.isin(pred, [1, 2, 3]), ones, pred)
            for i in [2, 3]:
                if i in labels:
                    labels.remove(i)
            mapping[1] = "NREM"
            for i in [2, 3]:
                if i in mapping:
                    del mapping[i]
            logger.info(f"Grouping all NREM stages into one. New labels: {labels} / {[mapping[i] for i in labels]}")
        else:
            # Custom grouping using source:target mappings
            try:
                group_map = {}
                for pair in group_classes.split(","):
                    source, target = map(int, pair.split(":"))
                    group_map[source] = target
                for source, target in group_map.items():
                    true = np.where(true == source, target, true)
                    pred = np.where(pred == source, target, pred)
                    if source in labels:
                        labels.remove(source)
                    if source in mapping:
                        del mapping[source]
                    if target not in labels:
                        labels.append(target)
                labels = sorted(list(set(labels)))
                logger.info(f"Applied custom class grouping. New labels: {labels} / {[mapping[i] for i in labels]}")
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid group_classes format. Expected 'wake_sleep', 'non_rem' or "
                               f"'source:target,...' (e.g. '2:1,3:1'). Got: {group_classes}") from e

    # Print macro metrics
    keep_mask = np.where(np.isin(true, labels))
    global_scores_str = (
        f"Accuracy: {np.round((true[keep_mask] == pred[keep_mask]).mean(), round)}\n"
        f"Macro F1: {np.round(f1_score(true[keep_mask], pred[keep_mask], average='macro'), round)}\n"
        f"Micro F1: {np.round(f1_score(true[keep_mask], pred[keep_mask], average='micro'), round)}\n"
        f"Kappa:    {np.round(cohen_kappa_score(true[keep_mask], pred[keep_mask]), round)}"
    )
    logger.info(f"Unweighted global scores:\n{global_scores_str}")

    cm = confusion_matrix(true, pred, labels=labels)
    if normalized:
        cm = cm.astype(np.float64)
        cm /= cm.sum(axis=1, keepdims=True)

    # Pretty print
    cm = pd.DataFrame(data=cm,
                      index=["True {}".format(mapping[i]) for i in labels],
                      columns=["Pred {}".format(mapping[i]) for i in labels])
    p = "Raw" if not normalized else "Normed"
    logger.info(f"\n\n{p} Confusion Matrix:\n" + str(cm.round(round)) + "\n")

    # Print stage-wise metrics
    f1 = f1_scores_from_cm(cm)
    prec = precision_scores_from_cm(cm)
    recall = recall_scores_from_cm(cm)
    metrics = pd.DataFrame({
        "F1": f1,
        "Precision": prec,
        "Recall/Sens.": recall
    }, index=[mapping[i] for i in labels])
    metrics = metrics.T
    metrics["mean"] = metrics.mean(axis=1)
    metrics_str = str(np.round(metrics.T, round))
    logger.info(f"\n\n{p} Metrics:\n" + metrics_str + "\n")

    if plot_cm:
        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, f"cm_{p}.png")
        # Get labels in correct order for plotting
        label_names = [mapping[i] for i in labels]
        plot_and_save_cm(plot_path, pred, true, len(labels), title=global_scores_str, cmap=cmap, class_labels=label_names)
        logger.info(f"Saved confusion matrix plot to: {plot_path}")

    return metrics


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    glob_to_metrics_df(
        true_pattern=args.true_pattern,
        pred_pattern=args.pred_pattern,
        out_dir=args.out_dir,
        wake_trim_min=args.wake_trim_min,
        ignore_classes=args.ignore_classes,
        group_classes=args.group_classes,
        normalized=args.normalized,
        round=args.round,
        period_length_sec=args.period_length_sec,
        show_pairs=args.show_pairs,
        plot_cm=args.plot_cm,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    entry_func()
