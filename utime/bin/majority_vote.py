import logging
import os
import numpy as np
from argparse import ArgumentParser
from glob import glob
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Majority vote across a set of channels.')
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help='Path to folder storing predictions for each dataset. '
                             'The specified folder must store sub-folders for each given dataset. '
                             'Each dataset folder must store results from each channel combination each '
                             'in a sub-folder named according to the channel combination.')
    parser.add_argument("--soft", action="store_true",
                        help="If using NxC shaped probability-like arrays, use this option to sum arrays "
                             "instead of computing mode.")
    parser.add_argument("--out_folder_name", type=str, default="majority",
                        help='Name of the output folder to store majority vote results. '
                             'Default is "majority".')
    parser.add_argument("--trim_to_min_length", action="store_true",
                        help="When set, trim arrays to minimum length if they have length mismatches. "
                             "Arrays will be trimmed to match the shortest array's length.")
    parser.add_argument("--element-wise-weights", action="store_true",
                        help="Apply element-wise weights from _WEIGHT.npy files when using --soft flag. "
                             "Weight files should be named {study_id}_WEIGHT.npy and located alongside "
                             "prediction files. If weight files are missing, falls back to unweighted averaging.")
    parser.add_argument("--channel_combinations", type=str, nargs="+", default=None,
                        help="Optional list of channel combination folder names to use. "
                             "Only predictions from these combinations will be included in majority vote. "
                             "If not specified, all available combinations will be used.")
    parser.add_argument("--combination_weights", type=float, nargs="+", default=None,
                        help="Optional list of weights for channel combinations. "
                             "Weights should be provided in the same order as --channel_combinations. "
                             "Weights are normalized automatically. If not specified, all combinations are weighted equally.")
    parser.add_argument("--softmax-fusion", action="store_true",
                        help="Use softmax fusion based on maximum confidences of each combination. "
                             "Computes max confidence for each combination and applies softmax with temperature "
                             "to get weights. Requires --soft flag. Cannot be used with --combination_weights or --element-wise-weights.")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature parameter for softmax fusion. Must be > 0. "
                             "Higher temperature = more uniform weights, lower = winner-take-all. "
                             "Typical range: 0.1-10, default is 2.0. Only used with --softmax-fusion.")
    parser.add_argument(
        "--n_classes",
        type=int,
        default=None,
        help="Number of in-range classes. Used for tie-breaking preference of labels in range "
             "[0, n_classes). Required for tie strategies that prefer in-range labels."
    )
    parser.add_argument(
        "--tie_strategy",
        type=str,
        default="prefer_in_range_random",
        choices=("prefer_in_range_random", "prefer_in_range_unknown", "unknown"),
        help="How to resolve ties during hard majority voting. "
             "'prefer_in_range_random': prefer labels in [0, n_classes), else pick randomly among tied. "
             "'prefer_in_range_unknown': prefer labels in [0, n_classes), else set unknown. "
             "'unknown': always set unknown on ties."
    )
    parser.add_argument(
        "--unknown_label",
        type=int,
        default=None,
        help="Label value to use when tie_strategy requires setting unknown."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when tie_strategy picks randomly among tied labels."
    )
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing output files and log files.')
    parser.add_argument("--log_file", type=str, default=None,
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is None (no log file)")
    return parser


def get_datasets(folder):
    """
    Returns a dictionary of dataset-ID: dataset directory paths
    """
    paths = glob(f"{folder}/*")
    return {os.path.split(p)[-1]: p for p in paths if os.path.isdir(p)}


def get_true_paths(dataset_dir):
    """
    Returns a dictionary of study-ID: true/target vector paths
    """
    # Support both:
    # - old layout: dataset_dir/<study_id>_TRUE.npy
    # - channel-combo layout: dataset_dir/<combo>/<study_id>_TRUE.npy
    # - split layout: dataset_dir/<split>/<combo>/<study_id>_TRUE.npy (when called on split_dir)
    true_paths = glob(os.path.join(dataset_dir, "*TRUE.np*"))
    true_paths += glob(os.path.join(dataset_dir, "*", "*TRUE.np*"))
    out = {}
    for p in true_paths:
        sid = os.path.split(p)[-1].split("_TRUE")[0]
        # Keep first seen (stable enough) – all TRUE vectors should be identical across combos
        out.setdefault(sid, p)
    return out


def get_prediction_paths(pred_dir):
    """
    Returns a dictionary of study-ID: predicted vector paths
    """
    pred_paths = glob(f'{pred_dir}/*PRED.np*')
    return {
        os.path.split(p)[-1].split("_PRED")[0]: p for p in pred_paths
    }


def extract_combination_name(path, dataset_dir):
    """
    Extract the channel combination name from a prediction file path.
    
    Args:
        path: Full path to prediction file
        dataset_dir: Base dataset directory path
        
    Returns:
        Combination name (folder name) or None if extraction fails
    """
    parts = path.split(os.sep)
    if len(parts) >= 2:
        # Path structure: .../dataset_dir/combination_name/study_id_PRED.npy
        # Find the part after dataset_dir
        try:
            dataset_idx = parts.index(os.path.basename(os.path.normpath(dataset_dir)))
            if dataset_idx + 1 < len(parts):
                return parts[dataset_idx + 1]
        except ValueError:
            # If dataset_dir basename not found, try using second-to-last part
            return parts[-2]
    return None


def get_input_channel_combinations(dataset_dir,
                                   study_id,
                                   allowed_combinations=None,
                                   combination_weights=None,
                                   exclude_combinations=None):

    # Find all prediction files for this study
    all_paths = glob(f'{dataset_dir}/*/*{study_id}_PRED.np*')
    exclude_combinations = set(exclude_combinations or [])
    
    if allowed_combinations is None:
        combination_names = []
        filtered_paths = []
        for path in all_paths:
            combination_name = extract_combination_name(path, dataset_dir)
            if combination_name in exclude_combinations:
                continue
            combination_names.append(combination_name)
            filtered_paths.append(path)
        return combination_names, filtered_paths, [1] * len(filtered_paths)
    
    # Filter by allowed combinations
    # Extract combination name from path: dataset_dir/combination_name/study_id_PRED.npy
    filtered_combination_names = []
    filtered_paths = []
    filtered_weights = []
    for path in all_paths:
        # Get the directory name (combination name) from the path
        combination_name = extract_combination_name(path, dataset_dir)
        if combination_name in exclude_combinations:
            continue

        if combination_name not in allowed_combinations:
            logger.debug(f"Skipping combination '{combination_name}' (not in allowed list)")
            continue

        filtered_combination_names.append(combination_name)
        filtered_paths.append(path)
        filtered_weights.append(combination_weights[allowed_combinations.index(combination_name)] if combination_weights is not None else 1)
    
    return filtered_combination_names, filtered_paths, filtered_weights if combination_weights is not None else [1] * len(filtered_paths)


def get_weight_path(pred_path):
    return pred_path.replace("_PRED.npy", "_WEIGHT.npy").replace("_PRED.npz", "_WEIGHT.npy")

def get_arrays(paths, trim_to_min_length=False):
    loaded = []
    element_wise_weights = []

    for i, arr_path in enumerate(paths):
        array = np.load(arr_path)
        loaded.append(array)

        weight_path = get_weight_path(arr_path)
        if os.path.exists(weight_path):
            weight_array = np.load(weight_path)
            element_wise_weights.append(weight_array)
        else:
            element_wise_weights.append(np.ones(array.shape[0], dtype=float))

    # Handle length mismatches if requested
    lengths = [a.shape[0] for a in loaded]
    min_len = min(lengths) if lengths else 0
    if any(length != min_len for length in lengths):
        msg = f"Length mismatch across arrays: {lengths}"
        if not trim_to_min_length:
            raise ValueError(msg + " (set --trim_to_min_length to allow trimming)")
        logger.warning(msg + f" - trimming all to min length {min_len}")
        loaded = [a[:min_len] for a in loaded]
        element_wise_weights = [w[:min_len] for w in element_wise_weights]

    # Validate weights lengths now that optional trimming applied
    for i, (arr, w) in enumerate(zip(loaded, element_wise_weights)):
        if w.shape[0] != arr.shape[0]:
            raise ValueError(
                f"Weight array length {w.shape[0]} doesn't match prediction length {arr.shape[0]} "
                f"for channel index {i}."
            )

    return np.stack(loaded), np.stack(element_wise_weights)


def get_weight_arrays(pred_paths):
    """
    Load weight arrays corresponding to prediction file paths.
    Weight files should be named by replacing _PRED.npy with _WEIGHT.npy
    
    Args:
        pred_paths: List of prediction file paths
        
    Returns:
        List of weight arrays (or None for missing files) and boolean indicating if any weights were found
    """
    weight_arrays = []
    weights_found = False
    
    for pred_path in pred_paths:
        # Replace _PRED.npy or _PRED.npz with _WEIGHT.npy
        weight_path = pred_path.replace("_PRED.npy", "_WEIGHT.npy").replace("_PRED.npz", "_WEIGHT.npy")
        
        if not os.path.exists(weight_path):
            weight_arrays.append(None)
            continue

        weight_arr = np.load(weight_path)
        weight_arrays.append(weight_arr)
        weights_found = True
    
    return weight_arrays, weights_found


def get_softmax_weights(confidences, temperature=2.0):

    max_confs = np.max(confidences, axis=2)
    
    # Apply softmax with temperature
    # Compute exp(max_confs / temperature) and normalize
    exp_confs = np.exp(max_confs / temperature)
    weights = exp_confs / np.sum(exp_confs)
    
    return weights


def apply_weights(channel_arrs, weights):
    """
    Apply weights to channel arrays.
    
    Args:
        channel_arrs: Array of shape (n_channels, n_samples, ...) or similar
        weights: List/array of weights. Each weight can be:
            - A scalar (number): multiplied to entire channel_arr
            - A 1D array: applied row-wise to first dimension of channel_arr
            
    Returns:
        Stacked weighted arrays
    """
    weighted_channel_arrs = []

    for channel_arr, weight in zip(channel_arrs, weights):
        if isinstance(weight, np.ndarray):
            weight = weight.reshape(-1, 1)

        weighted_channel_arrs.append(channel_arr * weight)

    return np.stack(weighted_channel_arrs, axis=0)


def _is_soft_prediction_array(arr: np.ndarray) -> bool:
    """
    Heuristic: soft predictions are (N, C) float-like arrays.
    """
    return isinstance(arr, np.ndarray) and arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating)


def _to_hard_labels(arr: np.ndarray) -> np.ndarray:
    """
    Convert either hard labels (N,) or soft (N, C) to hard labels (N,).
    """
    if _is_soft_prediction_array(arr):
        return arr.argmax(axis=-1)
    if arr.ndim != 1:
        raise ValueError(f"Expected hard labels to be 1D, got shape {arr.shape}")
    return arr


def _resolve_tie(tied_labels: np.ndarray,
                 args,
                 rng: np.random.Generator) -> int:
    """
    Resolve tie among tied_labels according to args.tie_strategy.
    """
    tied_labels = np.asarray(tied_labels)
    if tied_labels.size == 0:
        raise ValueError("No tied labels to resolve.")
    if tied_labels.size == 1:
        return int(tied_labels[0])

    if args.tie_strategy == "unknown":
        return int(args.unknown_label)

    # prefer_in_range_* strategies
    if args.n_classes is None:
        in_range = tied_labels
    else:
        in_range = tied_labels[(tied_labels >= 0) & (tied_labels < args.n_classes)]
    if in_range.size == 1:
        return int(in_range[0])
    if in_range.size > 1:
        if args.tie_strategy == "prefer_in_range_random":
            return int(rng.choice(in_range))
        # prefer_in_range_unknown
        return int(args.unknown_label)

    # No in-range labels among tied
    if args.tie_strategy == "prefer_in_range_random":
        return int(rng.choice(tied_labels))
    return int(args.unknown_label)


def hard_majority_vote(label_matrix: np.ndarray, args, rng: np.random.Generator) -> np.ndarray:
    """
    Hard majority vote over a stack of hard-label vectors.

    Args:
        label_matrix: shape (n_votes, n_samples)
    Returns:
        majority labels: shape (n_samples,)
    """
    if label_matrix.ndim != 2:
        raise ValueError(f"Expected label_matrix with shape (n_votes, n_samples), got {label_matrix.shape}")
    n_votes, n_samples = label_matrix.shape
    if n_votes == 0:
        raise ValueError("No votes provided.")

    out = np.empty((n_samples,), dtype=label_matrix.dtype)
    for i in range(n_samples):
        col = label_matrix[:, i]
        labels, counts = np.unique(col, return_counts=True)
        max_count = counts.max()
        tied = labels[counts == max_count]
        out[i] = _resolve_tie(tied, args=args, rng=rng)
    return out


def _fuse_stacked_predictions(stacked_arrs, args, rng: np.random.Generator,
                              combination_weights=None, element_wise_weights=None):
    """
    Fuse a stack of predictions (channels or splits) with soft or hard fusion.

    Args:
        stacked_arrs: (n_sources, N, C) for soft or (n_sources, N) for hard
        args: script args (soft, softmax_fusion, temperature, tie_strategy, etc.)
        rng: random generator for tie-breaking
        combination_weights: optional list of n_sources scalars; default equal
        element_wise_weights: optional (n_sources, N) or list of n_sources (N,) arrays; default ones

    Returns:
        (N, C) for soft or (N,) for hard
    """
    n = stacked_arrs.shape[0]
    if args.soft:
        if stacked_arrs.ndim == 2:
            return hard_majority_vote(stacked_arrs.astype(np.int64), args=args, rng=rng)
        if args.softmax_fusion:
            softmax_weights = get_softmax_weights(stacked_arrs, temperature=args.temperature)
            stacked_arrs = apply_weights(stacked_arrs, softmax_weights)
        combination_weights = combination_weights if combination_weights is not None else [1.0] * n
        if element_wise_weights is None:
            element_wise_weights = [np.ones(stacked_arrs.shape[1], dtype=float)] * n
        stacked_arrs = apply_weights(stacked_arrs, combination_weights)
        stacked_arrs = apply_weights(stacked_arrs, element_wise_weights)
        mj = np.sum(stacked_arrs, axis=0)
        mj = mj / np.sum(mj, axis=1, keepdims=True)
        return mj
    hard = np.array([_to_hard_labels(a) for a in stacked_arrs])
    return hard_majority_vote(hard, args=args, rng=rng)


def _detect_split_layout(dataset_dir_path: str) -> bool:
    """
    Detect whether dataset_dir_path contains suffix split folders.
    Old layout:  dataset/<combination>/<study>_PRED.npy
    Split layout: dataset/<split>/<combination>/<study>_PRED.npy (and dataset/<split>/<study>_TRUE.npy)
    """
    old = glob(os.path.join(dataset_dir_path, "*", "*_PRED.np*"))
    split = glob(os.path.join(dataset_dir_path, "*", "*", "*_PRED.np*"))
    # If both exist, prefer split layout if split-depth has more signal
    return (len(split) > 0) and (len(old) == 0 or len(split) >= len(old))


def _iter_split_dirs(dataset_dir_path: str):
    for p in glob(os.path.join(dataset_dir_path, "*")):
        if os.path.isdir(p):
            yield p


def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _save_if_allowed(path_no_ext: str, arr: np.ndarray, overwrite: bool):
    """
    Save using numpy semantics, where np.save will append .npy if not present.
    """
    # Match existing behavior: majority_vote previously used no extension in out_path.
    # We'll check for either <path> or <path>.npy existing.
    existing = os.path.exists(path_no_ext) or os.path.exists(path_no_ext + ".npy")
    if existing and not overwrite:
        logger.warning(f"Output file at {path_no_ext}(.npy) exists and --overwrite not set. Skipping.")
        return False
    np.save(path_no_ext, arr)
    return True


def _load_true_if_exists(true_paths_map: dict, study_id: str):
    p = true_paths_map.get(study_id)
    if not p:
        return None
    return np.load(p)


def _process_one_folder_as_dataset(dataset_dir_path: str, out_dir: str, args, rng: np.random.Generator):
    """
    Process a dataset folder in the old layout (no suffix split dirs).
    """
    _ensure_dir(out_dir)
    # Avoid including already computed majority results as "inputs"
    excluded_dirs = {args.out_folder_name, "majority"}
    pred_paths = []
    for p in glob(os.path.join(dataset_dir_path, "*", "*PRED.np*")):
        combo = os.path.basename(os.path.dirname(p))
        if combo in excluded_dirs:
            continue
        pred_paths.append(p)
    study_ids = set([os.path.split(s)[-1].split("_PRED")[0] for s in pred_paths])
    logger.info(f"Found {len(study_ids)} paths to study IDs")
    true_paths = get_true_paths(dataset_dir_path)

    for study_id in study_ids:
        logger.info(f"Processing study {study_id}")

        out_path = os.path.join(out_dir, f"{study_id}_PRED")
        if os.path.exists(out_path) and not args.overwrite:
            logger.warning(f"Output file at {out_path} exists and the --overwrite flag was not set. Skipping.")
            continue

        combination_names, channels, combination_weights = get_input_channel_combinations(
            dataset_dir_path,
            study_id,
            allowed_combinations=args.channel_combinations,
            combination_weights=args.combination_weights,
            exclude_combinations=excluded_dirs,
        )
        logger.info(f"Using {len(channels)} combinations ({combination_names}) for study {study_id} with weights {combination_weights}")

        if len(combination_names) == 0:
            logger.warning(f"No channel combinations found for study {study_id}. Skipping.")
            continue

        channel_arrs, element_wise_weights = get_arrays(channels, trim_to_min_length=args.trim_to_min_length)

        mj = _fuse_stacked_predictions(
            channel_arrs, args, rng,
            combination_weights=combination_weights,
            element_wise_weights=element_wise_weights,
        )

        logger.info(f"Saving majority with shape {mj.shape} to {out_path}")
        np.save(out_path, mj)

        # Also save true alongside majority if available
        true = _load_true_if_exists(true_paths, study_id)
        if true is not None:
            true_out = os.path.join(out_dir, f"{study_id}_TRUE")
            if args.trim_to_min_length and mj.shape[0] != true.shape[0]:
                min_len = min(mj.shape[0], true.shape[0])
                true = true[:min_len]
            _save_if_allowed(true_out, true, overwrite=args.overwrite)


def _process_split_dataset(dataset_dir_path: str, args, rng: np.random.Generator):
    """
    Process a dataset folder in the split layout:
      dataset/<split>/<combination>/<study>_PRED.npy
      dataset/<split>/<study>_TRUE.npy

    Each split is processed like a one-folder dataset (shared code path).
    Produces:
      - per-split majority: dataset/<split>/<out_folder_name>/<study>_{PRED,TRUE}.npy
      - overall across splits: dataset/<out_folder_name>/<study>_{PRED,TRUE}.npy
    """
    split_dirs = [d for d in _iter_split_dirs(dataset_dir_path) if os.path.isdir(d)]
    logger.info(f"Detected split layout with {len(split_dirs)} split folders")

    # Per-split majority: treat each split as a one-folder dataset (shared code)
    all_study_ids = set()
    split_true_maps = {}
    for split_dir in split_dirs:
        split_name = os.path.basename(os.path.normpath(split_dir))
        logger.info(f"Processing split '{split_name}'")
        out_dir_split = os.path.join(split_dir, args.out_folder_name)
        _process_one_folder_as_dataset(
            dataset_dir_path=split_dir,
            out_dir=out_dir_split,
            args=args,
            rng=rng,
        )
        # Collect study IDs and true paths for overall fusion
        pred_paths = glob(os.path.join(out_dir_split, "*_PRED.np*"))
        study_ids = set([os.path.split(s)[-1].split("_PRED")[0] for s in pred_paths])
        all_study_ids |= study_ids
        split_true_maps[split_dir] = get_true_paths(split_dir)

    # Overall across splits (same fusion as per-split: soft fusion or hard majority)
    out_dir_overall = os.path.join(dataset_dir_path, args.out_folder_name)
    _ensure_dir(out_dir_overall)

    for study_id in sorted(all_study_ids):
        # Collect per-split majority predictions (keep raw for soft fusion)
        split_preds = []
        split_trues = []

        for split_dir in split_dirs:
            split_out_pred = os.path.join(split_dir, args.out_folder_name, f"{study_id}_PRED.npy")
            if not os.path.exists(split_out_pred):
                continue
            arr = np.load(split_out_pred)
            split_preds.append(arr)

            true_paths = split_true_maps.get(split_dir) or get_true_paths(split_dir)
            tpath = true_paths.get(study_id)
            if tpath and os.path.exists(tpath):
                split_trues.append(_to_hard_labels(np.load(tpath)))

        if len(split_preds) == 0:
            continue

        # Align lengths (first dimension)
        lengths = [a.shape[0] for a in split_preds]
        min_len = min(lengths)
        if any(length != min_len for length in lengths):
            msg = f"Split majority prediction length mismatch for study {study_id}: {lengths}"
            if not args.trim_to_min_length:
                raise ValueError(msg + " (set --trim_to_min_length to allow trimming)")
            logger.warning(msg + f" - trimming to {min_len}")
            split_preds = [a[:min_len] for a in split_preds]

        stacked = np.stack(split_preds, axis=0)
        overall_pred = _fuse_stacked_predictions(stacked, args, rng)

        out_path_pred = os.path.join(out_dir_overall, f"{study_id}_PRED")
        logger.info(f"Saving overall majority (across splits) for {study_id} with shape {overall_pred.shape} to {out_path_pred}")
        _save_if_allowed(out_path_pred, overall_pred, overwrite=args.overwrite)

        # Overall true: hard majority across split trues (if available)
        if len(split_trues) > 0:
            lengths_t = [a.shape[0] for a in split_trues]
            min_len_t = min(lengths_t)
            if any(length != min_len_t for length in lengths_t):
                msg = f"Split TRUE length mismatch for study {study_id}: {lengths_t}"
                if not args.trim_to_min_length:
                    raise ValueError(msg + " (set --trim_to_min_length to allow trimming)")
                logger.warning(msg + f" - trimming to {min_len_t}")
                split_trues = [a[:min_len_t] for a in split_trues]
            true_stack = np.stack(split_trues, axis=0)
            overall_true = hard_majority_vote(true_stack, args=args, rng=rng)
            out_path_true = os.path.join(out_dir_overall, f"{study_id}_TRUE")
            _save_if_allowed(out_path_true, overall_true, overwrite=args.overwrite)


def run(args):
    rng = np.random.default_rng(args.seed)
    dataset_dirs = get_datasets(folder=args.dataset_dir)

    for dataset, dataset_dir_path in dataset_dirs.items():
        logger.info(f"Processing dataset '{dataset}'")

        if _detect_split_layout(dataset_dir_path):
            _process_split_dataset(dataset_dir_path=dataset_dir_path, args=args, rng=rng)
        else:
            out_dir = os.path.join(dataset_dir_path, args.out_folder_name)
            _process_one_folder_as_dataset(
                dataset_dir_path=dataset_dir_path,
                out_dir=out_dir,
                args=args,
                rng=rng
            )


def assert_args(args):
    if args.element_wise_weights and not args.soft:
        raise ValueError("--element-wise-weights can only be used with --soft flag. "
                        "Weights are not applicable to hard voting (mode).")
    
    if args.combination_weights:
        if not args.soft:
            raise ValueError("--combination-weights can only be used with --soft flag. "
                            "Weights are not applicable to hard voting (mode).")

        if not args.channel_combinations:
            raise ValueError("--channel_combinations must be specified when using --combination_weights.")

        if len(args.channel_combinations) != len(args.combination_weights):
            raise ValueError("--channel_combinations and --combination_weights must have the same length. "
                            f"Got {len(args.channel_combinations)} and {len(args.combination_weights)} respectively.")
    
    if args.softmax_fusion and not args.soft:
        raise ValueError("--softmax-fusion can only be used with --soft flag. "
                        "Fusion weights are not applicable to hard voting (mode).")
    
    if args.softmax_fusion and args.temperature <= 0:
        raise ValueError(f"--temperature must be > 0, got {args.temperature}. "
                        "Temperature controls the softmax distribution: "
                        "lower values favor winner-take-all, higher values favor uniform weights.")

    if args.tie_strategy in ("prefer_in_range_unknown", "unknown") and args.unknown_label is None:
        raise ValueError("--unknown_label is required when tie_strategy sets unknown on ties.")


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    assert_args(args)
    run(args)

if __name__ == "__main__":
    entry_func()
