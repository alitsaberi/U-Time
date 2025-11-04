import logging
import os
import numpy as np
from argparse import ArgumentParser
from scipy import stats
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
    true_paths = glob(f'{dataset_dir}/*TRUE.np*')
    return {
        os.path.split(p)[-1].split("_TRUE")[0]: p for p in true_paths
    }


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


def get_input_channel_combinations(dataset_dir, study_id, allowed_combinations=None, combination_weights=None):

    # Find all prediction files for this study
    all_paths = glob(f'{dataset_dir}/*/*{study_id}_PRED.np*')
    
    if allowed_combinations is None:
        combination_names = [extract_combination_name(path, dataset_dir) for path in all_paths]
        return combination_names, all_paths, [1] * len(combination_names)
    
    # Filter by allowed combinations
    # Extract combination name from path: dataset_dir/combination_name/study_id_PRED.npy
    filtered_combination_names = []
    filtered_paths = []
    filtered_weights = []
    for path in all_paths:
        # Get the directory name (combination name) from the path
        combination_name = extract_combination_name(path, dataset_dir)

        if combination_name not in allowed_combinations:
            logger.debug(f"Skipping combination '{combination_name}' (not in allowed list)")
            continue

        filtered_combination_names.append(combination_name)
        filtered_paths.append(path)
        filtered_weights.append(combination_weights[allowed_combinations.index(combination_name)] if combination_weights is not None else 1)
    
    return filtered_combination_names, filtered_paths, filtered_weights if combination_weights is not None else [1] * len(filtered_paths)


def get_weight_path(pred_path):
    return pred_path.replace("_PRED.npy", "_WEIGHT.npy").replace("_PRED.npz", "_WEIGHT.npy")

def get_arrays(paths):
    loaded = []
    element_wise_weights = []
    for i, arr_path in enumerate(paths):
        array = np.load(arr_path)
        loaded.append(array)

        weight_path = get_weight_path(arr_path)

        if os.path.exists(weight_path):
            weight_array = np.load(weight_path)

            if weight_array.shape[0] != array.shape[0]:
                raise ValueError(f"Weight array length {weight_array.shape[0]} doesn't match "
                                f"prediction length {array.shape[0]} for channel {i}. ")

            element_wise_weights.append(weight_array)
            print(f"element_wise_weights: {element_wise_weights}")
        else:
            element_wise_weights.append(np.ones(array.shape[0]))

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


def run(args):
    dataset_dirs = get_datasets(folder=args.dataset_dir)

    for dataset, dataset_dir_path in dataset_dirs.items():
        logger.info(f"Processing dataset '{dataset}'")

        # Create majority vote folder
        out_dir = f'{dataset_dir_path}/{args.out_folder_name}'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Get all study IDs
        study_ids = set([os.path.split(s)[-1].split("_PRED")[0] for s in glob(dataset_dir_path + "/**/*PRED.npy")])
        logger.info(f"Found {len(study_ids)} paths to study IDs")

        for study_id in study_ids:
            logger.info(f"Processing study {study_id} for dataset {dataset}")

            out_path = f'{out_dir}/{study_id}_PRED'
            if os.path.exists(out_path) and not args.overwrite:
                logger.warning(f"Output file at {out_path} exists and the --overwrite flag was not set. Skipping.")
                continue

            combination_names, channels, combination_weights = get_input_channel_combinations(
                dataset_dir_path,
                study_id, 
                allowed_combinations=args.channel_combinations,
                combination_weights=args.combination_weights,
            )
            logger.info(f"Using {len(channels)} combinations ({combination_names}) for study {study_id} with weights {combination_weights}")
            
            if len(combination_names) == 0:
                logger.warning(f"No channel combinations found for study {study_id}. Skipping.")
                continue
            
            channel_arrs, element_wise_weights = get_arrays(channels)

            if args.soft:

                if args.softmax_fusion:
                    softmax_weights = get_softmax_weights(channel_arrs, temperature=args.temperature)
                    channel_arrs = apply_weights(channel_arrs, softmax_weights)

                channel_arrs = apply_weights(channel_arrs, combination_weights)
                channel_arrs = apply_weights(channel_arrs, element_wise_weights)

                mj = np.sum(channel_arrs, axis=0)
                # Normalize per row so each row sums to 1
                mj = mj / np.sum(mj, axis=1, keepdims=True)
            else:
                mj = stats.mode(channel_arrs, axis=0)[0].squeeze()

            logger.info(f"Saving majority with shape {mj.shape} to {out_path}")
            np.save(out_path, mj)


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


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    assert_args(args)
    run(args)

if __name__ == "__main__":
    entry_func()
