"""
Script for running preprocessing on a all or a subset of data described in
a U-Time project directory, loading all selected files with specified:
    - scaler
    - strip function
    - quality control function
    - channel selection
    - re-sampling

Loaded (and processed) files according to those settings are then saved to a single H5 archive for all
specified datasets and dataset splits.

The produced, pre-processed H5 archive of data may be consumed by the 'ut train' script setting flag --preprocessed.

This script should be called form within a U-Time project directory
"""

import logging
import os
from pathlib import Path
import numpy as np
import h5py
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
from utime import Defaults
from utime.utils import flatten_lists_recursively
from utime.hyperparameters import YAMLHParams
from utime.utils.scriptutils import assert_project_folder, get_splits_from_all_datasets, add_logging_file_handler
from psg_utils.time_utils import TimeUnit
from scipy.stats import mode

logger = logging.getLogger(__name__)

USABILITY_LABELS_FILE_NAME_TEMPLATE = "{}.csv"

def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='TODO')  # TODO
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path to output h5 archive")
    parser.add_argument("--dataset_splits", type=str, required=True, nargs='*',
                        help="Dataset splits (e.g. train_data) to preprocess "
                             "and save; space-separated list.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous pre-processed data')
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of threads to use for loading and "
                             "writing. Note: HDF5 must be compiled in "
                             "thread-safe mode!")
    parser.add_argument("--log_file", type=str, default="preprocessing",
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is 'preprocessing'")
    parser.add_argument("--apply-usability-labels", action="store_true", help="Apply usability labels to the hypnogram.")
    parser.add_argument("--unusable-class", type=int, default=6, help="Class to use for unusable data.")
    return parser


def copy_dataset_hparams(hparams, hparams_out_path):
    groups_to_save = ('select_channels',
                      'alternative_select_channels',
                      'channel_sampling_groups')
    hparams = hparams.save_current(hparams_out_path, return_copy=True)
    groups = list(hparams.keys())
    for group in groups:
        if group not in groups_to_save:
            hparams.delete_group(group)
    hparams.save_current()


def add_dataset_entry(hparams_out_path, h5_path,
                      split_identifier, period_length_sec):
    field = f"{split_identifier}_data:\n" + \
            f"  data_dir: {h5_path}\n" + \
            f"  period_length: {period_length_sec}\n" + \
            f"  time_unit: SECOND\n" + \
            f"  identifier: {split_identifier.upper()}\n\n"
    with open(hparams_out_path, "a") as out_f:
        out_f.write(field)


def update_labels_with_usability_scores(
    y: np.ndarray,
    usability_scores: np.ndarray,
    unusable_class: int
) -> np.ndarray:    
    
    if y.size == 0 or usability_scores.size == 0:
        raise ValueError("Input arrays cannot be empty")
    
    scaling_factor = len(usability_scores) // len(y)
    logger.info(f"Scaling factor: {scaling_factor}")

    if len(usability_scores) != len(y) * scaling_factor:
        logger.warning(f"Mismatch between label length ({len(y)}) and usability scores length "
                       f"({len(usability_scores)}) with scaling factor {scaling_factor}. "
                       f"Truncating usability scores to {len(y) * scaling_factor}.")

    usability_scores = usability_scores[:len(y) * scaling_factor]
    
    # Reshape usability scores into chunks and compute mode
    usability_scores_reshaped = usability_scores.reshape(-1, scaling_factor)
    usability_mask = mode(usability_scores_reshaped, axis=1)[0].squeeze() == 0

    logger.info(f"Number of usable periods: {np.sum(usability_mask)}/{len(usability_mask)}")
    
    # Create copy to avoid modifying input array
    y_updated = y.copy()
    
    # Apply usability mask (0 = usable, anything else = unusable)
    y_updated = np.where(usability_mask, y_updated, unusable_class)
    
    return y_updated


def preprocess_study(h5_file_group, study):
    """
    TODO

    Args:
        h5_file_group:
        study:

    Returns:
        None
    """
    # Create groups
    study_group = h5_file_group.create_group(study.identifier)

    psg_group = study_group.create_group("PSG")
    with study.loaded_in_context(allow_missing_channels=True):
        X, y = study.get_all_periods()
        for chan_ind, channel_name in enumerate(study.select_channels):
            # Create PSG channel datasets
            psg_group.create_dataset(channel_name.original_name,
                                     data=X[..., chan_ind].ravel())
        # Create hypnogram dataset
        study_group.create_dataset("hypnogram", data=y)

        # Create class --> index lookup groups
        cls_to_indx_group = study_group.create_group('class_to_index')
        dtype = np.dtype('uint16') if len(y) <= 65535 else np.dtype('uint32')
        classes = study.hypnogram.classes
        for class_ in classes:
            inds = np.where(y == class_)[0].astype(dtype)
            cls_to_indx_group.create_dataset(
                str(class_), data=inds
            )

        # Set attributes, currently only sample rate is (/may be) used
        study_group.attrs['sample_rate'] = study.sample_rate


def preprocess_study_with_usability_scores(h5_file_group, unusable_class: int, study):
    usability_scores_path = Path(study.subject_dir) / USABILITY_LABELS_FILE_NAME_TEMPLATE.format(study.identifier)
    all_usability_scores = pd.read_csv(usability_scores_path)

    # Create groups for each selected EEG channel
    for channel_name in all_usability_scores.columns:
        study_group_name = f"{study.identifier}_{channel_name}"
        study_group = h5_file_group.create_group(study_group_name)
        usability_scores = all_usability_scores[channel_name].array

        psg_group = study_group.create_group("PSG")
        with study.loaded_in_context(allow_missing_channels=True):
            X, y = study.get_all_periods()

            # Create PSG datasets for the selected EEG channel and all non-EEG channels
            for chan_ind, channel in enumerate(study.select_channels):
                if channel.original_name == channel_name or channel.original_name not in all_usability_scores.columns.tolist():
                    psg_group.create_dataset(channel.original_name,
                                             data=X[..., chan_ind].ravel())
                    
            # Run usability detection for the selected channel
            y = update_labels_with_usability_scores(y, usability_scores, unusable_class)
            study_group.create_dataset("hypnogram", data=y)

            # Create class --> index lookup groups
            cls_to_indx_group = study_group.create_group('class_to_index')
            dtype = np.dtype('uint16') if len(y) <= 65535 else np.dtype('uint32')
            classes = list(study.hypnogram.classes) + [unusable_class]
            for class_ in classes:
                inds = np.where(y == class_)[0].astype(dtype)
                cls_to_indx_group.create_dataset(
                    str(class_), data=inds
                )

            # Set attributes, currently only sample rate is (/may be) used
            study_group.attrs['sample_rate'] = study.sample_rate

            logger.info(f"Created group {study_group_name} with {len(y)}, channels {list(psg_group.keys())} and classes {classes}")


def run(args):
    """
    Run the script according to args - Please refer to the argparser.

    args:
        args:    (Namespace)  command-line arguments
    """

    project_dir = os.path.abspath("./")
    assert_project_folder(project_dir)
    logger.info(f"Args dump: {vars(args)}")

    # Load hparams
    hparams = YAMLHParams(Defaults.get_hparams_path(project_dir), no_version_control=True)

    # Initialize and load (potentially multiple) datasets
    datasets = get_splits_from_all_datasets(hparams,
                                            splits_to_load=args.dataset_splits,
                                            return_data_hparams=True)

    # Check if file exists, and overwrite if specified
    if os.path.exists(args.out_path):
        if args.overwrite:
            os.remove(args.out_path)
        else:
            from sys import exit
            logger.info(f"Out file at {args.out_path} exists, and --overwrite was not set")
            exit(0)

    # Create dataset hparams output directory
    out_dir = Defaults.get_pre_processed_data_configurations_dir(project_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with ThreadPoolExecutor(args.num_threads) as pool:
        with h5py.File(args.out_path, "w") as h5_file:
            for dataset, dataset_hparams in datasets:
                # Create a new version of the dataset-specific hyperparameters
                # that contain only the fields needed for pre-processed data
                name = dataset[0].identifier.split("/")[0]
                hparams_out_path = os.path.join(out_dir, name + ".yaml")

                copy_dataset_hparams(dataset_hparams, hparams_out_path)

                # Update paths to dataset hparams in main hparams file
                hparams.set_group(f"/datasets/{name}",
                                  value=hparams_out_path,
                                  overwrite=True)

                # Save the hyperparameters to the pre-processed main hparams
                hparams.save_current(Defaults.get_pre_processed_hparams_path(
                    project_dir
                ))

                # Process each dataset
                for split in dataset:
                    # Add this split to the dataset-specific hparams

                    period_length = split.pairs[0].get_period_length_in(TimeUnit.SECOND)

                    add_dataset_entry(hparams_out_path,
                                      args.out_path,
                                      split.identifier.split("/")[-1].lower(),
                                      period_length)

                    # Overwrite potential load time channel sampler to None
                    channel_sampling_groups = dataset_hparams.get('channel_sampling_groups')
                    if channel_sampling_groups:
                        unique_channels = list(set(flatten_lists_recursively(
                            channel_sampling_groups
                        )))
                        logger.info(f"Found channel sampling groups '{channel_sampling_groups}'. "
                                    f"Saving unique channels {unique_channels}.")
                        split.set_channel_sampling_groups(None)
                        split.set_select_channels(unique_channels)

                    # Create dataset group
                    split_group = h5_file.create_group(split.identifier)

                    # Run the preprocessing
                    if args.apply_usability_labels:
                        process_func = partial(preprocess_study_with_usability_scores, split_group, args.unusable_class)
                    else:
                        process_func = partial(preprocess_study, split_group)

                    logger.info(f"Preprocessing dataset: {split}")
                    n_pairs = len(split.pairs)
                    for i, _ in enumerate(pool.map(process_func,
                                                   split.pairs)):
                        print("  {}/{}".format(i+1, n_pairs),
                              end='\r', flush=True)
                    print("")


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    run(args)


if __name__ == "__main__":
    entry_func()
