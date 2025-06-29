import os
from utime.utils.scriptutils.scriptutils import get_all_dataset_hparams
from psg_utils.dataset.sleep_study_dataset import SingleH5Dataset
from psg_utils.preprocessing.utils import set_preprocessing_pipeline
from typing import Generator, Tuple, List
from utime.hyperparameters import YAMLHParams
from psg_utils.dataset.sleep_study_dataset import H5Dataset
import numpy as np


def get_splits_from_h5_dataset(hparams: YAMLHParams, splits_to_load: Tuple[str, ...]) -> Generator[List[H5Dataset], None, None]:
    """
    Loads dataset splits from a single H5 dataset according to the specified split.

    Args:
        hparams: A YAMLHParams object storing references to one or more datasets.
        split_to_load: A string specifying the name of the sub-dataset to load according to its hparams description.

    Returns:
        Yields one or more splits of data from the dataset as described by 'hparams'.
    """

    def _get_split(h5_dataset: SingleH5Dataset, regex: str, hparams: dict) -> H5Dataset:
        h5_path = hparams['data_dir']
        if os.path.abspath(h5_path) != os.path.abspath(h5_dataset.h5_path):
            raise ValueError("All data must be stored in a single "
                             ".h5 file. Found two or more different files.")
        dataset = h5_dataset.get_datasets(
            load_match_regex=regex,
            period_length=hparams.get('period_length'),
            annotation_dict=hparams.get('sleep_stage_annotations')
        )
        assert len(dataset) == 1
        return dataset[0]
    
    data_hparams = get_all_dataset_hparams(hparams, dataset_ids=None)
    h5_dataset = None
    for dataset_id, hparams in data_hparams.items():

        splits = []

        for split_to_load in splits_to_load:
            if h5_dataset is None:
                h5_dataset = SingleH5Dataset(hparams[split_to_load]['data_dir'])
        
            splits.append(_get_split(
                h5_dataset=h5_dataset,
                regex=f'/{dataset_id}/{hparams[split_to_load]["identifier"]}',
                hparams=hparams[split_to_load]
            ))

        set_preprocessing_pipeline(*splits, hparams=hparams)
        
        yield splits


def get_splits_from_numpy_dataset(hparams: YAMLHParams, splits_to_load: Tuple[str, ...]) -> Generator[List[np.ndarray], None, None]:
    """
    Loads dataset splits from a numpy dataset according to the specified splits.

    Args:
        hparams: A YAMLHParams object storing references to one or more datasets.
        splits_to_load: A tuple of strings specifying the names of the sub-datasets to load according to their hparams descriptions.

    Returns:
        Yields one or more splits of data from the dataset as described by 'hparams'.
    """

    def _get_split(data_dir: str, regex: str, hparams: dict) -> np.ndarray:
        """
        Helper for returning a dataset from a numpy array according to
        regex and a hyperparameter set for a single dataset.
        """
        # Assuming data_dir contains numpy files matching the regex
        # This is a placeholder for actual loading logic
        data_path = os.path.join(data_dir, regex)
        dataset = np.load(data_path)
        return dataset

    data_hparams = get_all_dataset_hparams(hparams, dataset_ids=None)
    for dataset_id, hparams in data_hparams.items():

        splits = []

        for split_to_load in splits_to_load:
            data_dir = hparams[split_to_load]['data_dir']
            splits.append(_get_split(
                data_dir=data_dir,
                regex=f'{hparams[split_to_load]["identifier"]}.npy',
                hparams=hparams[split_to_load]
            ))

        set_preprocessing_pipeline(*splits, hparams=hparams)

        yield splits