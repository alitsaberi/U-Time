from pathlib import Path
import h5py
import numpy as np

from psg_utils.hypnogram.utils import ndarray_to_ids_format
from psg_utils.time_utils import TimeUnit


AROUSAL_LABELS = {
    1: "arousal",
    0: "non-arousal",
    -1: "unscored"
}

SAMPLE_RATE = 200

def extract_arousal(file_path: Path, period_length: int, time_unit: TimeUnit = TimeUnit.SECOND, sample_rate: int = SAMPLE_RATE):
    with h5py.File(file_path, "r") as f:
        arousal_labels = f["data"]["arousal"][:].squeeze()
        arousal_labels = np.vectorize(AROUSAL_LABELS.get)(arousal_labels)

    initials, durations, labels = ndarray_to_ids_format(
        array=arousal_labels,
        period_length=period_length,
        time_unit=time_unit,
        sample_rate=sample_rate
    )

    return initials, durations, labels
    
    