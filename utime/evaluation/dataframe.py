import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_eval_df(sequencer):
    ids = [ss.identifier for ss in sequencer.get_pairs()]
    classes = ["mean"] + ["cls {}".format(i) for i in range(sequencer.n_classes)]
    return pd.DataFrame(columns=ids, index=classes)


def add_to_eval_df(eval_dict, id_, values):
    """
    Add a per-class metric vector to an evaluation DataFrame.

    The evaluation DataFrame is expected to have index:
        ["mean"] + <one row per class label>

    Some metric functions (e.g., sklearn f1_score with average=None and no
    explicit labels) may return vectors shorter than the number of classes if
    only a subset of classes is present. To keep the DataFrame shape stable, we
    pad/truncate to the expected number of classes (filling missing entries with
    NaN) before computing the mean.
    """
    # Normalize to 1D float array
    arr = np.asarray(values, dtype=np.float64).reshape(-1)

    # Align to expected per-class length if eval_dict has an index
    expected = None
    try:
        expected = max(0, len(eval_dict.index) - 1)  # exclude "mean"
    except Exception:
        expected = None

    if expected is not None and expected >= 0 and arr.size != expected:
        aligned = np.full(shape=(expected,), fill_value=np.nan, dtype=np.float64)
        n = min(expected, arr.size)
        if n:
            aligned[:n] = arr[:n]
        arr = aligned

    mean = np.nanmean(arr) if arr.size else np.nan
    eval_dict[id_] = [mean] + arr.tolist()


def with_grand_mean_col(eval_dict, col_name="Grand mean"):
    means = np.mean(eval_dict, axis=1)
    eval_dict[col_name] = means
    cols = list(eval_dict.columns)
    cols.append(cols.pop(cols.index(col_name)))
    return eval_dict.loc[:, cols]


def log_eval_df_to_screen(eval_dict, round=4, txt=None):
    log = f"\n[*] {txt or 'EVALUATION RESULTS'}"
    logger.info(
        log + "\n" + "-"*len(log) + "\n" + str(eval_dict.round(round)) + "\n" + "-"*len(log)
    )


def log_eval_df_to_file(eval_dict, out_csv_file=None, out_txt_file=None, round=4):
    if out_csv_file:
        with open(out_csv_file, "w+") as out_csv:
            out_csv.write(eval_dict.to_csv())
    if out_txt_file:
        with open(out_txt_file, "w+") as out_txt:
            out_txt.write(eval_dict.round(round).to_string())


def log_eval_df(eval_dict, out_csv_file=None, out_txt_file=None, round=4, txt=None):
    log_eval_df_to_screen(eval_dict, round, txt)
    log_eval_df_to_file(eval_dict, out_csv_file, out_txt_file, round)
