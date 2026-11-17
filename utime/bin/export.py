"""
Export a trained model to ONNX or TensorFlow SavedModel.

Loads a model from a U-Time project directory and exports it for inference
outside this repository. Supports via --n_periods:

- omitted: training / sliding-window shape from hparams
- integer: fixed one-shot length
- 'dynamic': variable-length one-shot (USleep only)

ONNX export may optionally apply onnxruntime dynamic quantization.

Always writes a sibling metadata file ``model.yaml`` next to the artifact
(same directory as the ONNX file, or inside the TF SavedModel directory).
"""

import logging
import os
import shutil
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

from ruamel.yaml import YAML

from utime import Defaults
from utime.hyperparameters import YAMLHParams
from utime.utils.scriptutils import add_logging_file_handler, assert_project_folder
from utime.utils.system import find_and_set_gpus
from utime.bin.evaluate import get_and_load_model, get_and_load_one_shot_model

logger = logging.getLogger(__name__)

DEFAULT_ONNX_NAME = "model.onnx"
DEFAULT_TF_DIR_NAME = "tf"
DEFAULT_META_NAME = "model.yaml"
# Models known to support batch_shape[1] = None for variable-length one-shot.
DYNAMIC_PERIOD_MODELS = frozenset({"USleep"})
N_PERIODS_DYNAMIC = "dynamic"


def parse_n_periods(value):
    """
    Parse --n_periods: positive int (fixed one-shot) or 'dynamic' (variable length).
    Returns int or the sentinel string N_PERIODS_DYNAMIC.
    """
    if isinstance(value, str) and value.strip().lower() == N_PERIODS_DYNAMIC:
        return N_PERIODS_DYNAMIC
    try:
        n = int(value)
    except (TypeError, ValueError) as e:
        raise ArgumentTypeError(
            f"--n_periods must be a positive integer or '{N_PERIODS_DYNAMIC}', got {value!r}"
        ) from e
    if n < 1:
        raise ArgumentTypeError(
            f"--n_periods must be a positive integer or '{N_PERIODS_DYNAMIC}', got {value!r}"
        )
    return n


def get_argparser():
    parser = ArgumentParser(
        description="Export a trained model to ONNX or TensorFlow SavedModel."
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("onnx", "tf"),
        default="onnx",
        help="Export format. 'onnx' writes an ONNX file; 'tf' writes a TensorFlow "
        "SavedModel directory. Default: onnx",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output path. Defaults to <project_dir>/model/model.onnx for ONNX "
        f"or <project_dir>/model/{DEFAULT_TF_DIR_NAME} for TF SavedModel.",
    )
    parser.add_argument(
        "--weights_file_name",
        type=str,
        required=False,
        help="Exact name of the weights file in <project_dir>/model/ to use. "
        "If omitted, uses the 'best' weights selected by U-Time.",
    )
    parser.add_argument(
        "--n_periods",
        type=parse_n_periods,
        default=None,
        help="Period axis for export. Omit to use training batch_shape from hparams; "
        "pass a positive integer for fixed one-shot length; pass 'dynamic' for "
        "variable-length one-shot (USleep only).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (format=onnx only). Default: 13",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs to make visible for loading/export. Default: 0 (CPU).",
    )
    parser.add_argument("--force_gpus", type=str, default="")
    parser.add_argument(
        "--quantize",
        type=str,
        choices=("none", "dynamic"),
        default="none",
        help="ONNX quantization mode (format=onnx only). 'dynamic' performs "
        "weights-only quantization via onnxruntime and overwrites --out_path. "
        "Default: none",
    )
    parser.add_argument(
        "--quantize_weight_type",
        type=str,
        choices=("qint8", "quint8"),
        default="qint8",
        help="Weight quantization type for dynamic ONNX quantization. Default: qint8",
    )
    parser.add_argument(
        "--quantize_per_channel",
        action="store_true",
        help="Enable per-channel weight quantization for ONNX (if supported).",
    )
    parser.add_argument(
        "--quantize_reduce_range",
        action="store_true",
        help="Use reduced quantization range for ONNX (if supported).",
    )
    parser.add_argument(
        "--external_data",
        action="store_true",
        help="Export ONNX using external data format (useful for models >2GB).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="export.log",
        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
        "output log file for this script. Set to empty string to disable file logging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files or directories.",
    )
    return parser


def _default_out_path(project_dir: str, fmt: str) -> str:
    if fmt == "onnx":
        return os.path.join(project_dir, "model", DEFAULT_ONNX_NAME)
    return os.path.join(project_dir, "model", DEFAULT_TF_DIR_NAME)


def _meta_yaml_path(out_path: str, fmt: str) -> str:
    """
    Sidecar path: model.yaml in the ONNX file's directory, or inside the TF
    SavedModel directory.
    """
    if fmt == "tf" or os.path.isdir(out_path):
        return os.path.join(out_path, DEFAULT_META_NAME)
    return os.path.join(os.path.dirname(out_path) or ".", DEFAULT_META_NAME)


def _resolve_sample_rate_hz(hparams, project_dir: str):
    rate = hparams.get("set_sample_rate")
    if rate is None:
        rate = hparams.get("sample_rate")
    if rate is not None:
        return float(rate)

    datasets = hparams.get("datasets") or {}
    for rel_path in datasets.values():
        path = rel_path
        if not os.path.isabs(path):
            path = os.path.join(Defaults.get_hparams_dir(project_dir), path)
        if not os.path.exists(path):
            continue
        ds = YAMLHParams(path, no_version_control=True)
        rate = ds.get("set_sample_rate")
        if rate is None:
            rate = ds.get("sample_rate")
        if rate is not None:
            return float(rate)
    return None


def _class_labels(model, hparams):
    n_classes = int(getattr(model, "n_classes", None) or hparams["build"]["n_classes"])
    mapping = Defaults.get_class_int_to_stage_string()
    return [mapping.get(i, str(i)) for i in range(n_classes)]


def build_export_metadata(model, hparams, project_dir: str, n_periods_arg):
    """
    Collect inference-facing metadata for the model.yaml sidecar.
    """
    if n_periods_arg == N_PERIODS_DYNAMIC:
        n_periods_per_window = None
    else:
        n_periods_per_window = int(getattr(model, "n_periods", None) or hparams["build"]["batch_shape"][1])

    sample_rate_hz = _resolve_sample_rate_hz(hparams, project_dir)
    if sample_rate_hz is None:
        logger.warning(
            "Could not resolve sample_rate_hz from hparams or dataset configs; "
            "writing null in model.yaml."
        )

    return {
        "sample_rate_hz": sample_rate_hz,
        "n_periods_per_window": n_periods_per_window,
        "n_samples_per_period": int(model.input_dims),
        "n_channels": int(model.n_channels),
        "n_samples_per_prediction": int(model.data_per_prediction),
        "class_labels": _class_labels(model, hparams),
    }


def write_export_metadata(meta: dict, meta_path: str):
    meta_path = os.path.abspath(meta_path)
    parent = os.path.dirname(meta_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    logger.info(f"Writing export metadata to: {meta_path}")
    with open(meta_path, "w") as f:
        yaml.dump(meta, f)
    return meta_path


def _make_input_signature(model):
    # Keras model may have multiple inputs; tf2onnx expects a list/tuple of TensorSpec
    import tensorflow as tf

    sig = []
    for idx, inp in enumerate(model.inputs):
        shape = inp.shape
        dtype = inp.dtype if hasattr(inp, "dtype") else tf.float32
        name = inp.name.split(":")[0] if getattr(inp, "name", None) else f"input_{idx}"
        sig.append(tf.TensorSpec(shape=shape, dtype=dtype, name=name))
    return tuple(sig)


def _validate_args(args):
    if args.format == "tf":
        onnx_only = []
        if args.quantize != "none":
            onnx_only.append("--quantize")
        if args.external_data:
            onnx_only.append("--external_data")
        if args.opset != 13:
            onnx_only.append("--opset")
        if args.quantize_per_channel:
            onnx_only.append("--quantize_per_channel")
        if args.quantize_reduce_range:
            onnx_only.append("--quantize_reduce_range")
        if args.quantize_weight_type != "qint8":
            onnx_only.append("--quantize_weight_type")
        if onnx_only:
            raise ValueError(
                f"ONNX-only options not valid with --format tf: {', '.join(onnx_only)}"
            )


def _load_model_for_export(project_dir, hparams, args):
    model_cls = hparams["build"].get("model_class_name", "")
    batch_shape = list(hparams["build"]["batch_shape"])

    if args.n_periods == N_PERIODS_DYNAMIC:
        if model_cls not in DYNAMIC_PERIOD_MODELS:
            raise ValueError(
                f"--n_periods dynamic is not supported for model_class_name={model_cls!r}. "
                f"Supported: {sorted(DYNAMIC_PERIOD_MODELS)}. "
                "Use --n_periods with a fixed length (and pad at inference) instead."
            )
        hparams["build"]["batch_shape"][0] = 1
        hparams["build"]["batch_shape"][1] = None
        logger.info(
            f"Building variable-length one-shot model "
            f"(batch_shape={hparams['build']['batch_shape']})"
        )
        model = get_and_load_model(
            project_dir, hparams, args.weights_file_name, clear_previous=True
        )
    elif isinstance(args.n_periods, int):
        logger.info(
            f"Building fixed one-shot model with n_periods={args.n_periods} "
            f"(hparams period axis was {batch_shape[1]})"
        )
        model = get_and_load_one_shot_model(
            n_periods=args.n_periods,
            project_dir=project_dir,
            hparams=hparams,
            weights_file_name=args.weights_file_name,
            clear_previous=True,
        )
    else:
        logger.info(
            f"Building model with training/hparams batch_shape={batch_shape}"
        )
        model = get_and_load_model(
            project_dir, hparams, args.weights_file_name, clear_previous=True
        )

    input_sig = _make_input_signature(model)
    logger.info(f"Model inputs: {[s.name + str(tuple(s.shape)) for s in input_sig]}")
    return model


def export_to_onnx(model, out_path: str, opset: int, external_data: bool):
    try:
        import tf2onnx  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "Missing optional ONNX export dependencies. Install the extra with "
            '`pip install ".[onnx]"` (from repo), then retry.'
        ) from e

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    input_signature = _make_input_signature(model)
    logger.info(f"Exporting ONNX to: {out_path}")
    logger.info(f"Using opset: {opset}")
    logger.info(f"External data format: {bool(external_data)}")

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=out_path,
        large_model=bool(external_data),
    )
    return out_path


def export_to_tf(model, out_path: str):
    out_path = os.path.abspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    logger.info(f"Exporting TensorFlow SavedModel to: {out_path}")
    model.save(out_path, include_optimizer=False)
    return out_path


def quantize_dynamic_onnx(
    in_path: str,
    out_path: str,
    weight_type: str,
    per_channel: bool,
    reduce_range: bool,
):
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "Missing optional ONNX quantization dependencies. Install the extra with "
            '`pip install ".[onnx]"` (from repo), then retry.'
        ) from e

    wt = QuantType.QInt8 if weight_type == "qint8" else QuantType.QUInt8
    logger.info(f"Quantizing ONNX dynamically: {in_path} -> {out_path}")
    logger.info(
        f"Quantization settings: weight_type={weight_type}, "
        f"per_channel={bool(per_channel)}, reduce_range={bool(reduce_range)}"
    )
    quantize_dynamic(
        model_input=os.path.abspath(in_path),
        model_output=os.path.abspath(out_path),
        weight_type=wt,
        per_channel=bool(per_channel),
        reduce_range=bool(reduce_range),
    )
    return out_path


def _ensure_out_path_writable(out_path: str, overwrite: bool):
    if not os.path.exists(out_path):
        return
    if not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing path: {out_path} (use --overwrite)"
        )
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    else:
        os.remove(out_path)


def run(args):
    _validate_args(args)

    project_dir = os.path.abspath(Defaults.PROJECT_DIRECTORY)
    assert_project_folder(project_dir, evaluation=True)

    hparams = YAMLHParams(Defaults.get_hparams_path(project_dir))

    find_and_set_gpus(args.num_gpus, args.force_gpus)
    model = _load_model_for_export(project_dir, hparams, args)

    out_path = args.out_path or _default_out_path(project_dir, args.format)
    out_path = os.path.abspath(out_path)
    meta_path = _meta_yaml_path(out_path, args.format)
    _ensure_out_path_writable(out_path, args.overwrite)
    # For TF, meta lives inside out_path; wiping the dir already clears it.
    if args.format == "onnx":
        _ensure_out_path_writable(meta_path, args.overwrite)

    if args.format == "onnx":
        exported_path = export_to_onnx(
            model=model,
            out_path=out_path,
            opset=int(args.opset),
            external_data=bool(args.external_data),
        )
        if args.quantize == "dynamic":
            # ONNX Runtime quantizer requires a different output path. Write to a
            # temp file then replace to avoid a separate quantized artifact.
            p = Path(exported_path)
            tmp_out = str(p.with_name(p.stem + ".tmp_quant" + p.suffix))
            if os.path.exists(tmp_out):
                os.remove(tmp_out)

            quantize_dynamic_onnx(
                in_path=exported_path,
                out_path=tmp_out,
                weight_type=args.quantize_weight_type,
                per_channel=bool(args.quantize_per_channel),
                reduce_range=bool(args.quantize_reduce_range),
            )
            os.replace(tmp_out, exported_path)
    else:
        export_to_tf(model, out_path)

    meta = build_export_metadata(
        model=model,
        hparams=hparams,
        project_dir=project_dir,
        n_periods_arg=args.n_periods,
    )
    write_export_metadata(meta, meta_path)

    logger.info("Done.")


def entry_func(args=None):
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    logger.info(f"Args dump: {vars(args)}")
    run(args)


if __name__ == "__main__":
    entry_func()
