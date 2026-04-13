"""
Export a trained model to ONNX, optionally quantized.

This script loads a model from a U-Time project directory
and exports it to ONNX using tf2onnx. Optionally performs ONNX Runtime dynamic
quantization (weights-only int8/uint8) on the exported ONNX file.
"""

import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from utime import Defaults
from utime.hyperparameters import YAMLHParams
from utime.utils.scriptutils import add_logging_file_handler, assert_project_folder
from utime.utils.system import find_and_set_gpus
from utime.bin.evaluate import get_and_load_model

logger = logging.getLogger(__name__)


DEFAULT_OUT_NAME = "model.onnx"


def get_argparser():
    parser = ArgumentParser(description="Export a trained model to ONNX.")
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help=f"Output path for the ONNX model. Defaults to <project_dir>/model/{DEFAULT_OUT_NAME}",
    )
    parser.add_argument(
        "--weights_file_name",
        type=str,
        required=False,
        help="Exact name of the weights file in <project_dir>/model/ to use. "
        "If omitted, uses the 'best' weights selected by U-Time.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version to export with. Default: 13",
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
        help="Quantization mode. 'dynamic' performs weights-only quantization via onnxruntime and overwrites --out_path. Default: none",
    )
    parser.add_argument(
        "--quantize_weight_type",
        type=str,
        choices=("qint8", "quint8"),
        default="qint8",
        help="Weight quantization type for dynamic quantization. Default: qint8",
    )
    parser.add_argument(
        "--quantize_per_channel",
        action="store_true",
        help="Enable per-channel weight quantization (if supported).",
    )
    parser.add_argument(
        "--quantize_reduce_range",
        action="store_true",
        help="Use reduced quantization range (if supported).",
    )
    parser.add_argument(
        "--external_data",
        action="store_true",
        help="Export ONNX using external data format (useful for models >2GB).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="export_onnx.log",
        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
        "output log file for this script. Set to empty string to disable file logging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def _default_out_path(project_dir: str) -> str:
    return os.path.join(project_dir, "model", "model.onnx")


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


def export_to_onnx(model, out_path: str, opset: int, external_data: bool):
    try:
        import tf2onnx  # type: ignore[import-not-found]
    except Exception as e:
        raise RuntimeError(
            "Missing optional ONNX export dependencies. Install the extra with "
            '`pip install ".[onnx]"` (from repo), then retry.'
        ) from e

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    input_signature = _make_input_signature(model)
    logger.info(f"Exporting ONNX to: {out_path}")
    logger.info(f"Using opset: {opset}")
    logger.info(f"External data format: {bool(external_data)}")
    logger.info(f"Model inputs: {[s.name + str(tuple(s.shape)) for s in input_signature]}")

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=out_path,
        large_model=bool(external_data),
    )
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
        f"Quantization settings: weight_type={weight_type}, per_channel={bool(per_channel)}, reduce_range={bool(reduce_range)}"
    )
    quantize_dynamic(
        model_input=os.path.abspath(in_path),
        model_output=os.path.abspath(out_path),
        weight_type=wt,
        per_channel=bool(per_channel),
        reduce_range=bool(reduce_range),
    )
    return out_path


def run(args):
    project_dir = os.path.abspath(Defaults.PROJECT_DIRECTORY)
    assert_project_folder(project_dir, evaluation=True)

    # Load hyperparameters (same path used by predict.py)
    hparams = YAMLHParams(Defaults.get_hparams_path(project_dir))

    # Load model and weights
    find_and_set_gpus(args.num_gpus, args.force_gpus)
    model = get_and_load_model(project_dir, hparams, args.weights_file_name, clear_previous=True)

    out_path = args.out_path or _default_out_path(project_dir)
    out_path = os.path.abspath(out_path)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")

    exported_path = export_to_onnx(
        model=model,
        out_path=out_path,
        opset=int(args.opset),
        external_data=bool(args.external_data),
    )

    if args.quantize == "dynamic":
        # ONNX Runtime quantizer requires a different output path. We write to a
        # temp file and then replace the original export to avoid creating a
        # separate "quantized output" artifact.
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

    logger.info("Done.")


def entry_func(args=None):
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    logger.info(f"Args dump: {vars(args)}")
    run(args)


if __name__ == "__main__":
    entry_func()

