import logging
import os
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)


N_EPOCHS_PER_HOUR = 3600 / 30  # 30 seconds per epoch

# Standard hypnogram colors (Wake at top, deeper stages darker)
STAGE_COLORS = {
    0: "#E8E8E8",   # Wake - light gray (color blind friendly)
    1: "#87CEEB",   # N1 - light sky blue
    2: "#4A90D9",   # N2 - medium blue
    3: "#1E3A5F",   # N3 - dark blue
    4: "#2E8B57",   # REM - sea green
}
UNMAPPED_COLOR = "#B0B0B0"  # Gray for unmapped/artifact/unknown
# Colors for true labels when they disagree with pred (only these are shown)
TRUE_MISMATCH_MAPPED = "#C41E3A"    # Red for mapped stages
TRUE_MISMATCH_UNMAPPED = "#6B6B6B"  # Dark gray for unmapped (Artifact, etc.)


def _get_runs(y: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return list of (start_idx, end_idx, value) for consecutive runs."""
    if len(y) == 0:
        return []
    changes = np.where(np.diff(y.astype(int)) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(y)]])
    return [(s, e, int(y[s])) for s, e in zip(starts, ends)]


def _get_mismatch_runs(
    y_true: np.ndarray, y_pred: np.ndarray
) -> List[Tuple[int, int, int]]:
    """Return runs of (start, end, true_value) where pred != true.
    Splits by true value changes so each run has constant true (avoids overlap with pred)."""
    mismatch = y_pred.astype(int) != y_true.astype(int)
    if not np.any(mismatch):
        return []
    # Pad to detect run boundaries
    padded = np.concatenate([[False], mismatch, [False]])
    changes = np.where(np.diff(padded))[0]
    runs = []
    for i in range(0, len(changes), 2):
        run_start, run_end = changes[i], changes[i + 1]
        # Within this mismatch region, split by true value (true can change within run)
        y_true_seg = y_true[run_start:run_end].astype(int)
        true_changes = np.where(np.diff(y_true_seg) != 0)[0] + 1
        sub_starts = np.concatenate([[0], true_changes])
        sub_ends = np.concatenate([true_changes, [len(y_true_seg)]])
        for s, e in zip(sub_starts, sub_ends):
            value = int(y_true_seg[s])
            runs.append((run_start + s, run_start + e, value))
    return runs


PRED_BLOCK_HEIGHT = 0.3
TRUE_BLOCK_HEIGHT = 0.1


def _draw_hypnogram_blocks(
    ax,
    y: np.ndarray,
    value_to_idx: Dict[int, int],
    value_to_color: Dict[int, str],
    value_to_hatch: Dict[int, str],
    epoch_duration_h: float,
    y_offset: float,
    height: float = 0.5,
) -> None:
    """Draw blocky hypnogram segments using broken_barh."""
    runs = _get_runs(y)
    for start_idx, end_idx, value in runs:
        idx = value_to_idx.get(value, value)
        x = start_idx * epoch_duration_h
        w = (end_idx - start_idx) * epoch_duration_h
        color = value_to_color.get(value, UNMAPPED_COLOR)
        hatch = value_to_hatch.get(value, "")
        ax.broken_barh([(x, w)], (y_offset + idx - height / 2, height),
                       facecolors=color, edgecolor=None, linewidth=0.1,
                       hatch=hatch, alpha=0.9 if not hatch else 0.85)


def get_hypnogram(
    y_pred,
    y_true=None,
    id_=None,
    class_mapping=None,
    unmapped_class_mapping: Optional[Dict[int, str]] = None,
    y_pred_probs: Optional[np.ndarray] = None,
):
    """
    Create a proper blocky hypnogram with stage colors, optionally with hypnodensity.

    Args:
        y_pred: Predicted labels (integer array).
        y_true: True labels (optional). May contain values not in class_mapping.
        id_: Identifier for the title.
        class_mapping: Dict mapping class int -> label for main stages.
        unmapped_class_mapping: Optional dict for labels of values in y_true
            not in class_mapping (e.g. {5: "Artifact", 6: "Unknown"}).
        y_pred_probs: Optional (n_epochs, n_classes) probability array for hypnodensity.
    """
    if class_mapping is None:
        n_classes = int(np.max(y_pred)) + 1
        class_mapping = {i: f"Class {i}" for i in range(n_classes)}

    # Build full display mapping: main classes + unmapped from y_true
    # Display order: Wake first, REM second, then N1/N2/N3 (or Light/Deep)
    mapped_values = set(class_mapping.keys())
    main_keys = sorted(class_mapping.keys())
    if len(main_keys) >= 3:
        rem_key = main_keys[-1]
        ordered_keys = [0] + [rem_key] + [k for k in main_keys if k != 0 and k != rem_key]
    else:
        ordered_keys = main_keys
    all_labels = [class_mapping[k] for k in ordered_keys]
    value_to_idx = {v: i for i, v in enumerate(ordered_keys)}
    stage_colors_list = list(STAGE_COLORS.values())
    value_to_color = {
        v: STAGE_COLORS.get(v, stage_colors_list[i % len(stage_colors_list)])
        for i, v in enumerate(ordered_keys)
    }
    value_to_hatch = {}

    unmapped_in_true = set()
    if y_true is not None:
        unmapped_in_true = set(np.unique(y_true)) - mapped_values

    for val in sorted(unmapped_in_true):
        label = (unmapped_class_mapping or {}).get(val, f"Class {val}")
        all_labels.append(label)
        idx = len(all_labels) - 1
        value_to_idx[val] = idx
        value_to_color[val] = UNMAPPED_COLOR
        value_to_hatch[val] = ""

    epoch_duration_h = 1.0 / N_EPOCHS_PER_HOUR
    total_hours = len(y_pred) * epoch_duration_h

    has_hypnodensity = (
        y_pred_probs is not None
        and y_pred_probs.ndim == 2
        and len(y_pred_probs) == len(y_pred)
        and y_pred_probs.shape[1] >= max(ordered_keys) + 1
    )
    if has_hypnodensity:
        fig = plt.figure(figsize=(15, 7))
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.12, 1, 1], hspace=0.2)
        ax_legend = fig.add_subplot(gs[0])
        ax_legend.axis("off")
        ax_density = fig.add_subplot(gs[1])
        ax = fig.add_subplot(gs[2], sharex=ax_density)
        axes = [ax_density, ax]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(15, 3))
        axes = [axes]
    fig.subplots_adjust(left=0.08, top=0.92, bottom=0.12, right=0.88)
    fig.suptitle("Hypnogram for Identifier: {}".format(id_ or "???"), fontsize=14, fontweight="bold", y=1.02)

    # Hypnodensity (top, when available)
    if has_hypnodensity:
        ax_density = axes[0]
        ids = np.arange(len(y_pred_probs)) * epoch_duration_h
        colors = [value_to_color[k] for k in ordered_keys]
        probs_stacked = np.column_stack([y_pred_probs[:, k] for k in ordered_keys])
        ax_density.stackplot(ids, *probs_stacked.T, colors=colors, alpha=0.9)
        ax_density.set_ylabel("Probability")
        ax_density.set_ylim(0, 1)
        ax_density.set_xlim(0, total_hours)
        ax_density.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_density.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.7)

    ax = axes[-1]  # Hypnogram on bottom (or only) axes
    ax.set_ylabel("Stage")
    ax.set_ylim(-0.5, len(all_labels) - 0.5)
    ax.set_xlim(0, total_hours)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Time (hours)")

    # Predicted: thin step line (behind) + filled blocks (on top)
    pred_value_to_idx = {v: value_to_idx[v] for v in mapped_values if v in value_to_idx}
    pred_value_to_color = {v: value_to_color[v] for v in mapped_values if v in value_to_color}
    ids = np.arange(len(y_pred)) * epoch_duration_h
    y_pred_idx = np.array([pred_value_to_idx.get(int(v), int(v)) for v in y_pred])
    ax.step(ids, y_pred_idx, where="post", color="#000000", linewidth=0.2, zorder=1)
    _draw_hypnogram_blocks(
        ax, y_pred, pred_value_to_idx, pred_value_to_color, {},
        epoch_duration_h, 0, height=PRED_BLOCK_HEIGHT
    )

    # True: only show segments where pred != true as blocks (avoids overlap); different colors for mapped vs unmapped
    if y_true is not None:
        mismatch_runs = _get_mismatch_runs(y_true, y_pred)
        # Build value->color for mismatch segments (solid fill, no hatch)
        mismatch_value_to_color = {}
        for _, _, value in mismatch_runs:
            is_unmapped = value in unmapped_in_true
            mismatch_value_to_color[value] = TRUE_MISMATCH_UNMAPPED if is_unmapped else TRUE_MISMATCH_MAPPED
        # Draw as thin blocks at TRUE stage row (different from pred row when mismatch)
        for start_idx, end_idx, value in mismatch_runs:
            value_idx = value_to_idx.get(value, value)
            x = start_idx * epoch_duration_h
            w = (end_idx - start_idx) * epoch_duration_h
            color = mismatch_value_to_color[value]
            ax.broken_barh([(x, w)], (value_idx - TRUE_BLOCK_HEIGHT / 2, TRUE_BLOCK_HEIGHT),
                           facecolors=color, edgecolor="none",
                           alpha=0.9, zorder=10)
    # Legend: stages, unmapped/artifact, and ground truth mismatch (when y_true present)
    handles = []
    for v in ordered_keys:
        handles.append(mpatches.Patch(facecolor=value_to_color[v], alpha=0.9, label=class_mapping[v]))
    for v in sorted(unmapped_in_true):
        label = (unmapped_class_mapping or {}).get(v, f"Class {v}")
        handles.append(mpatches.Patch(facecolor=value_to_color[v], alpha=0.9, label=label))
    if y_true is not None:
        handles.append(mpatches.Patch(facecolor=TRUE_MISMATCH_MAPPED, alpha=0.9,
                                      label="Ground truth mismatch"))

    if has_hypnodensity:
        ax_legend.legend(handles=handles, loc="center", ncol=len(handles), fontsize=9, frameon=True)
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    return fig, ax


def plot_and_save_hypnogram(
    out_path,
    y_pred,
    y_true=None,
    id_=None,
    class_mapping=None,
    unmapped_class_mapping: Optional[Dict[int, str]] = None,
    y_pred_probs: Optional[np.ndarray] = None,
):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    outs = get_hypnogram(
        y_pred, y_true, id_,
        class_mapping=class_mapping,
        unmapped_class_mapping=unmapped_class_mapping,
        y_pred_probs=y_pred_probs,
    )
    outs[0].savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(outs[0])


def plot_confusion_matrix(y_true, y_pred, n_classes,
                          normalize=False, id_=None,
                          cmap: str = "Blues", title: Optional[str] = None,
                          class_labels: Optional[list] = None):
    """
    Adapted from sklearn 'plot_confusion_matrix.py'.

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    if normalize:
        title = title or 'Normalized Confusion Matrix for Identifier: {}'.format(id_ or "???")
    else:
        title = title or 'Confusion Matrix, Without Normalization for Identifier: {}'.format(id_ or "???")

    # Compute confusion matrix
    classes = np.arange(n_classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype(np.float64)
        cm /= cm.sum(axis=1, keepdims=True)

    # Use provided labels or generate default ones
    labels = class_labels or [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(False)
    return fig, ax


def plot_and_save_cm(out_path, pred, true, n_classes, id_=None, normalized=True, title: Optional[str] = None, cmap: str = "Blues", class_labels: Optional[list] = None):
    dir_ = os.path.split(out_path)[0]
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fig, ax = plot_confusion_matrix(true, pred, n_classes, normalized, id_, cmap=cmap, title=title, class_labels=class_labels)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
