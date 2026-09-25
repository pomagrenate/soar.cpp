"""
SOAR Ultralytics-Style Plotting & Visualization Engine
Generates publication-quality metric dashboards (results.png) and visual validation
overlays (val_batch_pred.png) with aesthetic styling matching Ultralytics YOLO.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend suitable for headless environments
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as PathEffects
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _smooth(values: List[float], weight: float = 0.6) -> List[float]:
    """Exponential Moving Average (EMA) smoothing matching TensorBoard/Ultralytics."""
    if not values:
        return []
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_results(csv_path: Union[str, Path] = "runs/results.csv",
                 save_dir: Optional[Union[str, Path]] = None,
                 dpi: int = 300) -> Optional[str]:
    """
    Plots an Ultralytics-style 5-panel metrics dashboard from results.csv.

    Columns supported in results.csv:
    epoch, train_loss, val_loss, val_dice, val_iou, lr
    """
    if not HAS_MATPLOTLIB:
        print("[SOAR Plotting] matplotlib not installed. Skipping plot_results.")
        return None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[SOAR Plotting] {csv_path} not found.")
        return None

    save_dir = Path(save_dir) if save_dir else csv_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / "results.png"

    # Read CSV
    import csv
    epochs, train_losses, val_losses, val_dices, val_ious, lrs = [], [], [], [], [], []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row.get("epoch", len(epochs) + 1)))
            train_losses.append(float(row.get("train_loss", row.get("loss", 0.0))))
            val_losses.append(float(row.get("val_loss", 0.0)))
            val_dices.append(float(row.get("val_dice", row.get("dice", 0.0))))
            val_ious.append(float(row.get("val_iou", row.get("iou", 0.0))))
            lrs.append(float(row.get("lr", 0.0)))

    if not epochs:
        return None

    # Ultralytics Style configuration
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=dpi)
    axes = axes.flatten()

    def style_ax(ax, title: str, xlabel: str = "Epoch", ylabel: str = ""):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8, color="#222222")
        ax.set_xlabel(xlabel, fontsize=10, color="#555555")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, color="#555555")
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 1. Train Loss
    ax = axes[0]
    ax.plot(epochs, train_losses, color="#4A90E2", alpha=0.3, linewidth=1.0)
    ax.plot(epochs, _smooth(train_losses), color="#0D47A1", linewidth=2.0, label="train/loss")
    style_ax(ax, "train/loss", ylabel="Loss")
    ax.legend(frameon=False, loc="upper right")

    # 2. Validation Loss
    ax = axes[1]
    if any(val_losses):
        ax.plot(epochs, val_losses, color="#FF9800", alpha=0.3, linewidth=1.0)
        ax.plot(epochs, _smooth(val_losses), color="#E65100", linewidth=2.0, label="val/loss")
    style_ax(ax, "val/loss", ylabel="Loss")
    if any(val_losses): ax.legend(frameon=False, loc="upper right")

    # 3. Validation Dice Score
    ax = axes[2]
    if any(val_dices):
        ax.plot(epochs, val_dices, color="#4CAF50", alpha=0.3, linewidth=1.0)
        ax.plot(epochs, _smooth(val_dices), color="#1B5E20", linewidth=2.0, label="metrics/dice")
        best_dice_idx = int(np.argmax(val_dices))
        ax.scatter([epochs[best_dice_idx]], [val_dices[best_dice_idx]], color="#D32F2F", s=40, zorder=5)
        ax.annotate(f"Best: {val_dices[best_dice_idx]:.4f}",
                    (epochs[best_dice_idx], val_dices[best_dice_idx]),
                    textcoords="offset points", xytext=(-20, 10),
                    fontsize=8, fontweight="bold", color="#D32F2F")
    style_ax(ax, "metrics/dice", ylabel="Dice Score")
    if any(val_dices): ax.legend(frameon=False, loc="lower right")

    # 4. Validation IoU Score
    ax = axes[3]
    if any(val_ious):
        ax.plot(epochs, val_ious, color="#9C27B0", alpha=0.3, linewidth=1.0)
        ax.plot(epochs, _smooth(val_ious), color="#4A148C", linewidth=2.0, label="metrics/iou")
        best_iou_idx = int(np.argmax(val_ious))
        ax.scatter([epochs[best_iou_idx]], [val_ious[best_iou_idx]], color="#D32F2F", s=40, zorder=5)
        ax.annotate(f"Best: {val_ious[best_iou_idx]:.4f}",
                    (epochs[best_iou_idx], val_ious[best_iou_idx]),
                    textcoords="offset points", xytext=(-20, 10),
                    fontsize=8, fontweight="bold", color="#D32F2F")
    style_ax(ax, "metrics/iou", ylabel="IoU Score")
    if any(val_ious): ax.legend(frameon=False, loc="lower right")

    # 5. Learning Rate
    ax = axes[4]
    ax.plot(epochs, lrs, color="#00ACC1", linewidth=2.0, label="lr")
    style_ax(ax, "x/lr", ylabel="Learning Rate")
    ax.legend(frameon=False, loc="upper right")

    # 6. Convergence Comparison
    ax = axes[5]
    ax.plot(epochs, _smooth(train_losses), color="#0D47A1", linewidth=1.5, label="Train Loss")
    if any(val_losses):
        ax.plot(epochs, _smooth(val_losses), color="#E65100", linewidth=1.5, label="Val Loss")
    style_ax(ax, "convergence (train vs val)", ylabel="Loss")
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[SOAR Plotting] Saved Ultralytics results dashboard to {out_file}")
    return str(out_file)


def plot_validation_batch(images: List[np.ndarray],
                          masks_true: List[np.ndarray],
                          masks_pred: List[np.ndarray],
                          save_path: Union[str, Path] = "runs/val_batch0_pred.jpg",
                          scores: Optional[List[Dict[str, float]]] = None,
                          dpi: int = 200) -> Optional[str]:
    """
    Renders an Ultralytics-style visual comparison grid of validation samples:
    [Raw Image + GT Contour] vs [Raw Image + Pred Mask + Confidence Badge]
    """
    if not HAS_MATPLOTLIB:
        print("[SOAR Plotting] matplotlib not installed. Skipping plot_validation_batch.")
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = min(len(images), 4)
    if n_samples == 0:
        return None

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples), dpi=dpi)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n_samples):
        img = np.squeeze(images[i])
        gt = np.squeeze(masks_true[i])
        pred = np.squeeze(masks_pred[i])

        # Normalize image to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        # Panel 1: Raw Image
        ax = axes[i, 0]
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(f"Sample {i + 1}: Raw Image", fontsize=10, fontweight="bold")
        ax.axis("off")

        # Panel 2: Ground Truth Overlay
        ax = axes[i, 1]
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        # Transparent Green GT overlay
        gt_overlay = np.zeros((*gt.shape, 4), dtype=np.float32)
        gt_overlay[gt > 0.5] = [0.1, 0.85, 0.2, 0.45]
        ax.imshow(gt_overlay)
        ax.contour(gt > 0.5, levels=[0.5], colors=["#00E676"], linewidths=1.0)
        ax.set_title("Ground Truth (Green)", fontsize=10, fontweight="bold")
        ax.axis("off")

        # Panel 3: Prediction Overlay with Badge
        ax = axes[i, 2]
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        # Transparent Cyan/Blue Pred overlay
        pred_overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
        pred_overlay[pred > 0.5] = [0.0, 0.7, 0.95, 0.45]
        ax.imshow(pred_overlay)
        ax.contour(pred > 0.5, levels=[0.5], colors=["#00E5FF"], linewidths=1.0)

        title = "Prediction (Cyan)"
        if scores and i < len(scores):
            d = scores[i].get("dice", 0.0)
            iou = scores[i].get("iou", 0.0)
            title += f" [Dice: {d:.3f} | IoU: {iou:.3f}]"

        ax.set_title(title, fontsize=10, fontweight="bold", color="#007791")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[SOAR Plotting] Saved Ultralytics validation visualizer to {save_path}")
    return str(save_path)
