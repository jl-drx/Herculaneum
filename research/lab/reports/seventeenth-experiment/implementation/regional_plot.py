"""Fixed-display regional panels; callers supply already verified arrays.

No experiment inputs are opened here. Raster y increases downward, coarse masks
are repeated exactly 3x3 for native overlays, and no registration is performed.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/herculaneum-regional-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/herculaneum-regional-cache")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


DISPLAY_PLANES = (8, 20, 32, 44, 56)
DISPLAY_POOLED = (0, 4, 8, 12, 16)
SUPPORTS = ("all", "surface", "primary_positive", "primary_negative")
SUPPORT_COLORS = {
    "all": "#333333", "surface": "#d58a00",
    "primary_positive": "#b32e8c", "primary_negative": "#007a99",
}
OVERLAY_COLORS = {"labels": "#ff40d3", "primary_mask": "#00ddff", "surface_mask": "#ffe600"}
PNG_CAP = 6 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(array, shape, dtype, name):
    _require(isinstance(array, np.ndarray), f"{name} must be an array")
    _require(array.shape == shape, f"{name} must have shape {shape}")
    _require(array.dtype == np.dtype(dtype), f"{name} must have dtype {dtype}")
    _require(np.isfinite(array).all(), f"{name} contains nonfinite values")


def _masks(masks):
    _array(masks["labels"], (384, 384), "uint8", "labels")
    _require(np.isin(masks["labels"], [0, 1]).all(), "Literal labels must be binary")
    for key in ("primary_mask", "surface_mask"):
        _array(masks[key], (384, 384), "bool", key)


def _title(case):
    x, y, width, height = case["bbox_source_xywh"]
    _require((width, height) == (1152, 1152), "Panel requires frozen 1152-square crop")
    _require(case["native_bbox_yxhw"] == [y, x, height, width], "XY/YX crop disagreement")
    return f"{case['id']} | source XYWH [{x}, {y}, {width}, {height}]"


def _raster(axis, array, title, vmin, vmax, cmap="gray", native=False, overlay=None):
    image = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    axis.set_title(title, fontsize=8)
    size = 1152 if native else 384
    axis.set_xticks([0, 577, 1151] if native else [0, 192, 383])
    axis.set_yticks([0, 577, 1151] if native else [0, 192, 383])
    axis.tick_params(labelsize=6, pad=1, length=2)
    if overlay is not None:
        for key in ("labels", "primary_mask", "surface_mask"):
            support = overlay[key] == 1 if key == "labels" else overlay[key]
            if native:
                support = np.repeat(np.repeat(support, 3, axis=0), 3, axis=1)
            # Constant masks have no internal contour; their literal mask panels
            # remain visible. Contour vertices use unchanged pixel coordinates.
            if support.any() and not support.all():
                axis.contour(support.astype(np.uint8), levels=[0.5], colors=[OVERLAY_COLORS[key]],
                             linewidths=0.45, alpha=0.85)
    axis.set_xlim(-0.5, size - 0.5)
    axis.set_ylim(size - 0.5, -0.5)
    return image


def _colorbar(figure, image, axes, label):
    bar = figure.colorbar(image, ax=axes, fraction=0.014, pad=0.012, aspect=28)
    bar.ax.tick_params(labelsize=6, length=2)
    bar.set_label(label, fontsize=7)


def _cross_section(axis, array, title, horizontal_label):
    half_spacing = 3.24 / 2
    image = axis.imshow(
        array, cmap="gray", vmin=0, vmax=65535, origin="upper", interpolation="nearest",
        aspect="auto", extent=(-0.5, 1151.5, 81 + half_spacing, -81 - half_spacing),
    )
    axis.axhline(0, color="#00ddff", linewidth=0.5, linestyle="--")
    axis.set_title(title, fontsize=8)
    axis.set_xlabel(horizontal_label, fontsize=7, labelpad=1)
    axis.set_ylabel("Nominal depth (μm)", fontsize=7, labelpad=1)
    axis.set_xticks([0, 577, 1151])
    axis.set_yticks([-81, 0, 81])
    axis.tick_params(labelsize=6, pad=1, length=2)
    return image


def _profile(axis, records, native):
    expected_planes = list(range(7, 58)) if native else list(range(8, 57, 3))
    _require([r["plane_index"] for r in records] == expected_planes, "Profile plane schedule differs")
    depth = np.array([r["nominal_depth_um"] for r in records], dtype=np.float64)
    expected_depth = 3.24 * (np.asarray(expected_planes, dtype=np.float64) - 32)
    _require(np.allclose(depth, expected_depth, rtol=0, atol=1e-10), "Profile nominal depths differ")
    for key in SUPPORTS:
        means = np.array([np.nan if r["supports"][key]["mean"] is None
                          else r["supports"][key]["mean"] for r in records], dtype=np.float64)
        stds = np.array([np.nan if r["supports"][key]["std"] is None
                         else r["supports"][key]["std"] for r in records], dtype=np.float64)
        _require(not np.isinf(means).any() and not np.isinf(stds).any(), "Infinite profile statistic")
        _require(np.all(stds[np.isfinite(stds)] >= 0), "Negative profile standard deviation")
        axis.plot(depth, means, color=SUPPORT_COLORS[key], linewidth=1.15, label=key.replace("_", " "))
        axis.fill_between(depth, means - stds, means + stds, color=SUPPORT_COLORS[key], alpha=0.075)
    axis.axvline(0, color="#555555", linewidth=0.6, linestyle="--")
    axis.set_xlim(-81, 81)
    axis.set_ylim(0, 65535 if native else 255)
    axis.set_xlabel("Nominal depth: 3.24 × (plane − 32) μm", fontsize=7, labelpad=2)
    axis.set_ylabel("Mean native uint16" if native else "Mean pooled CT (0–255)", fontsize=7)
    axis.set_title("Native: all 51 planes; expanded coarse supports" if native
                   else "Pooled: all 17 planes; literal coarse supports", fontsize=8)
    axis.tick_params(labelsize=6, pad=1, length=2)
    axis.grid(alpha=0.15, linewidth=0.5)
    axis.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.8)


def _save_png(figure, output_path):
    """Encode RGB pixels before writing; reject overwrites and oversized output."""
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())
    image = Image.fromarray(pixels).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=9)
    body = buffer.getvalue()
    _require(len(body) < PNG_CAP, "Panel exceeds frozen 6 MiB cap")
    output_path = Path(output_path)
    with output_path.open("xb") as handle:
        handle.write(body)
    return {"path": str(output_path.resolve()), "size_bytes": len(body),
            "width_px": image.width, "height_px": image.height, "format": "RGB PNG"}


def make_ct_panel(output_path, case, native_display, pooled_volume, masks, scores,
                  cross_row, cross_col, profiles):
    """Write the fixed native/coarse/depth overview and return its byte receipt."""
    title = _title(case)
    _require(set(native_display) == set(DISPLAY_PLANES), "Native display plane schedule differs")
    for plane in DISPLAY_PLANES:
        _array(native_display[plane], (1152, 1152), "uint16", f"native plane {plane}")
    _array(pooled_volume, (17, 384, 384), "float32", "pooled volume")
    _require(np.all((pooled_volume >= 0) & (pooled_volume <= 255)), "Pooled CT exceeds fixed scale")
    _array(scores, (384, 384), "float32", "saved scores")
    _require(np.all((scores >= 0) & (scores <= 1)), "Saved scores exceed fixed scale")
    _array(cross_row, (51, 1152), "uint16", "row cross-section")
    _array(cross_col, (51, 1152), "uint16", "column cross-section")
    _masks(masks)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8}):
        figure = plt.figure(figsize=(16, 12.8), dpi=100, facecolor="white")
        try:
            grid = figure.add_gridspec(5, 10, left=0.045, right=0.975, bottom=0.065, top=0.925,
                                      hspace=0.48, wspace=0.8, height_ratios=[1, 1, 1, 0.75, 1.15])
            native_axes, pooled_axes = [], []
            for column, (plane, pooled_index) in enumerate(zip(DISPLAY_PLANES, DISPLAY_POOLED)):
                axis = figure.add_subplot(grid[0, 2 * column:2 * column + 2])
                native_axes.append(axis)
                native_image = _raster(axis, native_display[plane], f"Native plane {plane}",
                                       0, 65535, native=True, overlay=masks)
                if plane == 32:
                    axis.axhline(577, color="#00ddff", linewidth=0.5, linestyle="--")
                    axis.axvline(577, color="#00ddff", linewidth=0.5, linestyle="--")
                axis = figure.add_subplot(grid[1, 2 * column:2 * column + 2])
                pooled_axes.append(axis)
                pooled_image = _raster(axis, pooled_volume[pooled_index],
                                       f"Pooled {pooled_index} | center {plane}", 0, 255, overlay=masks)
            _colorbar(figure, native_image, native_axes, "Native uint16")
            _colorbar(figure, pooled_image, pooled_axes, "Pooled CT")
            axis = figure.add_subplot(grid[2, 0:2])
            score_image = _raster(axis, scores, "Saved baseline score", 0, 1, cmap="viridis", overlay=masks)
            _colorbar(figure, score_image, [axis], "Score")
            for column, (key, label) in enumerate([
                ("labels", "Literal released labels"), ("primary_mask", "Literal primary support"),
                ("surface_mask", "Literal surface support"),
            ], start=1):
                _raster(figure.add_subplot(grid[2, 2 * column:2 * column + 2]),
                        masks[key], label + " (0/1)", 0, 1)
            legend_axis = figure.add_subplot(grid[2, 8:10])
            legend_axis.axis("off")
            handles = [Line2D([0], [0], color=OVERLAY_COLORS[key], lw=1.5, label=label)
                       for key, label in [("labels", "label = 1"), ("primary_mask", "primary"),
                                          ("surface_mask", "surface")]]
            legend_axis.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
                               title="Literal contours", title_fontsize=8)
            legend_axis.text(0, 0.35, "Native masks: exact 3× repeats\nRaster y increases downward\nNo flips or fitted registration\n\nProfiles: mean ± population SD\nEmpty supports: line gaps",
                             va="top", fontsize=7, linespacing=1.5)
            row_axis = figure.add_subplot(grid[3, :5])
            col_axis = figure.add_subplot(grid[3, 5:])
            _cross_section(row_axis, cross_row, "Native local row 577 | all planes 7–57", "Native local x (px)")
            _cross_section(col_axis, cross_col, "Native local column 577 | all planes 7–57", "Native local y (px)")
            _profile(figure.add_subplot(grid[4, :5]), profiles["native"], native=True)
            _profile(figure.add_subplot(grid[4, 5:]), profiles["pooled"], native=False)
            figure.suptitle(title, fontsize=12, y=0.975)
            figure.text(0.5, 0.95, "Fixed CT displays and saved scores; nominal depth does not establish physical surface or normal direction",
                        ha="center", fontsize=9)
            figure.text(0.5, 0.016, "Evaluation-informed diagnostic; all intensities/supports retain their original raster alignment. CT associations alone cannot identify ink.",
                        ha="center", fontsize=8)
            return _save_png(figure, output_path)
        finally:
            plt.close(figure)


def make_ir_panel(output_path, case, gray_native, gray_coarse, local_darkness, masks):
    """Write native/coarse IR, fixed local darkness, and the literal coarse masks."""
    title = _title(case)
    _array(gray_native, (1152, 1152), "uint8", "native infrared gray")
    _array(gray_coarse, (384, 384), "float64", "coarse infrared gray")
    _array(local_darkness, (384, 384), "float64", "coarse local darkness")
    _require(np.all((gray_coarse >= 0) & (gray_coarse <= 255)), "Infrared gray exceeds fixed scale")
    _masks(masks)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8}):
        figure = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
        try:
            grid = figure.add_gridspec(2, 3, left=0.05, right=0.95, bottom=0.09, top=0.87,
                                      hspace=0.3, wspace=0.3)
            native_axis = figure.add_subplot(grid[0, 0])
            native_image = _raster(native_axis, gray_native, "Registered IR: native crop", 0, 255,
                                   native=True, overlay=masks)
            coarse_axis = figure.add_subplot(grid[0, 1])
            _raster(coarse_axis, gray_coarse, "IR: exact 3×3 block mean", 0, 255, overlay=masks)
            _colorbar(figure, native_image, [native_axis, coarse_axis], "Gray (0–255)")
            local_axis = figure.add_subplot(grid[0, 2])
            local_image = _raster(local_axis, local_darkness, "Local darkness: fixed σ=16 native px", -64, 64,
                                  cmap="RdBu_r", overlay=masks)
            _colorbar(figure, local_image, [local_axis], "Blurred gray − gray")
            for column, (key, label) in enumerate([
                ("labels", "Literal released labels"), ("primary_mask", "Literal primary support"),
                ("surface_mask", "Literal surface support"),
            ]):
                _raster(figure.add_subplot(grid[1, column]), masks[key], label + " (0/1)", 0, 1)
            figure.suptitle(title, fontsize=12, y=0.965)
            figure.text(0.5, 0.925, "Fixed infrared association display | magenta: label = 1; cyan: primary; yellow: surface",
                        ha="center", fontsize=9)
            figure.text(0.5, 0.035, "Raster y downward; native contours are exact 3× repeats. Agreement with IR-derived labels is not independent registration accuracy.",
                        ha="center", fontsize=8)
            return _save_png(figure, output_path)
        finally:
            plt.close(figure)
