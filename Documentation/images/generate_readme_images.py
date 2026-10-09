"""Render README illustrations from saved notebook results (requires Pillow).

Run from the repository root: python Documentation/images/generate_readme_images.py
The notebook is the numerical source; these illustrations do not rerun the model.
"""

import base64
import json
import math
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INK = "#222222"
MUTED = "#555555"
BLUE = "#336699"
TEAL = "#477457"
ORANGE = "#B4772C"
FONT = next(
    (
        p
        for p in [
            Path("/System/Library/Fonts/Helvetica.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        if p.exists()
    ),
    None,
)


def font(size):
    return (
        ImageFont.truetype(str(FONT), size)
        if FONT
        else ImageFont.load_default(size=size)
    )


def canvas(height=1050):
    im = Image.new("RGB", (1800, height), "white")
    draw = ImageDraw.Draw(im)
    return im, draw


def text(draw, xy, value, size=27, fill=INK, anchor=None):
    draw.text(xy, value, font=font(size), fill=fill, anchor=anchor)


def figure(title):
    notebook = json.loads((ROOT / "Notebooks/reservoir_pandas_api.ipynb").read_text())
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            result = output.get("data", {}).get("application/vnd.plotly.v1+json")
            if (
                result
                and result.get("layout", {}).get("title", {}).get("text") == title
            ):
                return result
    raise ValueError(f"No saved figure: {title}")


def values(trace):
    data = trace["y"]
    if isinstance(data, list):
        return data
    if data["dtype"] != "f8":
        raise ValueError("Expected float64 notebook results")
    raw = base64.b64decode(data["bdata"])
    return struct.unpack("<" + "d" * (len(raw) // 8), raw)


def axes(draw, ymin, ymax, ticks):
    left, top, right, bottom = 155, 240, 1715, 875

    def point(i, y):
        return left + i / 1344 * (right - left), bottom - (y - ymin) / (ymax - ymin) * (
            bottom - top
        )

    text(draw, (left, 198), "Flow (cfs)", 24, MUTED)
    for tick in ticks:
        py = point(0, tick)[1]
        draw.line((left, py, right, py), fill="#E6E6E6" if tick else "#999999", width=2)
        text(draw, (left - 22, py), f"{tick:,}", 24, MUTED, "rm")
    for day in range(1, 16, 2):
        px = point((day - 1) * 96, 0)[0]
        text(draw, (px, bottom + 28), f"Jan {day:02}", 24, MUTED, "mt")
    draw.line((left, top, left, bottom, right, bottom), fill="#888888", width=2)
    return point


def line(draw, data, point, color, width=4):
    segment = []
    for i, y in enumerate(data):
        if math.isfinite(y):
            segment.append(point(i, y))
        else:
            if len(segment) > 1:
                draw.line(segment, fill=color, width=width)
            segment = []
    if len(segment) > 1:
        draw.line(segment, fill=color, width=width)


def legend(draw, x, y, label, color, band=False):
    if band:
        draw.rectangle((x, y + 5, x + 46, y + 25), fill=color)
    else:
        draw.line((x, y + 16, x + 46, y + 16), fill=color, width=5)
    text(draw, (x + 60, y), label, 24, MUTED)


def comparison():
    traces = figure("Chesbro: inflow comparison")["data"]
    data = [values(t) for t in traces]
    im, draw = canvas()
    text(draw, (155, 45), "Chesbro Reservoir: inflow comparison", 38)
    text(
        draw,
        (155, 100),
        "January 1–15, 2023 (UTC); 4-hour smoothing lag",
        24,
        MUTED,
    )
    legend(draw, 155, 155, "Raw water balance", "#BBBBBB")
    legend(draw, 555, 155, "6-hour centered mean", ORANGE)
    legend(draw, 1020, 155, "Revised inflow", BLUE)
    legend(draw, 1390, 155, "Upstream proxy", TEAL)
    point = axes(draw, -1200, 7200, [-1000, 0, 2000, 4000, 6000])
    line(draw, data[0], point, "#CCCCCC", 2)
    line(draw, data[4], point, TEAL, 3)
    line(draw, data[1], point, ORANGE, 4)
    line(draw, data[3], point, BLUE, 4)
    text(
        draw,
        (155, 975),
        "Source: reservoir example notebook. "
        "The upstream series is a proxy for comparison.",
        24,
        MUTED,
    )
    im.save(OUT / "inflow-comparison.png")


def uncertainty():
    traces = figure("Chesbro: model-based 95% inflow uncertainty")["data"]
    lower, upper, revised = [values(traces[i]) for i in (3, 4, 5)]
    im, draw = canvas()
    text(draw, (155, 45), "Chesbro Reservoir: revised inflow uncertainty", 38)
    text(
        draw,
        (155, 100),
        "January 1–15, 2023 (UTC); 4-hour smoothing lag",
        24,
        MUTED,
    )
    legend(draw, 155, 155, "Revised inflow", BLUE)
    legend(draw, 555, 155, "Model-based 95% interval", "#D3DFEB", True)
    point = axes(draw, -300, 2800, [0, 500, 1000, 1500, 2000, 2500])
    segment = []

    def polygon(indices):
        if len(indices) > 1:
            pts = [point(i, upper[i]) for i in indices] + [
                point(i, lower[i]) for i in reversed(indices)
            ]
            draw.polygon(pts, fill="#D3DFEB")

    for i, (lo, hi) in enumerate(zip(lower, upper, strict=True)):
        if math.isfinite(lo) and math.isfinite(hi):
            segment.append(i)
        else:
            polygon(segment)
            segment = []
    polygon(segment)
    line(draw, lower, point, "#9FB6CC", 2)
    line(draw, upper, point, "#9FB6CC", 2)
    line(draw, revised, point, BLUE, 4)
    text(
        draw,
        (155, 975),
        "Pointwise intervals from model covariance; "
        "not independently calibrated accuracy bounds.",
        24,
        MUTED,
    )
    im.save(OUT / "inflow-uncertainty.png")


def workflow():
    im, draw = canvas(520)
    text(draw, (85, 45), "KalmanFlow estimation workflow", 38)
    boxes = [(85, 150, 535, 340), (675, 150, 1125, 340), (1265, 150, 1715, 340)]
    labels = [
        (
            "Storage + outflow",
            ["Clean and align measurements", "Use timezone-aware timestamps"],
        ),
        (
            "Kalman filter",
            ["Update the water-balance state", "Publish causal inflow estimates"],
        ),
        (
            "Fixed-lag RTS smoother",
            ["Apply fixed-lag RTS smoothing", "Replace earlier published values"],
        ),
    ]
    for box, label in zip(boxes, labels, strict=True):
        draw.rectangle(box, fill="white", outline="#888888", width=2)
        x, y = box[0] + 28, box[1] + 26
        text(draw, (x, y), label[0], 30)
        for j, detail in enumerate(label[1]):
            text(draw, (x, y + 65 + j * 32), detail, 23, MUTED)
    for x in (535, 1125):
        draw.line((x + 22, 245, x + 115, 245), fill="#666666", width=3)
        draw.polygon([(x + 115, 245), (x + 99, 235), (x + 99, 255)], fill="#666666")
    text(
        draw,
        (675, 382),
        "Optional uncertainty: standard deviations "
        "from filter and smoother covariance.",
        24,
        MUTED,
    )
    text(
        draw,
        (85, 455),
        "Net inflow is the residual in the supplied storage–outflow balance.",
        24,
        MUTED,
    )
    im.save(OUT / "kalmanflow-workflow.png")


if __name__ == "__main__":
    comparison()
    uncertainty()
    workflow()
