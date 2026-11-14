#!/usr/bin/env python3
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
"""Plot diagnostic outputs from graphFlow_photon_scattering.

The script reads only the CSV files written by `--output-dir`; it never runs
the C++ example. It writes figure1..figure4 in PNG, SVG and PDF to the output
directory. Matplotlib and NumPy are runtime-only dependencies of this script,
not of the C++ example or CTest.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_pairs(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0] != "key":
                values[row[0]] = row[1]
    return values


def read_outcomes(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 2 and row[0] != "event":
                values[row[0]] = int(row[1])
    return values


def read_histogram(path: Path) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 3 and row[0] != "event":
                counts[int(row[1])] += int(row[2])
    return dict(sorted(counts.items()))


def read_traces(path: Path) -> dict[int, list[dict[str, str]]]:
    traces: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 10 and row[0] != "id":
                traces[int(row[0])].append(
                    {
                        "step": int(row[1]),
                        "kind": row[2],
                        "scatter_count": int(row[3]),
                        "x": float(row[4]),
                        "y": float(row[5]),
                        "z": float(row[6]),
                        "ux": float(row[7]),
                        "uy": float(row[8]),
                        "uz": float(row[9]),
                    }
                )
    return dict(traces)


def read_exits(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 13 and row[0] != "id":
                rows.append(
                    {
                        "id": int(row[0]),
                        "event": row[1],
                        "scatter_count": int(row[2]),
                        "x": float(row[3]),
                        "y": float(row[4]),
                        "z": float(row[5]),
                        "ux": float(row[6]),
                        "uy": float(row[7]),
                        "uz": float(row[8]),
                    }
                )
    return rows


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def figure_paths(traces: dict[int, list[dict[str, str]]], metadata: dict[str, str], out: Path) -> None:
    thickness = float(metadata.get("thickness", "1"))
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.axhline(0.0, color="0.4", linewidth=1.0, label="slab faces")
    axis.axhline(thickness, color="0.4", linewidth=1.0)

    chosen: dict[str, tuple[int, list[dict[str, str]]]] = {}
    for photon_id, vertices in traces.items():
        terminal = vertices[-1]["kind"] if vertices else "missing"
        if terminal in {"transmitted", "reflected", "absorbed"}:
            previous = chosen.get(terminal)
            if previous is None or vertices[-1]["scatter_count"] > previous[1][-1]["scatter_count"]:
                chosen[terminal] = (photon_id, vertices)
    for terminal, (photon_id, vertices) in sorted(chosen.items()):
        z = [vertex["z"] for vertex in vertices]
        x = [vertex["x"] for vertex in vertices]
        axis.plot(x, z, marker="o", linewidth=1.2, markersize=4, label=f"id {photon_id} {terminal}")
        for vertex in vertices:
            if vertex["kind"] == "scatter":
                axis.scatter([vertex["x"]], [vertex["z"]], s=34, facecolors="none", edgecolors="black", zorder=4)

    scripted_z = [0.0, 0.2, 0.4, 0.3, 0.0]
    axis.plot(
        [0.0] * len(scripted_z),
        [value * thickness for value in scripted_z],
        linestyle="--",
        marker="s",
        color="black",
        linewidth=1.1,
        label="scripted 3-scatter reflected",
    )
    axis.set_xlabel("transverse x (projection, length units)")
    axis.set_ylabel("z (length units)")
    axis.set_title("Photon paths through the slab (x-z projection of 3D tracks)")
    axis.legend(fontsize=8, loc="best")
    save(fig, out, "figure1_paths")


def poisson_pmf(mean: float, k: int) -> float:
    return math.exp(-mean) * mean**k / math.factorial(k)


def figure_spectrum(histogram: dict[int, int], metadata: dict[str, str], out: Path) -> None:
    counts = histogram
    photons = int(metadata.get("photons", "0"))
    sigma_s = float(metadata.get("sigma_s", "0"))
    thickness = float(metadata.get("thickness", "1"))
    mode = metadata.get("mode", "forward")
    mean = sigma_s * thickness

    if mode == "forward" and photons > 0:
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    else:
        fig, top = plt.subplots(figsize=(7.2, 4.4))
        bottom = None

    ks = sorted(counts)
    observed = [counts[k] for k in ks]
    top.bar(ks, observed, width=0.8, color="tab:blue", alpha=0.75, label="observed")
    if mode == "forward" and photons > 0:
        expected = [photons * poisson_pmf(mean, k) for k in ks]
        top.plot(ks, expected, color="tab:red", marker="o", linewidth=1.1, label=f"N*Poisson({mean:g})")
        if bottom is not None:
            residual = [
                (obs - exp) / math.sqrt(exp) if exp > 0 else 0.0 for obs, exp in zip(observed, expected)
            ]
            bottom.axhline(0.0, color="0.4", linewidth=1.0)
            bottom.bar(ks, residual, width=0.8, color="tab:green", alpha=0.7)
            bottom.set_ylabel("(obs-exp)/sqrt(exp)")
            bottom.set_xlabel("completed scatters k")
    else:
        top.set_xlabel("completed scatters k")
    top.set_ylabel("photons")
    top.set_title(f"Scatter-count distribution (mode={mode})")
    top.legend(fontsize=8)
    save(fig, out, "figure2_scatter_spectrum")


def figure_outcomes(
    outcomes: dict[str, int], histogram: dict[int, int], metadata: dict[str, str], out: Path
) -> None:
    photons = int(metadata.get("photons", "0"))
    events = ["transmitted", "reflected", "absorbed", "capped"]
    counts = [outcomes.get(event, 0) for event in events]
    fractions = [count / photons if photons else 0.0 for count in counts]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    left.bar(events, counts, color=["tab:blue", "tab:orange", "tab:red", "0.5"])
    left.set_ylabel("histories")
    left.set_title("Terminal outcomes")
    for index, count in enumerate(counts):
        left.text(index, count, str(count), ha="center", va="bottom", fontsize=8)
    left.tick_params(axis="x", rotation=20)

    uncollided = histogram.get(0, 0)
    probability = uncollided / photons if photons else 0.0
    error = math.sqrt(probability * (1.0 - probability) / photons) if photons else 0.0
    sigma_t = float(metadata.get("sigma_t", "0"))
    thickness = float(metadata.get("thickness", "1"))
    analytic = math.exp(-sigma_t * thickness)
    right.errorbar([0], [probability], yerr=[error], fmt="o", capsize=5, label="measured uncollided")
    right.axhline(analytic, color="tab:red", linestyle="--", label=f"exp(-sigma_t L)={analytic:.6g}")
    right.set_xlim(-0.6, 0.6)
    right.set_xticks([])
    right.set_ylabel("uncollided fraction")
    right.set_title("Uncollided transmission")
    right.legend(fontsize=8)
    right.text(
        0.02,
        0.02,
        f"transmitted={fractions[0]:.4f}\nreflected={fractions[1]:.4f}\n"
        f"absorbed={fractions[2]:.4f}\ncapped={fractions[3]:.4f}",
        transform=right.transAxes,
        fontsize=8,
        va="bottom",
    )
    save(fig, out, "figure3_outcomes")


def figure_detector(exits: list[dict[str, str]], metadata: dict[str, str], out: Path) -> None:
    distance = float(metadata.get("detector_distance", metadata.get("thickness", "1")))
    half_width = float(metadata.get("detector_half_width", str(2.0 * distance)))
    hit_x: list[float] = []
    hit_y: list[float] = []
    outside = 0
    for row in exits:
        if row["event"] not in {"transmitted", "reflected"}:
            continue
        if row["uz"] == 0.0:
            continue
        travel = distance / abs(row["uz"])
        x = row["x"] + travel * row["ux"]
        y = row["y"] + travel * row["uy"]
        hit_x.append(x)
        hit_y.append(y)
        if abs(x) > half_width or abs(y) > half_width:
            outside += 1

    fig, axis = plt.subplots(figsize=(5.6, 4.8))
    if hit_x:
        axis.hist2d(
            hit_x,
            hit_y,
            bins=40,
            range=[[-half_width, half_width], [-half_width, half_width]],
            cmap="viridis",
        )
        title = f"Exit-hit density at detector (outside={outside}/{len(hit_x)})"
    else:
        axis.text(0.5, 0.5, "no face exits in exits.csv", ha="center", va="center", transform=axis.transAxes)
        title = "Exit-hit density (no data)"
    axis.set_xlabel("detector x (length units)")
    axis.set_ylabel("detector y (length units)")
    axis.set_title(title)
    save(fig, out, "figure4_detector")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="output directory written by --output-dir")
    parser.add_argument("--out", type=Path, help="plot directory (default: INPUT/plots)")
    args = parser.parse_args()

    output = args.out if args.out is not None else args.input / "plots"
    output.mkdir(parents=True, exist_ok=True)

    metadata = read_pairs(args.input / "metadata.csv")
    outcomes = read_outcomes(args.input / "outcomes.csv")
    histogram = read_histogram(args.input / "scatter_histogram.csv")
    traces = read_traces(args.input / "traces.csv")
    exits = read_exits(args.input / "exits.csv")

    figure_paths(traces, metadata, output)
    figure_spectrum(histogram, metadata, output)
    figure_outcomes(outcomes, histogram, metadata, output)
    figure_detector(exits, metadata, output)
    print(f"wrote photon plots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
