from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path


def _simple_png(matrix, path: Path):
    """Tiny dependency-free fallback for smoke tests; matplotlib draws labels in normal runs."""
    n = len(matrix)
    scale = 24
    data = bytearray()
    flat = [float(x) for row in matrix for x in row if x is not None and not math.isnan(float(x))]
    low, high = (min(flat), max(flat)) if flat else (0.0, 1.0)
    for y in range(n * scale):
        data.append(0)
        for x in range(n * scale):
            value = matrix[y // scale][x // scale]
            t = (float(value) - low) / (high - low) if value is not None and high > low else 0.0
            data.extend((int(255 * t), int(100 + 100 * t), int(255 * (1 - t))))
    def chunk(tag, value):
        return struct.pack(">I", len(value)) + tag + value + struct.pack(">I", zlib.crc32(tag + value))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n * scale, n * scale, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(data))) + chunk(b"IEND", b"")
    path.write_bytes(png)


def heatmap(summary_rows: list[dict], sizes: tuple[int, ...], metric: str,
            path: Path, title: str) -> None:
    lookup = {(int(r["g_E"]), int(r["g_R"])): r for r in summary_rows}
    matrix = [[_number(lookup.get((e, r), {}).get(metric)) for r in sizes] for e in sizes]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _simple_png(matrix, path)
        return
    import numpy as np
    values = np.array([[x if x is not None else math.nan for x in row] for row in matrix])
    fig, ax = plt.subplots(figsize=(max(6, len(sizes) * 0.9), max(5, len(sizes) * 0.8)))
    image = ax.imshow(values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(sizes)), labels=sizes)
    ax.set_yticks(range(len(sizes)), labels=sizes)
    ax.set_xlabel("Retrieval chunk tokens (g_R)")
    ax.set_ylabel("Extraction chunk tokens (g_E)")
    ax.set_title(title)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, "NA" if value is None else f"{value:.2f}",
                    ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _number(value):
    try:
        return float(value) if value not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None
