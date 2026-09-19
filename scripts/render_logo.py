#!/usr/bin/env python3
"""Regenerate `src/indexter/_logo.py` from `indexter.png`.

The banner shown by `indexter init` is pixel art: the logo box-sampled down
and drawn with quadrant characters (2x2 pixels per terminal cell). It is
rendered here, offline, and embedded as a string constant, so the package
needs no image library and ships no PNG. Standard library only. Run it via
`just logo` after the logo changes. Not part of the `indexter` package:
excluded from the sdist (`source-exclude`) and from coverage.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "indexter.png"
TARGET = ROOT / "src" / "indexter" / "_logo.py"

COLUMNS = 52
INK_THRESHOLD = 0.45
QUADRANTS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"
SPARKLES = "✦✧"  # the PNG's star blobs are redrawn as these glyphs


def load_ink(path: Path) -> tuple[int, int, list[list[float]]]:
    """Decode an 8-bit, non-interlaced RGBA PNG into `ink[y][x]` in 0..1:
    how opaque and dark each pixel is."""
    data = path.read_bytes()
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length

    raw, bpp, stride = zlib.decompress(idat), 4, width * 4
    rows: list[bytearray] = []
    prev, offset = bytearray(stride), 0
    for _ in range(height):
        ftype = raw[offset]
        line = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if ftype == 1:
                line[x] = (line[x] + a) & 255
            elif ftype == 2:
                line[x] = (line[x] + b) & 255
            elif ftype == 3:
                line[x] = (line[x] + (a + b) // 2) & 255
            elif ftype == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line

    ink = [
        [(r[x * 4 + 3] / 255) * (1 - (r[x * 4] + r[x * 4 + 1] + r[x * 4 + 2]) / 765) for x in range(width)]
        for r in rows
    ]
    return width, height, ink


def sample(ink: list[list[float]], width: int, height: int, out_w: int, out_h: int) -> list[list[bool]]:
    """Box-sample `ink` down to `out_w` x `out_h` on/off pixels."""
    out = []
    for py in range(out_h):
        y0 = int(py * height / out_h)
        y1 = max(y0 + 1, int((py + 1) * height / out_h))
        row = []
        for px in range(out_w):
            x0 = int(px * width / out_w)
            x1 = max(x0 + 1, int((px + 1) * width / out_w))
            cells = [ink[y][x] for y in range(y0, y1) for x in range(x0, x1)]
            row.append(sum(cells) / len(cells) > INK_THRESHOLD)
        out.append(row)
    return out


def pixel_art(path: Path, columns: int) -> list[str]:
    width, height, ink = load_ink(path)
    rows = round(columns * height / width / 2)
    px = sample(ink, width, height, columns * 2, rows * 2)

    art = []
    for r in range(rows):
        top, bottom = px[2 * r], px[2 * r + 1]
        art.append(
            "".join(
                QUADRANTS[top[2 * c] + 2 * top[2 * c + 1] + 4 * bottom[2 * c] + 8 * bottom[2 * c + 1]]
                for c in range(columns)
            )
        )
    while art and not art[0].strip():
        art.pop(0)
    while art and not art[-1].strip():
        art.pop()

    # Every row above the glasses' top bar holds only the PNG's star blobs:
    # swap each blob for a single sparkle glyph.
    bar = next((i for i, line in enumerate(art) if line.count("▄") > columns // 4), None)
    if bar is None:
        raise ValueError(
            f"{path.name}: no glasses bar found, so the sparkle rows cannot be told from the art. "
            "Ink weights alpha by darkness, so light-on-transparent art samples as blank."
        )
    stars: list[tuple[int, int]] = []
    for y in range(bar):
        cells = list(art[y])
        for x, ch in enumerate(cells):
            if ch == " ":
                continue
            cells[x] = " "
            if all(abs(x - sx) > 2 or abs(y - sy) > 1 for sx, sy in stars):
                stars.append((x, y))
        art[y] = "".join(cells)
    for n, (x, y) in enumerate(stars):
        art[y] = art[y][:x] + SPARKLES[n % 2] + art[y][x + 1 :]
    return [line.rstrip() for line in art]


def render_module(art: list[str]) -> str:
    # json, not repr: double quotes, so the result is already ruff-formatted.
    body = "".join(f"    {json.dumps(line, ensure_ascii=False)},\n" for line in art)
    return (
        '"""The `indexter init` banner art. Generated by `scripts/render_logo.py` from\n'
        '`indexter.png` -- do not edit by hand; run `just logo`."""\n'
        "\n"
        "LOGO: tuple[str, ...] = (\n"
        f"{body}"
        ")\n"
    )


def main() -> None:
    # The art is quadrant glyphs and sparkles, so the locale encoding will not do.
    TARGET.write_text(render_module(pixel_art(SOURCE, COLUMNS)), encoding="utf-8")
    print(f"Wrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
