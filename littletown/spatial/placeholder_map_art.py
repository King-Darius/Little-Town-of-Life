"""Utility helpers to synthesise the placeholder town map artwork.

The previous iteration of the project shipped a binary ``placeholder_map.png``
so that the SAM2 pipeline had something to segment out-of-the-box.  Git-based
review tooling struggled with that binary blob which meant contributors could
not open pull requests.  To keep the workflow source-friendly we now generate
an equivalent illustrative map procedurally from the JSON layout that already
lives in the repository.

The :func:`ensure_placeholder_image` helper renders the layout grid using the
same colour palette expected by :mod:`littletown.spatial.sam`.  This keeps the
classification heuristics stable while allowing the PNG to be recreated on any
machine without committing binary assets to the repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

# The colour palette mirrors ``_SegmentSpec`` definitions in ``sam.py`` so that
# SAM-driven segmentation heuristics still work as expected.
_SEGMENT_COLOURS: Mapping[str, Tuple[int, int, int]] = {
    "T": (110, 110, 118),  # transport
    "R": (140, 140, 148),  # road
    "H": (164, 196, 252),  # residential
    "C": (250, 228, 176),  # civic
    "P": (124, 188, 108),  # park
    "O": (247, 177, 84),   # commerce
    "L": (232, 197, 255),  # leisure
    "M": (255, 204, 204),  # health / medical
    "I": (182, 153, 255),  # light industry
    "A": (216, 168, 255),  # AI lab / research
}

_BACKGROUND_COLOUR = (240, 240, 240)
_GRID_OUTLINE_COLOUR = (80, 80, 80, 96)  # Semi-transparent outline to hint roads


def _load_layout(layout_path: Path) -> Sequence[str]:
    data = json.loads(layout_path.read_text())
    grid = data.get("grid", {})
    rows = grid.get("rows")
    if not rows:
        raise ValueError(f"Layout file {layout_path} does not contain grid rows")
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("Layout rows must all be the same length")
    return rows


def _paint_grid(rows: Sequence[str], *, cell_size: int) -> Image.Image:
    width = len(rows[0]) * cell_size
    height = len(rows) * cell_size
    image = Image.new("RGB", (width, height), _BACKGROUND_COLOUR)
    draw = ImageDraw.Draw(image, "RGBA")

    for y, row in enumerate(rows):
        for x, code in enumerate(row):
            colour = _SEGMENT_COLOURS.get(code, _BACKGROUND_COLOUR)
            left = x * cell_size
            top = y * cell_size
            right = left + cell_size
            bottom = top + cell_size
            draw.rectangle((left, top, right, bottom), fill=colour)

    # Light grid overlay so the generated artwork matches the illustrated style.
    for x in range(len(rows[0]) + 1):
        draw.line(
            (
                x * cell_size,
                0,
                x * cell_size,
                height,
            ),
            fill=_GRID_OUTLINE_COLOUR,
            width=1,
        )
    for y in range(len(rows) + 1):
        draw.line(
            (
                0,
                y * cell_size,
                width,
                y * cell_size,
            ),
            fill=_GRID_OUTLINE_COLOUR,
            width=1,
        )
    return image


def ensure_placeholder_image(image_path: Path, layout_path: Path, *, cell_size: int = 32) -> Path:
    """Ensure the placeholder PNG exists, regenerating it from ``layout_path``.

    Parameters
    ----------
    image_path:
        Destination where the PNG should be written.
    layout_path:
        JSON file containing the ``grid.rows`` data used by the simulation.
    cell_size:
        Size of each tile cell in pixels. The default matches the GUI renderer.
    """

    image_path.parent.mkdir(parents=True, exist_ok=True)

    if image_path.exists():
        return image_path

    rows = _load_layout(layout_path)
    image = _paint_grid(rows, cell_size=cell_size)
    image.save(image_path, format="PNG")
    return image_path


__all__ = ["ensure_placeholder_image"]
