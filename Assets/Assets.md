# Asset Overview

The `Assets/assets` directory bundles CC0 Kenney packs that the Tkinter prototype loads at runtime:

- `catalog/kenney_roguelike-modern-city` and `catalog/kenney_roguelike-indoors` – tile atlases for exterior and interior zones.
- `tiles/` – frequently used tiles pre-extracted from the catalog for quick reference.
- `props/vehicles/PNG/Characters` – simple vehicle driver sprites that act as a legacy fallback.
- `agents/kenney_toon-characters-1/*/PNG/Poses` – high-resolution Toon Character poses used for animated citizens.

The GUI automatically discovers walk animations under `agents/` and assigns them to residents. Additions to these folders become available without code changes as long as they follow the same structure.

> Tip: drop themed props or signage into `tiles/`/`catalog/` and wire them into the POI definitions (see `smallville_gui.py`) to grant new tags (e.g., `"music"`, `"research"`) that enrich agent interests and reflection logs.
