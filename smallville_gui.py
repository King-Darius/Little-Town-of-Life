"""Tkinter simulation app for the Smallville+++ concept.

This module provides a one-click friendly graphical simulation that renders a
small town map, animates resident agents, and exposes basic controls so that
non-technical users can explore the prototype without additional setup.  The
implementation intentionally leans on the asset packs that ship with the
repository so that the prototype feels grounded in the provided visual design.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk
except ModuleNotFoundError as exc:  # pragma: no cover - handled in run_smallville.py
    raise SystemExit(
        "Pillow is required to run the Smallville GUI. Install dependencies "
        "or launch the app using run_smallville.py so that they are installed "
        "automatically."
    ) from exc


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


ROOT_DIR = Path(__file__).resolve().parent
ASSET_ROOT = ROOT_DIR / "Assets" / "assets"


def load_optional_config() -> Dict[str, object]:
    """Load optional simulation overrides from ``config.json`` if present."""

    config_path = ROOT_DIR / "config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Asset management
# ---------------------------------------------------------------------------


TileName = str


class AssetManager:
    """Loads and caches Tk-compatible images for tiles and agent sprites."""

    def __init__(self, asset_root: Path, cell_size: int) -> None:
        self.asset_root = asset_root
        self.cell_size = cell_size
        self._tile_cache: Dict[TileName, ImageTk.PhotoImage] = {}
        self._agent_cache: Dict[str, List[ImageTk.PhotoImage]] = {}
        self._fallback_tiles: Dict[TileName, str] = {
            "road": "#6c7a89",
            "residential": "#9ad0c2",
            "civic": "#f4d35e",
            "education": "#90caf9",
            "commerce": "#f7a072",
            "industry": "#c792ea",
            "leisure": "#81c784",
            "health": "#ffccbc",
            "park": "#a5d6a7",
            "transport": "#b0bec5",
            "ai_lab": "#ce93d8",
        }

        # Curated subset of the Kenney tile pack shipped with the repository.
        tile_folder = asset_root / "catalog" / "kenney_roguelike-modern-city" / "Tiles"
        self._tile_sources: Dict[TileName, Path] = {
            "road": tile_folder / "tile_0414.png",
            "residential": tile_folder / "tile_0010.png",
            "civic": tile_folder / "tile_0053.png",
            "education": tile_folder / "tile_0027.png",
            "commerce": tile_folder / "tile_0198.png",
            "industry": tile_folder / "tile_0540.png",
            "leisure": tile_folder / "tile_0307.png",
            "health": tile_folder / "tile_0706.png",
            "park": tile_folder / "tile_0717.png",
            "transport": tile_folder / "tile_0328.png",
            "ai_lab": tile_folder / "tile_0831.png",
        }

        self._agent_sources: Dict[str, Tuple[Sequence[Path], str]] = {}
        self._agent_order: List[str] = []
        self._discover_agent_variants()
        if not self._agent_sources:
            self._register_legacy_agent_variants()

    def _slug(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or "agent"

    def _discover_agent_variants(self) -> None:
        agent_root = self.asset_root / "agents"
        if not agent_root.exists():
            return

        color_cycle = cycle(
            [
                "#1976d2",
                "#6a1b9a",
                "#00897b",
                "#f4511e",
                "#5d4037",
            ]
        )

        for pack_dir in sorted(agent_root.iterdir()):
            if not pack_dir.is_dir():
                continue
            for character_dir in sorted(pack_dir.iterdir()):
                poses_dir = character_dir / "PNG" / "Poses"
                if not poses_dir.exists():
                    continue
                walk_frames = sorted(poses_dir.glob("*walk[0-9].png"), key=lambda p: p.name)
                idle_frames = sorted(poses_dir.glob("*idle*.png"), key=lambda p: p.name)
                sources: List[Path] = []
                if walk_frames:
                    sources.extend(walk_frames[:4])
                if not sources and idle_frames:
                    sources.extend(idle_frames[:1])
                if not sources:
                    continue
                variant_name = self._slug(character_dir.name)
                if variant_name in self._agent_sources:
                    continue
                self._agent_sources[variant_name] = (sources, next(color_cycle))
                self._agent_order.append(variant_name)

    def _register_legacy_agent_variants(self) -> None:
        people_folder = (
            self.asset_root
            / "props"
            / "vehicles"
            / "PNG"
            / "Characters"
        )
        legacy_variants = {
            "citizen": [
                people_folder / "man.png",
                people_folder / "man_walk1.png",
                people_folder / "man_walk2.png",
            ],
            "citizen_f": [
                people_folder / "woman.png",
                people_folder / "woman_walk1.png",
                people_folder / "woman_walk2.png",
            ],
        }
        for name, sources in legacy_variants.items():
            self._agent_sources[name] = (sources, "#1976d2")
            self._agent_order.append(name)

    def _load_image(self, path: Path, fallback_color: str) -> ImageTk.PhotoImage:
        if path.exists():
            image = Image.open(path).convert("RGBA").resize(
                (self.cell_size, self.cell_size), Image.NEAREST
            )
        else:
            image = Image.new("RGBA", (self.cell_size, self.cell_size), fallback_color)
        return ImageTk.PhotoImage(image)

    def tile(self, name: TileName) -> ImageTk.PhotoImage:
        if name not in self._tile_cache:
            path = self._tile_sources.get(name)
            fallback = self._fallback_tiles.get(name, "#cccccc")
            image = self._load_image(path if path else Path(), fallback)
            self._tile_cache[name] = image
        return self._tile_cache[name]

    def agent_frames(self, variant: str) -> List[ImageTk.PhotoImage]:
        if variant not in self._agent_cache:
            sources, fallback_color = self._agent_sources.get(
                variant, ([], "#1976d2")
            )
            fallback = Image.new("RGBA", (self.cell_size, self.cell_size), fallback_color)
            if not sources:
                frames = [ImageTk.PhotoImage(fallback)]
            else:
                frames = [self._load_image(path, fallback_color) for path in sources]
            self._agent_cache[variant] = frames
        return self._agent_cache[variant]

    @property
    def agent_variants(self) -> List[str]:
        return list(self._agent_order)


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------


NEEDS: Sequence[str] = (
    "hunger",
    "thirst",
    "energy",
    "hygiene",
    "safety",
    "money_pressure",
    "belonging",
    "esteem",
    "competence",
    "autonomy",
    "fun",
    "creativity",
    "curiosity",
    "civic_impact",
    "legacy",
)


@dataclass
class PointOfInterest:
    name: str
    kind: str
    xy: Tuple[int, int]
    satisfies: Dict[str, float]
    capacity: int = 4
    open_hours: Tuple[int, int] = (6, 22)
    cost: float = 0.0
    tags: Tuple[str, ...] = ()

    def is_open(self, hour: int) -> bool:
        if self.open_hours[0] <= self.open_hours[1]:
            return self.open_hours[0] <= hour < self.open_hours[1]
        # Overnight wrap-around (e.g., bars)
        return hour >= self.open_hours[0] or hour < self.open_hours[1]

    def crowding_penalty(self, occupants: int) -> float:
        if self.capacity <= 0:
            return 1.0
        ratio = occupants / self.capacity
        if ratio <= 1:
            return 1.0
        return 1.0 + min(1.5, (ratio - 1) * 0.75)


class TownMap:
    """Grid definition and path finding for the town."""

    layout: Sequence[str] = (
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        "TRRRRRRHHHHHHRRRRRRCCCCCCTTTTTT",
        "TRRRRRRHHHHHHRRRRRRCCCCCCTTTTTT",
        "TRRRRRRHHHHHHRRRRRRCCCCCCTTTTTT",
        "TRRRRRRRRRRRRRRRRRRRRRRRRRTTTTT",
        "TPPPPPPPPPPPPPPPPPPPPPPPPPPTTTT",
        "TPPPPPPPPPPPPPPPPPPPPPPPPPPTTTT",
        "TRRRRRRRRRRRRRRRRRRRRRRRRRTTTTT",
        "TRRRRRROOOOOORRRRRRSSSSSSTTTTT",
        "TRRRRRROOOOOORRRRRRSSSSSSTTTTT",
        "TRRRRRROOOOOORRRRRRSSSSSSTTTTT",
        "TRRRRRROOOOOORRRRRRSSSSSSTTTTT",
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        "TLLLLLLLLLLLLTTTTTMMMMMMMIIIIII",
        "TLLLLLLLLLLLLTTTTTMMMMMMMIIIIII",
        "TLLLLLLLLLLLLTTTTTMMMMMMMIIIIII",
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        "TTTTTTTTTTTTTAAATTTTTTTTTTTTTTT",
        "TTTTTTTTTTTTTAAATTTTTTTTTTTTTTT",
        "TTTTTTTTTTTTTAAATTTTTTTTTTTTTTT",
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
    )

    tile_map: Dict[str, TileName] = {
        "T": "transport",
        "R": "road",
        "H": "residential",
        "C": "civic",
        "P": "park",
        "O": "commerce",
        "S": "commerce",
        "L": "leisure",
        "M": "health",
        "I": "industry",
        "A": "ai_lab",
    }

    walkable: Sequence[TileName] = (
        "transport",
        "road",
        "residential",
        "civic",
        "park",
        "commerce",
        "leisure",
        "health",
        "industry",
        "ai_lab",
    )

    def __init__(self) -> None:
        self.height = len(self.layout)
        self.width = len(self.layout[0]) if self.height else 0
        self.tiles: List[List[TileName]] = []
        for row in self.layout:
            self.tiles.append([self.tile_map.get(ch, "park") for ch in row])

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def tile(self, x: int, y: int) -> TileName:
        return self.tiles[y][x]

    def is_walkable(self, x: int, y: int) -> bool:
        return self.tile(x, y) in self.walkable

    def neighbors(self, x: int, y: int) -> Iterable[Tuple[int, int]]:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny) and self.is_walkable(nx, ny):
                yield nx, ny

    def find_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]

        queue: List[Tuple[int, int]] = [start]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        for current in queue:
            if current == goal:
                break
            for neighbor in self.neighbors(*current):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    queue.append(neighbor)

        if goal not in came_from:
            return [start]

        path: List[Tuple[int, int]] = [goal]
        while path[-1] != start:
            prev = came_from[path[-1]]
            if prev is None:
                break
            path.append(prev)
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Personality and needs
# ---------------------------------------------------------------------------


@dataclass
class Personality:
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    @classmethod
    def random(cls) -> "Personality":
        return cls(
            openness=random.uniform(0.3, 0.9),
            conscientiousness=random.uniform(0.2, 0.85),
            extraversion=random.uniform(0.2, 0.95),
            agreeableness=random.uniform(0.4, 0.95),
            neuroticism=random.uniform(0.1, 0.7),
        )


NEED_TRAIT_WEIGHTS: Dict[str, str] = {
    "belonging": "extraversion",
    "esteem": "extraversion",
    "fun": "extraversion",
    "creativity": "openness",
    "curiosity": "openness",
    "competence": "conscientiousness",
    "autonomy": "openness",
    "civic_impact": "agreeableness",
    "legacy": "conscientiousness",
    "money_pressure": "conscientiousness",
    "hygiene": "conscientiousness",
    "safety": "neuroticism",
    "energy": "neuroticism",
    "hunger": "neuroticism",
    "thirst": "neuroticism",
}

NEED_DECAY_RATES: Dict[str, float] = {
    "hunger": 0.018,
    "thirst": 0.020,
    "energy": 0.012,
    "hygiene": 0.009,
    "safety": 0.004,
    "money_pressure": 0.003,
    "belonging": 0.006,
    "esteem": 0.005,
    "competence": 0.0045,
    "autonomy": 0.004,
    "fun": 0.007,
    "creativity": 0.005,
    "curiosity": 0.005,
    "civic_impact": 0.0035,
    "legacy": 0.0025,
}


@dataclass
class NeedState:
    values: Dict[str, float]
    decay_rates: Dict[str, float]
    recent_history: deque = field(default_factory=lambda: deque(maxlen=24))

    @classmethod
    def create(cls) -> "NeedState":
        values = {name: random.uniform(0.12, 0.38) for name in NEEDS}
        decay = dict(NEED_DECAY_RATES)
        return cls(values, decay)

    def tick(self, dt: float) -> None:
        for key, rate in self.decay_rates.items():
            self.values[key] = min(1.0, self.values[key] + rate * dt)
        snapshot = {k: round(v, 3) for k, v in self.values.items()}
        self.recent_history.append(snapshot)

    def satisfy(self, deltas: Dict[str, float]) -> None:
        for key, amount in deltas.items():
            if key in self.values:
                self.values[key] = max(0.0, self.values[key] - amount)

    def most_pressing(self) -> Tuple[str, float]:
        need = max(self.values.items(), key=lambda item: item[1])
        return need

    def composite_pressure(self, weights: Dict[str, float]) -> float:
        return sum(self.values.get(name, 0.0) * weight for name, weight in weights.items())


class KnowledgeBase:
    """Provides short, wholesome facts for agent conversations."""

    def __init__(self) -> None:
        brief = (ROOT_DIR / "docs" / "Smallville+++_Implementation_Brief.md").read_text()
        sentences = [
            line.strip(" >")
            for line in brief.splitlines()
            if len(line.split()) > 4 and not line.startswith("##")
        ]
        self.facts = sentences[:200] or [
            "Community thrives when neighbours check in on each other.",
            "Daily routines that balance rest, work, and play keep citizens energised.",
            "Town halls can host civic idea jams every Thursday night.",
        ]

    def random_fact(self) -> str:
        return random.choice(self.facts)


@dataclass
class AffectState:
    """Pleasantness (valence), energy (arousal), and control (dominance)."""

    valence: float
    arousal: float
    dominance: float

    @classmethod
    def random(cls) -> "AffectState":
        return cls(
            valence=random.uniform(-0.1, 0.3),
            arousal=random.uniform(-0.2, 0.2),
            dominance=random.uniform(-0.1, 0.2),
        )

    def adjust(self, dv: float, da: float, dd: float) -> None:
        self.valence = max(-1.0, min(1.0, self.valence + dv))
        self.arousal = max(-1.0, min(1.0, self.arousal + da))
        self.dominance = max(-1.0, min(1.0, self.dominance + dd))

    def as_inputs(self) -> Tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)


@dataclass
class InterestProfile:
    """Tracks a lightweight skill/interest vector per agent."""

    topics: Dict[str, float]

    @classmethod
    def default(cls) -> "InterestProfile":
        base_topics = {
            "music": random.uniform(0.1, 0.6),
            "art": random.uniform(0.1, 0.6),
            "writing": random.uniform(0.1, 0.6),
            "sports": random.uniform(0.1, 0.6),
            "civic": random.uniform(0.1, 0.6),
            "coding": random.uniform(0.1, 0.6),
            "gardening": random.uniform(0.1, 0.6),
        }
        return cls(base_topics)

    def boost(self, tag: str, amount: float) -> None:
        current = self.topics.get(tag, 0.0)
        self.topics[tag] = max(0.0, min(1.0, current + amount))

    def affinity(self, tags: Sequence[str]) -> float:
        if not tags:
            return 0.0
        return sum(self.topics.get(tag, 0.0) for tag in tags) / len(tags)


@dataclass
class Memory:
    timestamp: int
    content: str
    importance: float
    tags: Tuple[str, ...] = ()


class MemoryStream:
    """Implements a simple generative-agents style memory buffer."""

    def __init__(self, *, reflection_threshold: float = 1.2) -> None:
        self.events: List[Memory] = []
        self.reflection_threshold = reflection_threshold
        self._recent_importance = 0.0

    def record(self, memory: Memory) -> None:
        self.events.append(memory)
        self._recent_importance += memory.importance

    def recent(self, limit: int = 6) -> List[Memory]:
        return self.events[-limit:]

    def should_reflect(self) -> bool:
        return self._recent_importance >= self.reflection_threshold

    def reflect(self, agent_name: str, knowledge: KnowledgeBase, tick: int) -> str:
        topics = [tag for mem in self.recent(5) for tag in mem.tags]
        if topics:
            dominant = max(set(topics), key=topics.count)
        else:
            dominant = "community"
        insight = knowledge.random_fact()
        summary = (
            f"{agent_name} journals about {dominant} after a thoughtful day: {insight}"
        )
        self.record(Memory(timestamp=tick, content=summary, importance=0.4, tags=(dominant,)))
        self._recent_importance = 0.0
        return summary


class LiquidCore:
    """Tiny liquid neural network core inspired by Liquid AI research."""

    def __init__(self, seed: Optional[int] = None, size: int = 8) -> None:
        self.size = size
        self.state = [0.0 for _ in range(size)]
        rng = random.Random(seed)
        input_dim = 8
        self.weights_input = [
            [rng.uniform(-0.25, 0.25) for _ in range(input_dim)] for _ in range(size)
        ]
        self.weights_recurrent = [
            [rng.uniform(-0.18, 0.18) for _ in range(size)] for _ in range(size)
        ]
        self.bias = [rng.uniform(-0.05, 0.05) for _ in range(size)]
        self.dt = 0.1
        self.tau = [rng.uniform(0.6, 1.4) for _ in range(size)]

    def advance(self, inputs: Sequence[float]) -> Dict[str, object]:
        padded_inputs = list(inputs)[:8]
        while len(padded_inputs) < 8:
            padded_inputs.append(0.0)
        new_state: List[float] = [0.0] * self.size
        for i in range(self.size):
            total = self.bias[i]
            total += sum(
                w * x for w, x in zip(self.weights_input[i], padded_inputs)
            )
            total += sum(
                w * h for w, h in zip(self.weights_recurrent[i], self.state)
            )
            z = math.tanh(total)
            tau = max(0.3, self.tau[i])
            new_h = self.state[i] + self.dt * (z - self.state[i]) / tau
            new_state[i] = max(-1.5, min(1.5, new_h))
        self.state = new_state
        explore_gain = 0.2 + 0.4 * ((self.state[0] + 1.5) / 3.0)
        speech_gain = 0.2 + 0.4 * ((self.state[1] + 1.5) / 3.0)
        satisfy_mod = {
            "fun": 1.0 + 0.25 * self.state[2],
            "creativity": 1.0 + 0.25 * self.state[3],
            "belonging": 1.0 + 0.2 * self.state[4],
            "competence": 1.0 + 0.2 * self.state[5],
        }
        return {
            "explore_gain": max(0.05, min(0.85, explore_gain)),
            "speech_gain": max(0.05, min(0.85, speech_gain)),
            "satisfy_mod": satisfy_mod,
        }

    def snapshot(self) -> Tuple[float, ...]:
        return tuple(round(v, 3) for v in self.state)

@dataclass
class Agent:
    name: str
    position: Tuple[int, int]
    personality: Personality
    needs: NeedState
    sprite_variant: str
    frames: List[ImageTk.PhotoImage]
    affect: AffectState
    interests: InterestProfile
    memory: MemoryStream
    liquid_core: LiquidCore
    cash: float = 120.0
    home_xy: Optional[Tuple[int, int]] = None
    job: str = "Freelancer"
    relations: Dict[str, float] = field(default_factory=dict)
    life_plan: List[str] = field(default_factory=list)
    facing: str = "down"
    target: Optional[PointOfInterest] = None
    path: List[Tuple[int, int]] = field(default_factory=list)
    frame_index: int = 0
    last_step_time: float = field(default_factory=time.monotonic)
    cooldown: int = 0
    last_reflection_tick: int = 0

    def core_inputs(self, hour: int, occupancy: Counter) -> Sequence[float]:
        pressing_need, pressing_value = self.needs.most_pressing()
        crowd_level = min(1.0, occupancy.get(self.position, 0) / 4.0)
        return (
            pressing_value * 2 - 1,
            self.needs.values.get("belonging", 0.0) * 2 - 1,
            self.needs.values.get("fun", 0.0) * 2 - 1,
            self.needs.values.get("competence", 0.0) * 2 - 1,
            math.sin(math.tau * hour / 24.0),
            math.cos(math.tau * hour / 24.0),
            crowd_level * 2 - 1,
            self.affect.valence,
        )

    def _trait_bias(self, need: str) -> float:
        trait_name = NEED_TRAIT_WEIGHTS.get(need)
        if not trait_name:
            return 1.0
        trait_value = getattr(self.personality, trait_name)
        return 0.65 + trait_value * 0.7

    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _update_facing(self, next_pos: Tuple[int, int]) -> None:
        dx = next_pos[0] - self.position[0]
        dy = next_pos[1] - self.position[1]
        if abs(dx) > abs(dy):
            self.facing = "right" if dx > 0 else "left"
        elif dy != 0:
            self.facing = "down" if dy > 0 else "up"

    def _advance_animation(self) -> None:
        now = time.monotonic()
        if now - self.last_step_time > 0.25:
            self.frame_index = (self.frame_index + 1) % len(self.frames)
            self.last_step_time = now

    def _record_memory(
        self, tick: int, description: str, importance: float, tags: Tuple[str, ...]
    ) -> None:
        self.memory.record(Memory(timestamp=tick, content=description, importance=importance, tags=tags))

    def _update_life_plan(self, rng: random.Random) -> None:
        pressing_need, _ = self.needs.most_pressing()
        top_interest = max(self.interests.topics.items(), key=lambda item: item[1])[0]
        self.life_plan = [
            f"Rebalance {pressing_need}",
            f"Advance {top_interest} practice",
            f"Check in with neighbours",
        ]
        if rng.random() < 0.3:
            self.life_plan.append("Sketch ideas for civic projects")

    def decide_target(
        self,
        world: TownMap,
        pois: Sequence[PointOfInterest],
        hour: int,
        rng: random.Random,
        occupancy: Counter,
        core_mod: Dict[str, object],
    ) -> Optional[PointOfInterest]:
        satisfy_mod = core_mod.get("satisfy_mod", {})
        explore_gain = core_mod.get("explore_gain", 0.2)
        best_score = -math.inf
        best_choice: Optional[PointOfInterest] = None
        pressing_need, pressing_value = self.needs.most_pressing()
        for poi in pois:
            if not poi.is_open(hour):
                continue
            if poi.cost > 0 and self.cash < poi.cost * 0.8:
                continue
            distance = self._distance(self.position, poi.xy)
            distance_penalty = 1 + 0.2 * distance
            satisfaction = 0.0
            for need, level in self.needs.values.items():
                relief = poi.satisfies.get(need, 0.0)
                if relief <= 0:
                    continue
                relief *= satisfy_mod.get(need, 1.0)
                weight = level * self._trait_bias(need)
                if need == pressing_need:
                    weight *= 1.35
                satisfaction += weight * relief
            interest_bonus = 1.0 + 0.35 * self.interests.affinity(poi.tags)
            affect_bonus = 1.0 + 0.25 * self.affect.valence + 0.15 * self.affect.arousal
            crowd_penalty = poi.crowding_penalty(occupancy.get(poi.xy, 0))
            if poi.cost >= 0:
                money_penalty = 1.0 + (poi.cost / max(1.0, self.cash + 1.0))
            else:
                money_penalty = max(0.6, 1.0 + (poi.cost / max(1.0, self.cash + 1.0)))
            satisfaction = (
                satisfaction
                * interest_bonus
                * affect_bonus
                / (distance_penalty * crowd_penalty * money_penalty)
            )
            satisfaction += rng.uniform(-0.05, 0.05)
            if satisfaction > best_score:
                best_score = satisfaction
                best_choice = poi
        if best_choice is not None and pressing_value < 0.4 and rng.random() < explore_gain:
            best_choice = rng.choice(pois)
        self.target = best_choice
        if best_choice is None:
            self.path = [self.position]
        else:
            self.path = world.find_path(self.position, best_choice.xy)
        return best_choice

    def reflect_if_needed(
        self, tick: int, knowledge: KnowledgeBase, rng: random.Random
    ) -> Optional[str]:
        if not self.memory.should_reflect():
            return None
        if tick - self.last_reflection_tick < 18:
            return None
        self.last_reflection_tick = tick
        self._update_life_plan(rng)
        reflection = self.memory.reflect(self.name, knowledge, tick)
        self.affect.adjust(0.1, -0.05, 0.05)
        return reflection

    def step(
        self,
        world: TownMap,
        pois: Sequence[PointOfInterest],
        hour: int,
        knowledge: KnowledgeBase,
        rng: random.Random,
        occupancy: Counter,
        tick: int,
    ) -> List[str]:
        events: List[str] = []
        self.needs.tick(1.0)
        if self.cooldown > 0:
            self.cooldown -= 1

        core_mod = self.liquid_core.advance(self.core_inputs(hour, occupancy))

        if not self.target or not self.target.is_open(hour):
            choice = self.decide_target(world, pois, hour, rng, occupancy, core_mod)
            if choice:
                events.append(f"{self.name} plots a visit to {choice.name}.")

        if len(self.path) > 1:
            next_pos = self.path[1]
            self._update_facing(next_pos)
            self.position = next_pos
            self.path = self.path[1:]
            self._advance_animation()
        else:
            self.frame_index = 0

        if self.target and self.position == self.target.xy:
            satisfy = {
                need: amount * core_mod.get("satisfy_mod", {}).get(need, 1.0)
                for need, amount in self.target.satisfies.items()
            }
            self.needs.satisfy(satisfy)
            if self.target.cost != 0:
                if self.target.cost > 0:
                    self.cash = max(0.0, self.cash - self.target.cost)
                else:
                    self.cash += abs(self.target.cost)
                self.needs.values["money_pressure"] = max(
                    0.0, self.needs.values.get("money_pressure", 0.0) - 0.12
                )
            self.affect.adjust(0.05, -0.05, 0.02)
            if self.cooldown == 0:
                summary = (
                    f"{self.name} enjoys {self.target.name} to restore "
                    f"{', '.join(k for k, v in satisfy.items() if v)}."
                )
                events.append(summary)
                self._record_memory(tick, summary, 0.4, tuple(satisfy.keys()))
                self.cooldown = 4

        speech_gain = core_mod.get("speech_gain", 0.2)
        social_pressure = self.needs.values.get("belonging", 0.0)
        if rng.random() < speech_gain and social_pressure > 0.35:
            fact = knowledge.random_fact()
            dialogue = f"{self.name} shares: {fact}"
            events.append(dialogue)
            self.needs.values["belonging"] = max(0.0, social_pressure - 0.2)
            self.affect.adjust(0.03, 0.02, 0.01)
            self._record_memory(tick, dialogue, 0.3, ("conversation",))

        reflection = self.reflect_if_needed(tick, knowledge, rng)
        if reflection:
            events.append(reflection)

        return events


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------


class Simulation:
    """Core loop orchestrating agents and time progression."""

    def __init__(
        self,
        world: TownMap,
        pois: Sequence[PointOfInterest],
        agents: Sequence[Agent],
        knowledge: KnowledgeBase,
    ) -> None:
        self.world = world
        self.pois = list(pois)
        self.agents = list(agents)
        self.knowledge = knowledge
        self.tick_count = 0
        self._running = False
        self.speed_ms = 400
        self._rng = random.Random(42)
        self.day_count = 0
        self.director = SimulationDirector()

    @property
    def hour(self) -> int:
        return (self.tick_count // 6) % 24

    @property
    def day(self) -> int:
        return self.day_count

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def toggle(self) -> None:
        self._running = not self._running

    def running(self) -> bool:
        return self._running

    def step(self) -> List[str]:
        if not self._running:
            return []
        events: List[str] = []
        previous_hour = self.hour
        self.tick_count += 1
        occupancy = Counter(agent.position for agent in self.agents)
        current_hour = self.hour
        for agent in self.agents:
            events.extend(
                agent.step(
                    self.world,
                    self.pois,
                    current_hour,
                    self.knowledge,
                    self._rng,
                    occupancy,
                    self.tick_count,
                )
            )
        if previous_hour == 23 and current_hour == 0:
            self.day_count += 1
            events.extend(
                self.director.end_of_day(self.day_count, self.agents, self.tick_count, self.knowledge)
            )
        return events


# ---------------------------------------------------------------------------
# Narrative director and targeted evolution helpers
# ---------------------------------------------------------------------------


class SimulationDirector:
    """Curates day summaries and gentle evolutionary nudges."""

    def __init__(self) -> None:
        self.episode = 0

    def end_of_day(
        self,
        day: int,
        agents: Sequence[Agent],
        tick: int,
        knowledge: KnowledgeBase,
    ) -> List[str]:
        highlights: List[str] = [f"Day {day} closes over Smallville+++. "]
        for agent in agents:
            if not agent.memory.events:
                continue
            latest = agent.memory.events[-1].content
            highlights.append(f"{agent.name}'s diary: {latest[:120]}")
        if day % 3 == 0:
            self.episode += 1
            champion = max(
                agents,
                key=lambda a: a.interests.topics.get("civic", 0.0)
                + (1 - a.needs.values.get("civic_impact", 0.0)),
            )
            champion.interests.boost("civic", 0.08)
            champion.needs.values["civic_impact"] = max(
                0.0, champion.needs.values.get("civic_impact", 0.0) - 0.25
            )
            story = (
                f"Evolution sprint {self.episode}: {champion.name} steers a civic jam "
                f"after studying {knowledge.random_fact()}."
            )
            champion.memory.record(
                Memory(timestamp=tick, content=story, importance=0.5, tags=("civic",))
            )
            highlights.append(story)
        return highlights[:6]


# ---------------------------------------------------------------------------
# Tkinter UI
# ---------------------------------------------------------------------------


class AgentPanel(ttk.Frame):
    def __init__(self, master: tk.Widget, *, simulation: Simulation, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.simulation = simulation

        self.tree = ttk.Treeview(self, columns=("need", "value"), show="tree")
        self.tree.heading("#0", text="Agent")
        self.tree.column("#0", width=160, stretch=False)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for agent in self.simulation.agents:
            header = (
                f"{agent.name} — target: {agent.target.name if agent.target else 'exploring'}"
                f" — cash {agent.cash:5.1f}"
            )
            parent = self.tree.insert("", tk.END, text=header)
            mood = (
                f"mood V:{agent.affect.valence:+.2f} A:{agent.affect.arousal:+.2f} D:{agent.affect.dominance:+.2f}"
            )
            self.tree.insert(parent, tk.END, text=mood)
            for need in NEEDS:
                value = agent.needs.values.get(need, 0.0)
                pct = int(value * 100)
                self.tree.insert(parent, tk.END, text=f"{need:>14}: {pct:3d}%")
            if agent.life_plan:
                self.tree.insert(parent, tk.END, text="Plan: " + " | ".join(agent.life_plan[:3]))
            self.tree.item(parent, open=True)


class EventLog(ttk.Frame):
    def __init__(self, master: tk.Widget, *, max_lines: int = 8, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.max_lines = max_lines
        self.text = tk.Text(self, height=max_lines * 2, state=tk.DISABLED, wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True)

    def append(self, lines: Sequence[str]) -> None:
        if not lines:
            return
        self.text.configure(state=tk.NORMAL)
        current = self.text.get("1.0", tk.END).splitlines()
        current = [line for line in current if line.strip()]
        for line in lines:
            current.append(f"• {line}")
        if len(current) > self.max_lines:
            current = current[-self.max_lines :]
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "\n".join(current) + "\n")
        self.text.configure(state=tk.DISABLED)


class ControlPanel(ttk.Frame):
    def __init__(self, master: tk.Widget, simulation: Simulation, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.simulation = simulation
        self.status_var = tk.StringVar(value="Ready")

        self.start_button = ttk.Button(self, text="Start / Pause", command=self.toggle)
        self.start_button.grid(row=0, column=0, padx=4, pady=4)

        ttk.Label(self, text="Tick speed (ms)").grid(row=0, column=1, padx=4)
        self.speed = tk.IntVar(value=self.simulation.speed_ms)
        self.speed_scale = ttk.Scale(
            self,
            from_=120,
            to=800,
            command=self._on_speed_change,
            orient=tk.HORIZONTAL,
        )
        self.speed_scale.set(self.simulation.speed_ms)
        self.speed_scale.grid(row=0, column=2, padx=4, sticky="ew")

        self.status = ttk.Label(self, textvariable=self.status_var)
        self.status.grid(row=0, column=3, padx=6)

        self.clock_var = tk.StringVar(value="Day 0 • Hour 00")
        self.clock = ttk.Label(self, textvariable=self.clock_var)
        self.clock.grid(row=0, column=4, padx=6)

        self.columnconfigure(2, weight=1)

    def toggle(self) -> None:
        self.simulation.toggle()
        self.status_var.set("Running" if self.simulation.running() else "Paused")

    def _on_speed_change(self, _value: str) -> None:
        value = int(float(_value))
        self.simulation.speed_ms = value
        self.status_var.set(f"Tick ≈ {value} ms")


class SmallvilleApp(tk.Tk):
    def __init__(self, *, auto_start: bool = True) -> None:
        super().__init__()
        self.title("Smallville+++ Town Life Prototype")
        self.geometry("1280x840")
        self.resizable(True, True)

        self.cell_size = 32
        self.assets = AssetManager(ASSET_ROOT, self.cell_size)
        self.world = TownMap()

        config = load_optional_config()
        agent_count = int(config.get("init_agents", 8))

        self.pois = create_default_pois()
        knowledge = KnowledgeBase()
        agents = create_agents(
            agent_count,
            self.world,
            self.assets,
        )
        self.simulation = Simulation(self.world, self.pois, agents, knowledge)

        self.canvas = tk.Canvas(
            self,
            width=self.world.width * self.cell_size,
            height=self.world.height * self.cell_size,
            background="#2b2b2b",
        )
        self.canvas.grid(row=0, column=0, rowspan=3, sticky="nsew")

        self.agent_panel = AgentPanel(self, simulation=self.simulation)
        self.agent_panel.grid(row=0, column=1, sticky="nsew")

        self.event_log = EventLog(self)
        self.event_log.grid(row=1, column=1, sticky="nsew")

        self.controls = ControlPanel(self, self.simulation)
        self.controls.grid(row=2, column=1, sticky="ew")

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=4)
        self.rowconfigure(1, weight=2)
        self.rowconfigure(2, weight=0)

        self._tile_refs: List[ImageTk.PhotoImage] = []
        self._agent_items: Dict[str, int] = {}

        self._draw_static_map()

        if auto_start:
            self.after(200, self.start_simulation)

        self.after(200, self._tick)

    def start_simulation(self) -> None:
        self.simulation.start()
        self.controls.status_var.set("Running")

    def _draw_static_map(self) -> None:
        self.canvas.delete("map")
        self._tile_refs.clear()
        for y in range(self.world.height):
            for x in range(self.world.width):
                tile = self.world.tile(x, y)
                img = self.assets.tile(tile)
                image_id = self.canvas.create_image(
                    x * self.cell_size,
                    y * self.cell_size,
                    image=img,
                    anchor=tk.NW,
                    tags=("map", "tile"),
                )
                self._tile_refs.append(img)

        for poi in self.pois:
            px, py = poi.xy
            self.canvas.create_text(
                px * self.cell_size + self.cell_size / 2,
                py * self.cell_size + self.cell_size / 2,
                text=poi.name.split()[0],
                fill="#1b1b1b",
                font=("Segoe UI", 8, "bold"),
                tags=("map", "poi_label"),
            )

    def _tick(self) -> None:
        events = self.simulation.step()
        if events:
            self.event_log.append(events)
        self._render_agents()
        self.agent_panel.refresh()
        self.controls.clock_var.set(
            f"Day {self.simulation.day:02d} • Hour {self.simulation.hour:02d} — Tick {self.simulation.tick_count}"
        )
        self.after(self.simulation.speed_ms, self._tick)

    def _render_agents(self) -> None:
        for agent in self.simulation.agents:
            cx = agent.position[0] * self.cell_size + self.cell_size / 2
            cy = agent.position[1] * self.cell_size + self.cell_size / 2
            frame = agent.frames[agent.frame_index % len(agent.frames)]
            if agent.name not in self._agent_items:
                item = self.canvas.create_image(
                    cx,
                    cy,
                    image=frame,
                    anchor=tk.CENTER,
                    tags=("agent", agent.name),
                )
                self._agent_items[agent.name] = item
            else:
                item = self._agent_items[agent.name]
                self.canvas.coords(item, cx, cy)
                self.canvas.itemconfigure(item, image=frame)


def create_default_pois() -> List[PointOfInterest]:
    """Convenience function describing the important venues in town."""

    return [
        PointOfInterest(
            name="Sunrise Homes",
            kind="residential",
            xy=(6, 2),
            satisfies={
                "energy": 0.4,
                "hygiene": 0.25,
                "belonging": 0.2,
                "safety": 0.15,
            },
            open_hours=(0, 24),
            tags=("home", "rest"),
        ),
        PointOfInterest(
            name="Bluebird Cafe",
            kind="commerce",
            xy=(20, 8),
            satisfies={
                "hunger": 0.45,
                "thirst": 0.35,
                "belonging": 0.2,
                "fun": 0.15,
            },
            open_hours=(6, 22),
            cost=4.0,
            tags=("food", "music"),
        ),
        PointOfInterest(
            name="Civic Hall",
            kind="civic",
            xy=(20, 2),
            satisfies={
                "civic_impact": 0.3,
                "esteem": 0.2,
                "legacy": 0.1,
            },
            open_hours=(8, 20),
            tags=("civic", "leadership"),
        ),
        PointOfInterest(
            name="Wellness Clinic",
            kind="health",
            xy=(25, 14),
            satisfies={
                "safety": 0.4,
                "energy": 0.2,
                "esteem": 0.1,
            },
            open_hours=(7, 21),
            cost=6.0,
            tags=("health",),
        ),
        PointOfInterest(
            name="Innovator Lab",
            kind="ai_lab",
            xy=(14, 17),
            satisfies={
                "competence": 0.3,
                "curiosity": 0.25,
                "creativity": 0.2,
            },
            open_hours=(9, 19),
            cost=3.0,
            tags=("coding", "research"),
        ),
        PointOfInterest(
            name="Riverside Park",
            kind="park",
            xy=(5, 6),
            satisfies={
                "fun": 0.3,
                "belonging": 0.25,
                "creativity": 0.1,
            },
            open_hours=(5, 23),
            tags=("nature", "sports"),
        ),
        PointOfInterest(
            name="Downtown Offices",
            kind="commerce",
            xy=(12, 8),
            satisfies={
                "money_pressure": 0.35,
                "competence": 0.2,
                "esteem": 0.15,
            },
            open_hours=(7, 18),
            cost=-8.0,
            tags=("career", "coding"),
        ),
        PointOfInterest(
            name="Makers Guild",
            kind="leisure",
            xy=(2, 13),
            satisfies={
                "creativity": 0.3,
                "fun": 0.25,
                "autonomy": 0.2,
            },
            open_hours=(10, 22),
            cost=2.0,
            tags=("art", "maker"),
        ),
        PointOfInterest(
            name="Metro Hub",
            kind="transport",
            xy=(2, 0),
            satisfies={
                "belonging": 0.15,
                "autonomy": 0.2,
                "curiosity": 0.1,
            },
            open_hours=(0, 24),
            tags=("travel", "community"),
        ),
        PointOfInterest(
            name="Aurora Studio",
            kind="leisure",
            xy=(9, 15),
            satisfies={
                "creativity": 0.35,
                "legacy": 0.2,
                "esteem": 0.2,
            },
            open_hours=(11, 23),
            cost=5.0,
            tags=("art", "music"),
        ),
        PointOfInterest(
            name="Community Library",
            kind="education",
            xy=(17, 4),
            satisfies={
                "curiosity": 0.35,
                "competence": 0.25,
                "legacy": 0.15,
            },
            open_hours=(8, 21),
            tags=("writing", "research"),
        ),
    ]


def create_agents(
    count: int,
    world: TownMap,
    assets: AssetManager,
) -> List[Agent]:
    rng = random.Random(123)
    agents: List[Agent] = []
    spawn_tiles = [
        (x, y)
        for y in range(world.height)
        for x in range(world.width)
        if world.tile(x, y) in {"road", "transport", "park"}
    ]
    residential_tiles = [
        (x, y)
        for y in range(world.height)
        for x in range(world.width)
        if world.tile(x, y) == "residential"
    ]
    variants = assets.agent_variants or ["citizen", "citizen_f"]
    for idx in range(count):
        position = rng.choice(spawn_tiles)
        personality = Personality.random()
        needs = NeedState.create()
        variant = variants[idx % len(variants)]
        frames = assets.agent_frames(variant)
        affect = AffectState.random()
        interests = InterestProfile.default()
        memory = MemoryStream()
        liquid_core = LiquidCore(seed=rng.randint(0, 1_000_000))
        home = rng.choice(residential_tiles) if residential_tiles else position
        cash = rng.uniform(80.0, 180.0)
        agent = Agent(
            name=f"Agent {idx + 1}",
            position=position,
            personality=personality,
            needs=needs,
            sprite_variant=variant,
            frames=frames,
            affect=affect,
            interests=interests,
            memory=memory,
            liquid_core=liquid_core,
            cash=cash,
            home_xy=home,
        )
        agent.life_plan = ["Morning stroll", "Deep work block", "Community hangout"]
        agents.append(agent)
        agent._record_memory(
            tick=0,
            description=f"{agent.name} settles into town with {agent.cash:.0f} credits.",
            importance=0.2,
            tags=("arrival",),
        )
    return agents


def main() -> None:
    app = SmallvilleApp()
    app.mainloop()


if __name__ == "__main__":
    main()

