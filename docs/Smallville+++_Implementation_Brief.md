# GPT Agent Implementation Brief — “Smallville+++”

> Goal: Build a one-click Tkinter town sim with rich motives, personalities, evolving life paths, local tiny-LLM tutoring, external knowledge ingestion (optional), a minimal liquid-neural core per agent (safe/clamped), and **ready-to-use CC0 placeholders** for interiors, props, roads, and animated characters.

---

## 0) Zero-friction asset sources (CC0) — download these now

These are **public-domain (CC0)** packs you can drop in as placeholders. They include interiors, exteriors, and **4-direction walking animations** so you’ll immediately see activity inside buildings.

### Must-have tiles + characters

* **Kenney — RPG Urban Pack** (16×16 exteriors/roads/buildings + small cast). License: **CC0**. Download page lists tile size and license. ([kenney.nl][1])
* **Kenney — RPG Urban Kit (itch mirror)** (same content, mentions **“6 characters in four directions with walking animations.”**). Great for immediate walk cycles. ([itch.io][2])

### Interiors & furniture

* **Kenney — Roguelike Indoors** (large set of furniture & indoor tiles; CC0). Also mirrored on OpenGameArt. ([kenney.nl][3])

### Industrial/warehouse/concrete floors & walls

* **0x72 — 16×16 Dungeon Tileset** (CC0; explicitly “use for whatever you like”). Great for basements, factory backrooms, storage. ([itch.io][4])
* **0x72 — Dungeon Tileset II** (newer recolor; CC0). ([itch.io][5])

### Clinical props/floors (optional)

* **OpenGameArt — Hospital room** (CC0; top-down bed + tiled floor). Useful for a clinic/hospital cell. ([OpenGameArt.org][6])

### Kenney index (browse if you need complements)

* Kenney asset browser (all **CC0**; 2D/3D/UI/audio). ([kenney.nl][7])

> Why these? They’re CC0 (no attribution needed), stylistically consistent (16×16), and include **characters with 4-dir walk cycles** (from the Urban Kit) so your sim shows movement in/out/inside buildings from day one. ([itch.io][2])

---

## 1) File layout (minimal upload)

```
smallville_pp.py          # single-file Tkinter app (implements everything below)
assets/
  tiles/
    ground.png wall.png road.png
    cafe_floor.png home_floor.png office_floor.png store_floor.png
    school_floor.png clinic_floor.png factory_floor.png park_grass.png
    poi.png
  props/
    tree.png bench.png lamp.png sign.png desk.png bed.png shelf.png table.png stage.png
  agents/
    # From Kenney RPG Urban kit (4-dir walk cycles recommended)
    agent_blue_idle_down.png
    agent_blue_idle_up.png
    agent_blue_idle_left.png
    agent_blue_idle_right.png
    agent_blue_walk_down_0.png ... agent_blue_walk_down_3.png
    agent_blue_walk_up_0.png   ... agent_blue_walk_up_3.png
    agent_blue_walk_left_0.png ... agent_blue_walk_left_3.png
    agent_blue_walk_right_0.png... agent_blue_walk_right_3.png
    # repeat for green/orange/purple
config.json                # optional knobs (see below)
```

**How to fill the folder with placeholders:**

* From **RPG Urban Pack/Kit**, export road/sidewalk/building-edge tiles → `tiles/road.png`, `tiles/ground.png`, `tiles/wall.png`, and cut the **character frames** into separate 16×16 PNGs named as above (4 frames per direction + idle). The itch page confirms **four directions with walking animations** are included. ([itch.io][2])
* From **Roguelike Indoors**, cut tables, chairs, shelves, beds, desks → `props/*` and indoor floors to `tiles/*` (café/home/office/school/store). ([OpenGameArt.org][8])
* From **0x72** tilesets, grab concrete/factory floors/walls → `tiles/factory_floor.png`, extra walls if needed. CC0, easy to slice. ([itch.io][4])
* Clinic bed/floor: crop from **Hospital room** → `props/bed.png`, `tiles/clinic_floor.png`. ([OpenGameArt.org][6])

(If you want me to name exact sheet coordinates from each pack, I can provide a cutting manifest; the Kenney packs also ship **separate sprites** folders which makes this trivial. ([itch.io][2]))

---

## 2) Config (optional)

```json
{
  "grid": [34,22],
  "init_agents": 8,
  "episodes": 10,
  "speed_ms": 55,
  "softmax_temperature": 0.75,
  "web_learning": { "wikipedia": true, "open_library": true, "gutenberg": true, "wikidata": true },
  "reading_budget_per_day_chars": 2500,
  "lnn": { "hidden_neurons": 8, "dt": 0.1, "unroll": 1 },
  "animations": { "enabled": true, "fps": 6 }
}
```

---

## 3) Core architecture

### 3.1 World & zones

* Grid (default **34×22**) with:

  * **Districts**: Residential, Civic (City Hall), Education (school, library, **AI Lab**), Healthcare (clinic), Commerce (café, store), Industry (office, factory), Leisure (park, gym, music venue), Transport (roads/bus stops).
* **POIs** with `open_hours`, `capacity`, `satisfy[need]` vectors, and optional fees.

### 3.2 Agents

* **State**:

  * **Needs** (urgency ∈ [0,1]):

    * *Base*: hunger, thirst, energy (sleep), hygiene, safety, money_pressure
    * *Social*: belonging/relatedness, esteem/status
    * *Growth*: competence, autonomy
    * *Expression*: fun/play, creativity, curiosity
    * *Civic/legacy*: civic_impact, legacy
  * **Personality** (Big Five O,C,E,A,N ∈ [0,1])
  * **Affect** (PAD ∈ [-1,1]³)
  * **Interests** (topic → skill ∈ [0,1]) e.g., music, art, writing, sports, coding, cooking, civic
  * **Assets**: home_xy (rent/own), cash, job(role, seniority), tools (guitar, sketchbook, laptop)
  * **Relations**: likes_people[name] (float), reputation (0–100)
  * **Dialogue**: template+bigram fallback + safe sanitizer
  * **LiquidCore**: 8-neuron liquid cell (modulates exploration/speech/tiny decay tweaks; *cannot bypass safety*)

### 3.3 Planners

* **Reactive** (mask invalid moves; reroute if queues full).
* **Utility** (per tick):
  `U(target) = Σ need_i * satisfy[target,i] * TraitBias(personality) * AffectMod(PAD) * SatiationMask(lower-layers)`
  Softmax over candidate places, ε-greedy exploration.
* **Liquid modulation** (heads): adjustments to exploration gain, action logits, speech rate (always clamped).
* **Episodic** (daily 2–4 step fragments; γ≈0.95).
* **Life planner** (weekly: move home, switch job, run for mayor) with cooldowns & budget checks.

### 3.4 Safety layer (fixed, non-learnable)

* **Action masks**: impassables, money can’t go negative, venue capacity/open hours.
* **Cooldowns**: high-risk actions (job switch, home purchase, campaign start).
* **Language guard**: sanitizer + token limit + topic whitelist; never long quotes.
* **I/O limits**: domain allowlist & per-day budgets (Wikipedia, Open Library, Gutenberg, Wikidata).
* **Numeric clamps**: LiquidCore h/tau bounds; NaN/Inf detection with rollback.

---

## 4) Animation system (using online placeholders)

**Goal:** Render 4-direction movement with 4-frame walking loops + idle facing up/down/left/right.

**Expected filenames (per color):**

```
agents/agent_blue_idle_down.png
agents/agent_blue_idle_up.png
agents/agent_blue_idle_left.png
agents/agent_blue_idle_right.png
agents/agent_blue_walk_down_0.png ... _3.png
agents/agent_blue_walk_up_0.png   ... _3.png
agents/agent_blue_walk_left_0.png ... _3.png
agents/agent_blue_walk_right_0.png... _3.png
```

> From **Kenney RPG Urban Kit**, cut the 4-dir walking animations into the above files (16×16 each). The kit explicitly lists “characters in four directions with walking animations.” ([itch.io][2])

**Animation logic (Tkinter):**

* Keep a per-agent `facing ∈ {up,down,left,right}`;
* If moving, advance frame index every `tick % (T/frames)`; else use the `idle_facing` frame;
* Draw with `create_image` at cell center (Tk PhotoImage); scale is handled by cell size.

---

## 5) Dialogue & learning

### 5.1 Dialogue (safe, light)

* Sanitized templates + a small **bigram** learner per agent (already implemented in earlier code), intent weights: greet/share/ask/report/plan/chitchat.
* PAD and Extraversion bias **speech probability**; Relatedness gains on positive exchanges.

### 5.2 External knowledge adapters (optional, safe)

* **Wikipedia** (TextExtracts `exintro`/`explaintext`) → brief notes that agents can reference. ([kenney.nl][9])
* **Open Library** (subjects/search JSON) → pick study books/projects (metadata only). ([Sprite Fusion, Free 2D tilemap editor][10])
* **Project Gutenberg** → public-domain excerpts for “deep reading” periods. ([OpenGameArt.org][6])
* **Wikidata** (SPARQL) → civic facts (mayor role, term), cached per episode. ([OpenGameArt.org][8])

**Guards:** character/episode budgets, domain allowlist, exponential backoff, strip HTML/markup, never store long quotations.

---

## 6) Liquid Neural Core (per agent)

A tiny Liquid Time-constant cell (N=8) with discrete fused-Euler update and clamped parameters.

**Inputs (scaled to [-1,1]):** local perception (poi proximities, people density), time of day (sin/cos), selected needs (e.g., hunger/energy/autonomy/competence/relatedness), PAD, user_controlled flag.

**Outputs (heads):** `logits_action[K]`, `speech_rate`, `explore_gain`, `satisfy_gain[needs_subset]`.

**Update (one unroll per tick):**

```
z      = tanh(Wxh h + Wxi x + b)             # N
alpha  = softplus(Wah h + Wai x + a0)        # N
Ahat   = A_scale * tanh(WhA h + WiA x + A0)  # N
h'     = (h + dt * z * Ahat) / (1 + dt*(tau_inv + z))
h      = clamp(h', -h_max, h_max)
heads  = linear(h) → logits_action, speech_rate, explore_gain, satisfy_gain
```

**Safety:** clamp `tau_inv ∈ [0, τ_max]`, `|W| ≤ w_max`, rollback on NaN/Inf. Output heads pass through **action masks** and rate limiters.

**Training:**

* **Within episode**: tiny RMSProp/Adam on heads only using reward = Δ(−composite need) + small social bonus − penalties.
* **Across episodes**: elite selection copies **traits/persistence/heads** with small noise.

---

## 7) Venue effects (quantified)

Per tick contributions (tune in `config.json`):

| Venue/Action        | +Needs per tick                                     | Costs/Side effects  |
| ------------------- | --------------------------------------------------- | ------------------- |
| Home (sleep/shower) | −energy 0.10, −hygiene 0.04                         | time                |
| Café/Restaurant     | −hunger 0.08, +belonging 0.02, −thirst 0.06         | −cash 1             |
| Library             | +competence 0.05, +curiosity 0.04                   | −energy 0.02        |
| Music Venue         | +fun 0.05, +creativity 0.06 (perform: +esteem 0.05) | −energy 0.03        |
| Maker Space         | +creativity 0.06, +competence 0.04                  | −cash 1 (materials) |
| Gym/Stadium         | +fun 0.04, +esteem 0.03                             | −energy 0.05        |
| School (AI Lab)     | +competence 0.06, +autonomy 0.02                    | −energy 0.03        |
| City Hall           | +esteem 0.03, +civic_impact 0.05                    | time                |
| Work (on shift)     | −money_pressure 0.03, +competence 0.03              | −energy 0.03        |

---

## 8) Tiny local LLM @ School (optional)

If `local_tiny_llm.py` + `weights.npz` exist, use a 1–2 layer tiny transformer (128–256 dim, ctx 256) for short lessons & quizzes; else fallback to n-gram + retrieval.

**Tutor API:**

```python
class Tutor:
    def __init__(self, model=None): ...
    def lesson(self, topic:str)->str: ...      # ≤ 200 tokens
    def quiz(self, topic:str)->list[str]: ...
    def check(self, topic:str, answer:str)->dict:  # {"correct": bool, "explanation": str}
```

**Guards:** daily token budget, topic whitelist, sanitized outputs.

---

## 9) JSON schemas (snapshots & map)

**Agent snapshot:**

```json
{
  "id":"A3",
  "home":[6,5],
  "job":{"role":"office","seniority":0.3},
  "cash":84,
  "needs":{"hunger":0.22,"energy":0.41,"hygiene":0.18,"safety":0.12,
           "money_pressure":0.29,"belonging":0.35,"esteem":0.20,
           "competence":0.28,"autonomy":0.17,"fun":0.26,"creativity":0.33,"curiosity":0.37},
  "traits":{"O":0.7,"C":0.5,"E":0.6,"A":0.55,"N":0.3},
  "affect":{"P":0.2,"A":0.1,"D":0.0},
  "interests":{"music":0.4,"art":0.2,"coding":0.3},
  "likes_people":{"A1":0.42,"A4":0.18},
  "reputation":58,
  "projects":[{"type":"song","progress":0.35}],
  "notes":["Mayor term is 4 years…"]
}
```

**Venue spec (can live in `map.json`):**

```json
{
  "cafe": {
    "xy":[18,6],
    "open":[[200,500]],
    "cap":8,
    "satisfy":{"hunger":0.08,"belonging":0.02,"thirst":0.06},
    "cost":{"cash":1}
  }
}
```

---

## 10) Controls & UI (Tkinter)

* Start/Pause/Reset; speed slider; episodes spinbox.
* **C** to claim/release control of focused agent; arrows/WASD to move; **E** to interact; **T** to talk; **Tab** to cycle focus.
* Status bar shows episode/step/persistence; dialogue log shows recent utterances; speech bubbles render over agents.

---

## 11) Milestones (for the GPT agent to implement)

**M1 — Baseline run**

* Load tiles/props/animated agents from the CC0 packs above; draw grid; move agents with 4-dir animations; ensure interiors render.

**M2 — Psychology & utility planner**

* Implement needs hierarchy (with satiation mask), Big Five traits, PAD affect; build utility function and softmax picker.

**M3 — LiquidCore + Safety**

* Add 8-neuron liquid cell + clamps; wire to exploration/speech/tiny decay; enforce safety masks, cooldowns, numeric guards.

**M4 — Social/culture & projects**

* Relationships graph; bands/study circles; maker/music projects produce artifacts and esteem.

**M5 — Life paths**

* Home purchase/rent; job switch/retraining; run for mayor; events (market day, open mic, council); simple election.

**M6 — External reading (optional)**

* Wikipedia/Open Library/Gutenberg/Wikidata adapters with budgets and cache; notes inform dialogue and planners.

**M7 — Polish**

* A* pathfinding; save/load; animated walk cycles at 6 FPS; HUD tooltips for needs.

---

## 12) Test plan

* **Smoke**: no NaNs; safety masks block illegal moves; animation cycles advance correctly.
* **Metrics**: moving average of composite need ↓ over time; competence increases after study; esteem bumps after performances/promotions.
* **A/B**: Liquid heads on/off; external learning on/off.
* **Long run (10 episodes)**: at least one home purchase, one job switch, one election.

---

## 13) Prompts for the GPT coding agent (you can paste as tasks)

**Task A — Asset intake + animation**

> In `smallville_pp.py`, load PNGs from `assets/tiles|props|agents`. Implement an `Animation` helper that selects idle or walk frame based on velocity and facing (up/down/left/right) at `animations.fps` frames per second. Use Tk `PhotoImage` cache to avoid reloading. If sprites are missing, draw colored ovals as fallback.

**Task B — Planner & psychology**

> Add Need/Traits/PAD to `Agent`; implement `utility(target)` and softmax choice with temperature from config. Add a `satiation_mask = 1/(1+10*base_need_sum)` for higher layers when base needs are starving.

**Task C — LiquidCore + Safety**

> Implement `LiquidCore.step(x)` as specified; clamp parameters; wrap all action outputs with masks/cooldowns. If numeric anomalies occur, revert `h` and zero heads for one tick.

**Task D — Life paths & events**

> Implement home purchase, job switch, campaign flows with cooldowns and budget checks; add event scheduler (daily/weekly/seasonal) that spawns temporary POIs.

**Task E — External knowledge (optional)**

> Add adapters for Wikipedia/Open Library/Gutenberg/Wikidata with domain allowlist and per-agent/day budgets; parse into short notes; reference in dialogue.

---

## 14) Exact URLs to download placeholders now

* Kenney **RPG Urban Pack** (16×16, CC0). Use for roads, exteriors, and characters: Download page lists tile size & CC0 license. ([kenney.nl][1])
* Kenney **RPG Urban Kit** (itch mirror). **Includes 6 characters in four directions with walking animations** (use these for `agents/*_walk_*` frames). ([itch.io][2])
* Kenney **Roguelike Indoors** (CC0 furniture/interior tiles): tables, chairs, shelves, sofas, kitchen. ([kenney.nl][3])
* **0x72** 16×16 Dungeon Tileset (CC0) and **DungeonTileset II** (CC0) — concrete/industrial floors, walls, crates: great for factory/warehouse. ([itch.io][4])
* **OpenGameArt** Hospital room (CC0) — clinic bed/floor. ([OpenGameArt.org][6])
* Kenney all assets index (browse CC0 if you need more props or variants). ([kenney.nl][7])

---

## 15) Packaging

* Put the downloaded PNGs in the `assets/` tree with the exact filenames above.
* Zip the whole project as `smallville_plusplus.zip`:

  ```
  smallville_pp.py
  assets/tiles/*.png
  assets/props/*.png
  assets/agents/*.png
  (optional) config.json
  ```
* The app will run *even if* some assets are missing—fallbacks render—but you’ll get full animated life if the **RPG Urban Kit** character frames are present. ([itch.io][2])

---

## 16) Notes on licensing & safety

* Kenney and 0x72 packs are **CC0**; attribution not required (but appreciated). Pages explicitly state CC0 / “use for whatever you like.” ([kenney.nl][1])
* OpenGameArt listings above are CC0; always double-check the page license badge before downloading alternates. ([OpenGameArt.org][6])
* External text adapters read short excerpts; store *summaries/facts*, not long quotations; respect rate limits and keep a strict domain allowlist.

---

If you want, I can also provide a **cutting manifest** for the Urban Kit and Roguelike Indoors (tile sheet coordinates → destination filename) so a script can slice all the frames into the exact `agents/*` names automatically.

[1]: https://kenney.nl/assets/rpg-urban-pack?utm_source=chatgpt.com "RPG Urban Pack"
[2]: https://kenney-assets.itch.io/rpg-urban-kit?utm_source=chatgpt.com "RPG Urban Kit by Kenney (Assets) - itch.io"
[3]: https://kenney.nl/assets/roguelike-indoors?utm_source=chatgpt.com "Roguelike Indoors"
[4]: https://0x72.itch.io/16x16-dungeon-tileset?utm_source=chatgpt.com "16x16 Dungeon Tileset by 0x72 - Itch.io"
[5]: https://0x72.itch.io/dungeontileset-ii?utm_source=chatgpt.com "16x16 DungeonTileset II by 0x72 - itch.io"
[6]: https://opengameart.org/content/hospital-room?utm_source=chatgpt.com "Hospital room"
[7]: https://kenney.nl/assets?utm_source=chatgpt.com "Assets"
[8]: https://opengameart.org/content/roguelike-indoor-pack?utm_source=chatgpt.com "Roguelike Indoor pack"
[9]: https://kenney.nl/assets/tag%3Aroguelike?utm_source=chatgpt.com "Roguelike"
[10]: https://www.spritefusion.com/tilesets/dungeon?utm_source=chatgpt.com "Dungeon Tileset"

