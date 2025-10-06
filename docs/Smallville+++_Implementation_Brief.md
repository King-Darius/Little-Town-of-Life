# GPT Agent Implementation Brief — “Smallville+++”

> Goal: Build a one-click Tkinter town sim with rich motives, personalities, evolving life paths, local tiny-LLM tutoring, external knowledge ingestion (optional), a minimal liquid-neural core per agent (safe/clamped), and **ready-to-use CC0 placeholders** for interiors, props, roads, and animated characters.

---

## 1) File layout (minimal upload)

```
smallville_gui.py          # Tkinter app (implements everything below)
run_smallville.py          # one-click launcher that installs requirements and starts the GUI
requirements.txt           # Python dependencies (currently Pillow)
Assets/
  assets/
    catalog/               # Kenney roguelike city & indoor tile atlases
    tiles/                 # convenience tiles extracted from the packs
    props/                 # props & vehicles (used for fallback sprites)
    agents/                # Kenney Toon Characters (idle/run/walk poses)
config.json                # optional knobs (see below)
```


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

> **Prototype status:** the bundled Tkinter implementation realises this stack with a hybrid approach — a handcrafted utility scorer blended with an 8-neuron liquid micro-core (drawing on Liquid AI's grafting experiments) that modulates exploration/speech/satisfaction, plus a nightly reflective planner that refreshes each agent's life-plan snippets.

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

> Slice any compatible 4-direction sprite sheet into the filenames above (16×16 each) so the renderer can alternate idle and walk frames without additional metadata.

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

* **Wikipedia** (TextExtracts `exintro`/`explaintext`) → brief notes that agents can reference.
* **Open Library** (subjects/search JSON) → pick study books/projects (metadata only).
* **Project Gutenberg** → public-domain excerpts for “deep reading” periods.
* **Wikidata** (SPARQL) → civic facts (mayor role, term), cached per episode.

**Guards:** character/episode budgets, domain allowlist, exponential backoff, strip HTML/markup, never store long quotations.

> **Prototype status:** current build anchors reflections to curated facts from this brief, keeping offline play deterministic while leaving HTTP connectors as future work hooks.

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

> **Prototype status:** each agent carries a seeded, non-learning liquid core today; the `SimulationDirector` triggers civic "targeted evolution" sprints every 3 in-game days, nudging interests/civic needs to echo OpenEvolve/Liquid AI style search without runtime training.

---

## 7) Research threads reflected in this repo

* [Generative Agents](https://github.com/joonspk-research/generative_agents) — memory stream, reflection thresholding, and daily journaling.
* [OpenEvolve](https://github.com/codelion/openevolve) — episodic civic sprints for directed evolution of community roles.
* Liquid AI research logs — Liquid core modulation ([LFM-1B math reasoning](https://www.liquid.ai/research/lfm-1b-math-can-small-models-be-concise-reasoners)), diffusion grafting, targeted evolution, and liquid/convolutional hybrids for edge deployment.

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

* Load tiles/props/animated agents from the provided `Assets/assets/` placeholders; draw the grid; move agents with 4-dir animations; ensure interiors render.

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

> In `smallville_gui.py`, load PNGs from `Assets/assets/tiles|props|agents`. Implement an `Animation` helper that selects idle or walk frame based on velocity and facing (up/down/left/right) at `animations.fps` frames per second. Use Tk `PhotoImage` cache to avoid reloading. If sprites are missing, draw colored ovals as fallback.

**Task B — Planner & psychology**

> Add Need/Traits/PAD to `Agent`; implement `utility(target)` and softmax choice with temperature from config. Add a `satiation_mask = 1/(1+10*base_need_sum)` for higher layers when base needs are starving.

**Task C — LiquidCore + Safety**

> Implement `LiquidCore.step(x)` as specified; clamp parameters; wrap all action outputs with masks/cooldowns. If numeric anomalies occur, revert `h` and zero heads for one tick.

**Task D — Life paths & events**

> Implement home purchase, job switch, campaign flows with cooldowns and budget checks; add event scheduler (daily/weekly/seasonal) that spawns temporary POIs.

**Task E — External knowledge (optional)**

> Add adapters for Wikipedia/Open Library/Gutenberg/Wikidata with domain allowlist and per-agent/day budgets; parse into short notes; reference in dialogue.

---

## 14) Notes on safety and sourcing

* Maintain a curated `Assets/assets/` directory with consistent licensing (public-domain or original work) so the simulator can be shared without extra clearance.
* When integrating optional external adapters, ensure HTTP requests target allowlisted domains and respect per-day budgets to avoid rate-limit violations.
* External text adapters should store *summaries/facts*, not long quotations; respect rate limits and keep a strict domain allowlist.

---

Maintain a living changelog for downstream coders so architectural updates and tuning decisions stay transparent and easy to diff.

