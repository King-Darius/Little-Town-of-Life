# Little Town of Life

As you can see, this is still an early work in progress.  
<img width="1246" height="840" alt="image" src="https://github.com/user-attachments/assets/a2e20f49-7604-44a0-9f75-398097f627f3" />


This repository hosts design collateral for the "Smallville+++" Tkinter life simulation concept. The core deliverables are a comprehensive implementation brief compiled for GPT-based coding agents and a runnable Tkinter prototype that streams in the bundled Kenney assets.

## Contents

- `docs/Smallville+++_Implementation_Brief.md` — step-by-step plan covering assets, architecture, simulation psychology, safety, testing, and prompts for autonomous coding agents.
- `Assets/` — CC0 Kenney packs already extracted into folders (`assets/catalog`, `assets/tiles`, `assets/props`, `assets/agents`). The GUI automatically discovers tiles and agents from here.
- `smallville_gui.py` — Tkinter-based prototype that renders the town map, loads sprites from `Assets/assets`, and animates autonomous residents with liquid neural micro-cores, reflective memory streams, and day-by-day civic evolution beats inspired by recent generative-agent research.
- `run_smallville.py` — one-click launcher that installs any missing Python dependencies (using `requirements.txt`) before starting the GUI.
- `requirements.txt` — pinned third-party dependencies for reproducible installation.

## Getting Started

1. Install Python 3.9+.
2. (Optional) Review the implementation brief for the proposed project layout and simulation roadmap.
3. Launch the playable prototype by running `python run_smallville.py`. The script installs the dependencies listed in `requirements.txt` (currently Pillow) automatically if they are missing and then opens the Tkinter application.
4. Use the start/pause button and tick speed slider in the UI to explore the simulation. Character sprites are picked automatically from the Kenney "Toon Characters" pack.
5. Watch the agent panel for evolving needs (Maslow + self-actualisation blend), affective state (PAD), life plans, and cash flow. The event log surfaces conversational snippets, venue visits, nightly reflections, and targeted-evolution beats directed by the civic sprint scheduler.

## Feature highlights

- **Generative memory stream** — every citizen journals impactful events and periodically reflects, summarising the day with insights drawn from the implementation brief (inspired by [Generative Agents](https://github.com/joonspk-research/generative_agents)).
- **Liquid neural modulation** — an eight-neuron liquid time-constant cell per agent gently modulates exploration, speech cadence, and satisfaction gains, borrowing ideas from Liquid AI's [Diffusion Transformer grafting](https://www.liquid.ai/research/exploring-diffusion-transformer-designs-via-grafting) and [liquid neural networks](https://www.liquid.ai/research/convolutional-multi-hybrids-for-edge-devices).
- **Targeted evolution loop** — every third day the simulation director curates a civic sprint, boosting the leading organiser and nudging the town towards long-term civic impact (in the spirit of Liquid AI's [targeted evolution](https://www.liquid.ai/research/automated-architecture-synthesis-via-targeted-evolution) and OpenEvolve).
- **Expanded needs model** — residents balance physiological, social, growth, expression, and legacy needs with per-need trait biases and affect-driven planners aligned with the roadmap in `docs/Smallville+++_Implementation_Brief.md`.

## Licensing Notes

All referenced third-party art assets in the brief are CC0/public-domain. Always confirm licenses when downloading updates or alternatives.

