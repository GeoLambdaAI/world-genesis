<p align="right">
  <em>A research project by <a href="https://www.geolambda.ai"><strong>GeoLambda GmbH</strong></a></em>
</p>

<p align="center">
  <sub><em>This simulation was developed primarily with Claude Code, Anthropic's agentic CLI powered by the Claude Opus 4.7 model. The collaboration served as a real-world stress test of the latest coding LLM through extensive prompt engineering.</em></sub>
</p>

# World Genesis: Autonomous Agent Civilization Simulator

**A physics-based, AI-driven simulation of human civilization on Planet Earth —
from the Out-of-Africa migration 70,000 years ago to climate futures beyond 2100.**

Each autonomous agent uses a JEPA world model (LeCun 2022; Maes et al. 2026)
to perceive, plan, and act. Agents eat, trade, build, reproduce, form societies,
wage wars, and evolve — all emergently, on real Earth geography with real climate data.

[![License: AGPL v3+](https://img.shields.io/badge/License-AGPL%20v3+-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/GeoLambdaAI/world-genesis/actions/workflows/test.yml/badge.svg)](https://github.com/GeoLambdaAI/world-genesis/actions/workflows/test.yml)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](https://www.geolambda.ai)

> **Status:** v0.1.0 — initial public release.

<p align="center">
  <a href="static/world-genesis.jpg">
    <img src="static/world-genesis.jpg" alt="World Genesis — global view showing autonomous agents distributed across all continents, with live macro state, JEPA agent cognition, and emergent geopolitics side panels" width="820">
  </a>
  <br>
  <sub><em>World Genesis running the Present-Day scenario — autonomous agents on a real population-density-weighted distribution, with live macro state (CO₂, temperature, population), JEPA agent cognition, and emergent nations.</em></sub>
</p>

---

## Architecture Overview

```
                          +-----------------------+
                          |   Leaflet.js Frontend  |
                          |  Satellite / OSM tiles |
                          +-----------+-----------+
                                      |  WebSocket (SocketIO)
                          +-----------v-----------+
                          |   Flask Server (app.py) |
                          +-----------+-----------+
                                      |
          +---------------------------v---------------------------+
          |                    World Engine (world.py)             |
          |  Tick loop: agents -> businesses -> settlements ->    |
          |  macro ODE -> geopolitics -> resources -> UI emit     |
          +---+----------+----------+----------+----------+------+
              |          |          |          |          |
    +---------v--+  +----v----+  +-v--------+ +v-------+ +v-----------+
    | Agents     |  | Macro   |  | Geo-     | | Bridge | | History    |
    | (agents.py)|  | (macro  |  | politics | | (bridge| | (history   |
    | JEPA world |  |  .py)   |  | (.py)    | |  .py)  | |  .py)      |
    | model,     |  | 14-state|  | Nations, | | Macro  | | Paleo-     |
    | traits,    |  | ODE:    |  | alliances| | <-> Agnt| | climate,   |
    | skills,    |  | CO2,    |  | trade,   | | <-> Geo | | migration, |
    | memory     |  | temp,   |  | conflict | |        | | Diamond,   |
    +---------+--+  | SLR,    |  | (IFs)    | +--------+ | Dawkins    |
              |     | tension |  +----------+             +------------+
    +---------v--+  +---------+
    | Shared     |
    | World Model|
    | (shared_   |
    |  world_    |
    |  model.py) |
    | Batch JEPA |
    +------------+
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Scenarios](#scenarios)
3. [Scientific Foundations](#scientific-foundations)
4. [Module Reference](#module-reference)
5. [Data Pipeline](#data-pipeline)
6. [Performance](#performance)
7. [References](#references)

---

## Quick Start

> **For step-by-step operational guidance** — env vars, LLM setup, log analysis, troubleshooting — see [`HowTo.md`](HowTo.md).

### Requirements

- Python 3.11+
- ~17 MB disk for pre-computed Earth data

### Installation

```bash
cd interactive_simulation
pip install numpy flask flask-socketio eventlet scipy networkx shapely requests
```

### Generate Earth Data (one-time)

```bash
python generate_landmask.py        # ~10s — rasterizes Natural Earth coastlines
python generate_earth_data.py      # ~1s  — climate zones, resources, biomes
python generate_present_day_data.py # ~15s — World Bank API, NOAA, NASA (needs internet)
```

### Run

```bash
python app.py
# Open http://localhost:5000
# Select scenario → Click Start
```

---

## Scenarios

### Scenario A: 70,000 Years of Human History

Agents begin as small bands in **East Africa** (~68,000 BCE). Over thousands of
ticks they migrate through Arabia to Asia, Europe, Australia, and eventually
the Americas via the Beringia land bridge. Agriculture emerges in the Fertile
Crescent. Civilizations rise and fall. The Industrial Revolution triggers the
macro ODE system (CO2, warming, resource depletion). The simulation continues
into the future.

**Time scale**: 200 years/tick (Paleolithic) → 1 month/tick (Modern)

### Scenario B: Present Day → Future

Initializes from **real-world data** (World Bank API, NOAA, NASA GISS):
- 300 agents distributed proportional to real population density
- 140 nations from World Bank economic indicators
- 10 active conflicts with geolocation (Ukraine, Gaza, Sudan, Myanmar...)
- CO2 = 427 ppm, temperature = +1.19°C (actual 2025 values)
- Macro ODE active from tick 0

**Time scale**: 1 month/tick

---

## Scientific Foundations

The simulation integrates research from seven distinct scientific domains.
Every equation in the codebase cites its source.

### 1. JEPA World Model — Agent Cognition

Each agent perceives the world through a **Joint Embedding Predictive Architecture**
(JEPA), as proposed by Yann LeCun.

**Core papers:**
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*.
  Position paper, Meta AI. — Sections 3.1-3.3: cognitive architecture with
  world model, cost module, actor, and configurator.
- Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026).
  *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture
  from Pixels*. arXiv:2603.19312. — Directional finite-difference gradient
  estimation (replaces random perturbation), temporal path straightness metric.
- Qu, H., Morel, M., McCabe, M., Bietti, A., Lanusse, F., Ho, S., & LeCun, Y. (2026).
  *Representation Learning for Spatiotemporal Physical Systems*. arXiv:2603.13227.
  — Linear probing of latent embeddings to test if physical parameters are captured.

**Implementation** (`world_model.py`, `shared_world_model.py`):

| Component | Architecture | Reference |
|-----------|-------------|-----------|
| Encoder | 3-layer MLP (obs → hidden → hidden → latent), RMSNorm + GELU | LeCun 2022, Section 3.1 |
| Predictor | MLP with **Adaptive Layer Normalization** (AdaLN) — action conditions each layer's scale and shift | Maes et al. 2026, Section 3.2 |
| SIGReg | Stochastic Isotropic Gaussian Regularization via Cramer-Wold theorem — random projections + Epps-Pulley normality test | Maes et al. 2026, Section 4 |
| CEM Planner | Cross-Entropy Method: sample action sequences, rollout in latent space, select elites, refine | LeCun 2022, Section 3.4 |
| Training | L = L\_pred + lambda * SIGReg(Z), directional finite-difference gradients (3 directions per weight, central difference) | Maes et al. 2026, Algorithm 1 |

**Loss function:**

```
L = ||z_hat_{t+1} - z_{t+1}||^2 + lambda * SIGReg(Z)
```

where `z_hat_{t+1} = Predictor(Encoder(x_t), a_t)` and `z_{t+1} = Encoder(x_{t+1})`.

**Agent decision loop** (Kahneman's Dual Process Theory):
- **System 1** (JEPA, every tick): observe → encode → needs evaluation → CEM plan → execute
- **System 2** (LLM, social actions only): trade negotiation, governance speech, social dialogue

### 2. System Dynamics — Macro Model

The global state evolves via a **14-variable ODE system** inspired by the Club of Rome.

**Core references:**
- Meadows, D. H., Meadows, D. L., Randers, J., & Behrens, W. W. (1972).
  *The Limits to Growth*. Universe Books. — World3 model structure:
  population-resource-pollution feedback loops.
- Meadows, D. H., Randers, J., & Meadows, D. L. (2004).
  *Limits to Growth: The 30-Year Update*. Chelsea Green. — Calibrated depletion rates.
- Dixson-Decleve, S., Gaffney, O., Ghosh, J., Randers, J., Rockstrom, J.,
  & Stoknes, P. E. (2022). *Earth for All: A Survival Guide for Humanity*.
  New Society Publishers. — Social tension model: f(inequality, food insecurity,
  environmental degradation).
- Nordhaus, W. D. (2017). *Revisiting the social cost of carbon*.
  PNAS 114(7). — DICE model: GDP growth sector, climate damage function D = a*T^2.

**Climate sub-model** (two-layer energy balance):

```
dT/dt = (1/C) * [F(CO2) - lambda*T - gamma*(T - T_deep)]
F = 5.35 * ln(CO2/280)   [Myhre et al. 1998]
```

| Parameter | Value | Source |
|-----------|-------|--------|
| Climate sensitivity (ECS) | 3.0°C / 2xCO2 | IPCC AR6 WG1, Table 7.SM.1 |
| Ocean heat capacity C | 7.0 W*yr/m^2/°C | Held et al. (2010), lower end |
| Climate feedback lambda | 1.1 W/m^2/°C | IPCC AR6 |
| Deep ocean coupling gamma | 0.7 W/m^2/°C | Gregory (2000) |
| CO2 forcing coefficient | 5.35 W/m^2 | Myhre et al. (1998) |
| Natural CO2 absorption | 44% of emissions | Friedlingstein et al. (2024) |
| Base emission rate | 42 GtCO2/yr | Friedlingstein et al. (2024) |

**Resource depletion** follows Hubbert-style curves (Hubbert, 1956),
not linear depletion. Technology provides a balancing loop
(S-curve growth per Romer 1990).

**Validated**: BAU scenario 2025→2100 passes 8/8 checks against IPCC AR6 projections.

### 3. Geopolitics — Emergent Nation-States

Nations are **never pre-defined**. They emerge organically when agent settlements
grow large enough and merge. Interstate dynamics follow established models.

**Core references:**
- Hughes, B. B. (2019). *International Futures (IFs): Building and Using
  Global Models*. Pardee Center, University of Denver. — Conflict probability
  model with calibrated logistic regression coefficients.
- Liberal peace theory (Russett, 1993; Oneal & Russett, 1999):
  trade interdependence reduces interstate conflict probability.
- Tinbergen, J. (1962). *Shaping the World Economy*. — Gravity model of trade.

**Conflict probability** (per nation-dyad per tick):

```
P(conflict) = sigmoid(
    beta_0                              # base rate (~-4.5)
    + beta_1 * resource_competition     # scarce resources → conflict
    + beta_2 * power_parity             # near-peer → more likely
    + beta_3 * (1 - trade_interdep)     # liberal peace theory
    + beta_4 * social_tension           # Earth4All link
    + beta_5 * territorial_overlap      # border proximity
    - beta_6 * shared_alliances         # mutual allies → peace
    - beta_7 * diplomatic_history       # positive history
)
```

Calibrated for ~0.1-0.5% interstate conflict probability per dyad-year.

### 4. Earth System — Real Geography

The simulation runs on real Earth geography derived from multiple datasets.

**Data sources:**
- Natural Earth (naturalearthdata.com) — 110m land polygons, rivers, lakes.
  Rasterized to 0.25° land mask (720x1440) via Shapely point-in-polygon.
- Climate zones classified via **Whittaker biome diagram** (Whittaker 1975):
  temperature x precipitation → 12 biome types.

**Climate model components** (in `generate_earth_data.py`):

| Layer | Method | References |
|-------|--------|-----------|
| Temperature | Latitude + elevation lapse rate (-6.5°C/km) + continentality + ocean currents (Gulf Stream, Kuroshio, Humboldt, Benguela) | Hartmann (2016) *Global Physical Climatology*; Peixoto & Oort (1992) |
| Precipitation | ITCZ + Hadley cell subsidence + mid-latitude storm tracks + monsoon regions + orographic effects | Schneider et al. (2014); Hoskins & Valdes (1990); Roe (2005) |
| Soil fertility | FAO GAEZ methodology: biome + precipitation + temperature + known breadbaskets | Licker et al. (2010); Mueller et al. (2012); Schlenker & Roberts (2009) |
| Mineral deposits | Tectonic/orogenic belts + known provinces | USGS; Marshak (2019); Arndt et al. (2017); Sillitoe (2010) |
| Freshwater | Precipitation + Natural Earth rivers/lakes + known aquifer regions | Doll et al. (2003); Vorosmarty et al. (2010); Schewe et al. (2014) |
| Fossil fuels | Known sedimentary basin locations | USGS World Petroleum Assessment; BGR (2019) |

### 5. Paleoclimate — Ice Ages

The historical scenario models 70,000 years of climate oscillation.

**Data sources:**
- EPICA Community Members (2004). Eight glacial cycles from an Antarctic ice core.
  *Nature* 429, 623-628. — CO2 record for 800 kyr.
- Petit, J. R., et al. (1999). Climate and atmospheric history from the Vostok
  ice core. *Nature* 399, 429-436. — Temperature record for 420 kyr.
- Jouzel, J., et al. (2007). Orbital and millennial Antarctic climate variability
  over the past 800,000 years. *Science* 317, 793-796.
- Marcott, S. A., et al. (2013). A reconstruction of regional and global temperature
  for the past 11,300 years. *Science* 339, 1198-1201.
- Spratt, R. M., & Lisiecki, L. E. (2016). A Late Pleistocene sea level stack.
  *Climate of the Past* 12, 1079-1092.
- Clark, P. U., et al. (2009). The Last Glacial Maximum. *Science* 325, 710-714.
  — LGM ice sheet reconstructions.
- Stringer, C. (2012). *The Origin of Our Species*. Penguin. — Anatomically
  modern human dispersal timeline (Out of Africa, ~70 kya); informs Scenario A
  initial conditions.

**Key events modeled:**

| Event | Year (BP) | CO2 (ppm) | Temp (°C) | Sea Level (m) |
|-------|----------|-----------|-----------|---------------|
| Out of Africa | 70,000 | 200 | -6.0 | -80 |
| Last Glacial Maximum | 21,000 | 185 | -8.0 | -130 |
| Younger Dryas | 12,000 | 235 | -5.0 | -65 |
| Holocene Optimum | 6,000 | 270 | +0.5 | -5 |
| Pre-industrial | 200 | 280 | -0.3 | 0 |

### 6. Geographic Determinism — Diamond's Thesis

Agents experience different agricultural potential, disease resistance,
and technology diffusion rates based on their geographic location.

**Core reference:**
- Diamond, J. (1997). *Guns, Germs, and Steel: The Fates of Human Societies*.
  W. W. Norton.

**Implementation** (`history.py: GeographicAdvantage`):

| Factor | Eurasia | Africa | Americas | Source |
|--------|---------|--------|----------|--------|
| Continental axis multiplier | 1.5x (E-W) | 0.7x (N-S) | 0.5-0.6x (N-S) | Diamond Ch. 10 |
| Domesticable large mammals | 4 (sheep, goat, cattle, pig) | 0 | 1 (llama) | Diamond Ch. 9 |
| Founder crops | 8 (wheat, barley, lentils...) | 3 (sorghum, millet, cowpea) | 3 (maize, squash, beans) | Diamond Ch. 8 |
| Disease resistance (from animal proximity) | High (zoonotic exposure) | Low | Very low | Diamond Ch. 11 |

Agricultural origins: Fertile Crescent (11,500 BP), Yellow River (10,000 BP),
Yangtze (9,000 BP), Mesoamerica (9,000 BP), Andes (8,000 BP).
Diffusion modeled at ~1 km/year along latitude, slower across climate barriers.

### 7. Evolution — Agent Adaptation

Over many generations, agent populations accumulate adaptations to local
environments (cold tolerance, altitude adaptation, disease resistance).

**Core reference:**
- Dawkins, R. (2009). *The Greatest Show on Earth: The Evidence for Evolution*.
  Transworld Publishers.

**Implementation**: Trait inheritance via crossover + mutation (mutation rate 15%).
Environmental selection pressure: agents better adapted to local temperature,
altitude, and disease environment have higher survival and reproduction rates.

---

## Module Reference

### Core Simulation

| Module | Lines | Purpose |
|--------|-------|---------|
| `agents.py` | 1,165 | Autonomous agents: JEPA cognition, physics, traits, skills, memory, social actions |
| `world.py` | 810 | World engine: tick loop, resources, businesses, settlements, scenario dispatch |
| `world_model.py` | 454 | JEPA implementation: encoder, predictor (AdaLN), SIGReg, CEM planner |
| `shared_world_model.py` | 212 | Single shared JEPA for all agents with batch encode/plan |
| `macro.py` | 493 | 14-state ODE: climate, resources, pollution, socioeconomics |
| `geopolitics.py` | 622 | Emergent nations, alliances, trade (gravity model), conflict (IFs) |
| `bridge.py` | 357 | Bidirectional coupling: agents <-> macro <-> geopolitics |
| `history.py` | 849 | 70,000-year timeline: paleoclimate, migration, Diamond, Dawkins |

### Extensions

| Module | Lines | Purpose |
|--------|-------|---------|
| `llm_module.py` | 713 | LLM social cognition (Ollama/OpenAI): trade negotiation, governance speech, social dialogue |
| `god_mode.py` | 450 | Interventional experiments: whisper, commandment, drought, plague, climate nudge |
| `scenarios.py` | 334 | Scenario A (historical) and B (present-day) configuration |
| `earth.py` | 478 | Real geography: Natural Earth land mask, Whittaker biomes, resource lookup |

### Data Generation (run once)

| Script | Purpose |
|--------|---------|
| `generate_landmask.py` | Rasterize Natural Earth 110m polygons to 0.25° land mask |
| `generate_earth_data.py` | Compute 9 Earth system grids at 0.5° (temp, precip, biome, fertility, minerals, freshwater, fossil) |
| `generate_present_day_data.py` | Fetch World Bank API + NOAA + NASA data for Scenario B |

### Performance Building Blocks

| Module | Purpose |
|--------|---------|
| `agent_state.py` | Structure-of-Arrays storage + cKDTree — benchmarked at 173 tps (2000 agents) |

### Tests

| Test | Validates |
|------|-----------|
| `test_macro.py` | BAU scenario 2025-2100: 8/8 IPCC-calibrated checks |
| `test_llm_module.py` | Fallback mode, JSON parsing, rate limiting: 9/9 |
| `test_agent_state.py` | SoA operations, KDTree, batch metabolism: 4/4 + benchmarks |

---

## Data Pipeline

### Pre-computed Data Files (in `data/`)

| File | Resolution | Source | Size |
|------|-----------|--------|------|
| `landmask.npy` | 0.25° (720x1440) | Natural Earth 110m + Shapely | 1.0 MB |
| `earth_terrain.npy` | 0.5° (360x720) | Whittaker biome diagram | 253 KB |
| `earth_temperature.npy` | 0.5° | Latitude + lapse rate + ocean currents | 2.0 MB |
| `earth_precipitation.npy` | 0.5° | ITCZ + Hadley + monsoon + orographic | 2.0 MB |
| `earth_biome.npy` | 0.5° | Whittaker: temp x precip -> 12 biomes | 253 KB |
| `earth_fertility.npy` | 0.5° | FAO GAEZ-inspired + breadbaskets | 2.0 MB |
| `earth_minerals.npy` | 0.5° | USGS provinces + tectonic belts | 2.0 MB |
| `earth_freshwater.npy` | 0.5° | Precipitation + rivers + aquifers | 2.0 MB |
| `earth_fossil_fuels.npy` | 0.5° | USGS petroleum basins | 2.0 MB |
| `ne_110m_land.geojson` | 110m | Natural Earth (public domain) | 138 KB |
| `ne_110m_rivers.geojson` | 110m | Natural Earth | 38 KB |
| `ne_110m_lakes.geojson` | 110m | Natural Earth | 37 KB |
| `present_day_*.json/npy` | 2° / country | World Bank API + NOAA + NASA | ~5 MB |

---

## Performance

### Current Benchmarks (ARM64, Python 3.11, NumPy 2.4)

| Agents | ms/tick | tps | Scenario |
|--------|---------|-----|----------|
| 25 | 28 | 35.7 | Historical (Out of Africa) |
| 37 | 39 | 25.7 | Historical (after 200 ticks) |
| 300 | 328 | 3.0 | Present Day |

### Optimizations Applied

1. **Shared JEPA World Model** — 1 model for N agents (not N copies)
2. **Tick-skipping** — CEM plan every 3 ticks, cached behavior between
3. **cKDTree** — O(log N) spatial queries (was O(N) hash grid)

### Future SoA Building Blocks (benchmarked, not yet integrated)

| Component | 2000 agents | Source |
|-----------|-------------|--------|
| Physics + spatial | 5.8 ms | `agent_state.py` |
| JEPA encode + plan | 294 ms | `shared_world_model.py` |
| **Total (projected)** | **~300 ms (3.3 tps)** | |

---

## References

### Machine Learning & AI Architecture

1. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *Meta AI Position Paper*.
2. Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. *arXiv:2603.19312*.
3. Qu, H., Morel, M., McCabe, M., Bietti, A., Lanusse, F., Ho, S., & LeCun, Y. (2026). Representation Learning for Spatiotemporal Physical Systems. *arXiv:2603.13227*.
4. Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.

### Climate Science

5. IPCC (2021). *Climate Change 2021: The Physical Science Basis* (AR6 WG1). Cambridge University Press.
6. Myhre, G., et al. (1998). New estimates of radiative forcing due to well mixed greenhouse gases. *Geophysical Research Letters* 25(14), 2715-2718.
7. Held, I. M., et al. (2010). Probing the fast and slow components of global warming. *Journal of Climate* 23(9), 2418-2427.
8. Gregory, J. M. (2000). Vertical heat transports in the ocean and their effect on time-dependent climate change. *Climate Dynamics* 16, 501-515.
9. Friedlingstein, P., et al. (2024). Global Carbon Budget 2024. *Earth System Science Data*.

### Paleoclimate & Earth History

10. EPICA Community Members (2004). Eight glacial cycles from an Antarctic ice core. *Nature* 429, 623-628.
11. Petit, J. R., et al. (1999). Climate and atmospheric history from the Vostok ice core. *Nature* 399, 429-436.
12. Jouzel, J., et al. (2007). Orbital and millennial Antarctic climate variability. *Science* 317, 793-796.
13. Marcott, S. A., et al. (2013). A reconstruction of regional and global temperature for the past 11,300 years. *Science* 339, 1198-1201.
14. Spratt, R. M., & Lisiecki, L. E. (2016). A Late Pleistocene sea level stack. *Climate of the Past* 12, 1079-1092.
15. Clark, P. U., et al. (2009). The Last Glacial Maximum. *Science* 325, 710-714.

### System Dynamics & Economics

16. Meadows, D. H., et al. (1972). *The Limits to Growth*. Universe Books.
17. Meadows, D. H., Randers, J., & Meadows, D. L. (2004). *Limits to Growth: The 30-Year Update*. Chelsea Green.
18. Dixson-Decleve, S., Gaffney, O., Ghosh, J., Randers, J., Rockstrom, J., & Stoknes, P. E. (2022). *Earth for All: A Survival Guide for Humanity*. New Society Publishers.
19. Piketty, T. (2014). *Capital in the Twenty-First Century*. Harvard University Press.
20. Nordhaus, W. D. (2017). Revisiting the social cost of carbon. *PNAS* 114(7), 1518-1523.
21. Romer, P. M. (1990). Endogenous technological change. *Journal of Political Economy* 98(5), S71-S102.
22. Hubbert, M. K. (1956). Nuclear energy and the fossil fuels. *Shell Development Company Publication* 95.

### Geography, Evolution & Civilization

23. Diamond, J. (1997). *Guns, Germs, and Steel: The Fates of Human Societies*. W. W. Norton.
24. Dawkins, R. (2009). *The Greatest Show on Earth: The Evidence for Evolution*. Transworld Publishers.
25. Stringer, C. (2012). *The Origin of Our Species*. Penguin.
26. Marshak, S. (2019). *Earth: Portrait of a Planet* (6th ed.). W. W. Norton. ISBN 978-0393640137.
27. Whittaker, R. H. (1975). *Communities and Ecosystems* (2nd ed.). Macmillan.

### Geopolitics & Conflict

28. Hughes, B. B. (2019). *International Futures (IFs): Building and Using Global Models*. Elsevier Academic Press.
29. Russett, B. (1993). *Grasping the Democratic Peace*. Princeton University Press.
30. Oneal, J. R., & Russett, B. (1999). The Kantian peace: The pacific benefits of democracy, interdependence, and international organizations, 1885–1992. *World Politics* 52(1), 1-37.
31. Tinbergen, J. (1962). *Shaping the World Economy*. Twentieth Century Fund.

### Agriculture & Resources

32. Licker, R., et al. (2010). Mind the gap: how do climate and agricultural management explain the 'yield gap' of croplands? *Global Ecology and Biogeography* 19(6), 769-782.
33. Mueller, N. D., et al. (2012). Closing yield gaps through nutrient and water management. *Nature* 490, 254-257.
34. Arndt, N. T., et al. (2017). Future global mineral resources. *Geochemical Perspectives* 6(1), 1-171.
35. Sillitoe, R. H. (2010). Porphyry copper systems. *Economic Geology* 105(1), 3-41.
36. Doll, P., et al. (2003). A global hydrological model for deriving water availability indicators. *Journal of Hydrology* 270(1-2), 105-134.
37. Vorosmarty, C. J., et al. (2010). Global threats to human water security and river biodiversity. *Nature* 467, 555-561.
38. BGR (2019). *Energy Study 2019: Data and Developments Concerning German and Global Energy Supplies*. Bundesanstalt fur Geowissenschaften und Rohstoffe.

### Ecology & Biogeography

39. Hartmann, D. L. (2016). *Global Physical Climatology* (2nd ed.). Elsevier.
40. Schneider, T., et al. (2014). Migrations and dynamics of the ITCZ. *Nature* 513, 45-53.
41. Hoskins, B. J., & Valdes, P. J. (1990). On the existence of storm-tracks. *Journal of the Atmospheric Sciences* 47(15), 1854-1864.
42. Roe, G. H. (2005). Orographic precipitation. *Annual Review of Earth and Planetary Sciences* 33, 645-671.
43. Schlenker, W., & Roberts, M. J. (2009). Nonlinear temperature effects indicate severe damages to U.S. crop yields. *PNAS* 106(37), 15594-15598.
44. Schewe, J., et al. (2014). Multimodel assessment of water scarcity under climate change. *PNAS* 111(9), 3245-3250.

### Earth Observation Data

45. NOAA Global Monitoring Laboratory — Mauna Loa CO2 record. https://gml.noaa.gov/
46. NASA GISS Surface Temperature Analysis (GISTEMP v4). https://data.giss.nasa.gov/gistemp/
47. World Bank Open Data. https://data.worldbank.org/
48. Natural Earth. https://www.naturalearthdata.com/ (public domain).
49. USGS Mineral Commodity Summaries. https://www.usgs.gov/
50. Peixoto, J. P., & Oort, A. H. (1992). *Physics of Climate*. AIP Press.

---

## License

AGPL-3.0-or-later

## Author

Dr. Gerrit Tombrink, GeoLambda GmbH
