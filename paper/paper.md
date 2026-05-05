---
title: "World Genesis: A Multi-Scale Agent-Based Simulator of Human Civilization with JEPA Cognition and Coupled Earth-System Dynamics"
tags:
  - Python
  - agent-based modelling
  - climate simulation
  - JEPA
  - world model
  - macroeconomics
  - geopolitics
  - paleoclimate
authors:
  - name: Gerrit Tombrink
    affiliation: 1
    corresponding: true
affiliations:
  - name: GeoLambda GmbH, Germany
    index: 1
date: 4 May 2026
bibliography: paper.bib
---

# Summary

`World Genesis` is an open-source simulator that couples autonomous agents to
a coupled Earth-system, macroeconomic, and geopolitical model on real Earth
geography. Each agent perceives, plans, and acts through a Joint Embedding
Predictive Architecture (JEPA) world model [@lecun2022path; @maes2026leworldmodel],
while a 14-state ordinary-differential-equation (ODE) layer governs global
climate, resources, and socioeconomics. Nation-states emerge organically when
agent settlements coalesce, with bilateral trade and conflict probability
following established gravity-model and International Futures specifications.

The simulator ships with two scenarios. *Scenario A* runs from the
Out-of-Africa migration ~70,000 years before present through to climate
futures beyond 2100, integrating paleoclimate from EPICA ice cores and
geographic determinism in the sense of @diamond1997guns. *Scenario B*
initialises from present-day data (World Bank, NOAA, NASA) and projects
forward at monthly resolution. Every parameter, equation, and boundary
condition is grounded in published literature, and the macro layer passes
8/8 IPCC AR6-calibrated validation checks at 2100.

# Statement of need

Existing tools occupy disjoint corners of the design space: integrated
assessment models such as DICE [@nordhaus2017revisiting] and World3
[@meadows1972limits] deliver coherent climate-economic dynamics but treat
populations as aggregate stocks; agent-based models such as Sugarscape and
Schelling-style frameworks deliver emergent behaviour but lack physical
grounding; large language model agents now match human conversational
fidelity but have no persistent world model. `World Genesis` bridges these
traditions: a single shared JEPA world model supplies vector cognition to
hundreds of agents simultaneously, while a calibrated macro ODE supplies
the physical and economic constraints those agents inhabit. The result is
a research instrument suitable for studying emergent civilisation
dynamics, climate-policy counterfactuals, and the coupling between
individual decisions and planetary boundaries.

The package targets three audiences. First, **climate and Earth-system
researchers** who want a tractable testbed for integrated assessment
beyond representative-agent assumptions. Second, **AI researchers** studying
JEPA-style architectures [@maes2026leworldmodel; @qu2026representation] in a
non-toy setting with reward-free exploration over thousands of episodes.
Third, **economists and political scientists** investigating emergent
nation-formation, trade, and conflict from primitive agent interactions
[@hughes2019international; @russett1993grasping].

# Features

- **JEPA cognition** with adaptive layer normalisation, SIGReg
  regularisation [@maes2026leworldmodel], and a Cross-Entropy Method planner.
- **14-state macro ODE** spanning two-layer energy balance, Hubbert resource
  curves [@hubbert1956nuclear], DICE damages [@nordhaus2017revisiting], and
  Earth4All social tension [@dixsondecleve2022earth].
- **Emergent geopolitics**: nation formation by settlement coalescence;
  Tinbergen [@tinbergen1962shaping] gravity trade; logistic conflict
  probability per @hughes2019international.
- **Real Earth geography**: Natural Earth coastlines rasterised to 0.25°,
  Whittaker [@whittaker1975communities] biomes, USGS resource provinces.
- **Reproducibility**: deterministic seeded runs, structured CSV+JSON
  logging, an 8/8 IPCC validation suite, a figure-generation pipeline,
  and a published validation report.

# Validation

Running the BAU macro path 2025–2100 produces CO₂ = 504.8 ppm,
ΔT = +2.05 °C, sea-level rise = 0.572 m, and population = 8.44 B,
all within bounds derived from the IPCC SSP1-2.6 to SSP3-7.0 envelope
and UN World Population Prospects 2024. Full reference values, tolerance
bands, and the reproduction command appear in `docs/validation.md`.

# Acknowledgements

This work was developed at GeoLambda GmbH. The author thanks the open
research communities behind Natural Earth, the World Bank Open Data
initiative, NOAA GML, NASA GISS, and the IPCC for the data infrastructure
that makes calibration possible.

# References
