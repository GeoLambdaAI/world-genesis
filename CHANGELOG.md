# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-03

### Added

- Initial public release under AGPL-3.0-or-later.
- **Agent cognition** — JEPA world model (LeCun 2022; Maes et al. 2026): encoder, AdaLN predictor, SIGReg regularization, CEM planner. Shared model for batch inference across all agents.
- **Macro layer** — 14-state ODE system: climate (IPCC AR6 calibrated, 8/8 validation checks), Hubbert resource depletion, DICE damage function, Earth4All social tension, endogenous Romer technology growth.
- **Geopolitics** — Emergent nation-states from settlement coalescence, gravity-model trade (Tinbergen 1962), International Futures conflict probability with liberal-peace coupling.
- **Scenarios** — (A) 70,000-year historical from Out-of-Africa with paleoclimate, Diamond geographic determinism, and Dawkins evolutionary adaptation; (B) present-day initialized from World Bank, NOAA, and NASA data.
- **Earth system** — Real geography from Natural Earth (110m), Whittaker biome diagram, USGS resource provinces, FAO GAEZ-inspired soil fertility.
- **LLM integration** — Optional System-2 cognition for trade negotiation, governance speech, and social dialogue (Ollama / OpenAI / Mistral compatible).
- **Reproducibility** — Seeded determinism via `world.seed`, structured run logging via `sim_logger.py` writing JSON metadata + per-tick CSV.
- **Web frontend** — Flask + SocketIO real-time visualization with Leaflet.js satellite/OSM tiles.

### Security

- **Path-traversal hardening** of `/api/logger/download/<run_id>`: `run_id` is regex-validated to the `YYYYMMDD_HHMMSS` format produced by `sim_logger.py`, and the resolved path is verified to stay under the repository's `logs/` directory.
- **Loopback-by-default**: the server now binds to `127.0.0.1`. Set `BIND_HOST=0.0.0.0` explicitly to expose the unauthenticated control surface (a warning is printed in that case). Place behind an authenticated reverse proxy before exposing publicly.
- **CORS restricted** to `http://localhost:5000` and `http://127.0.0.1:5000` by default, replacing the prior allow-all policy. Override via `CORS_ALLOWED_ORIGINS` (comma-separated).
- **Ephemeral secret key**: when `FLASK_SECRET_KEY` is unset, a per-process `secrets.token_hex(32)` is generated instead of falling back to a predictable shared default. A warning is printed at startup.
- **`SECURITY.md`** added — threat model, in-scope vs. out-of-scope mitigations, operator deployment checklist, and a vulnerability-disclosure process via GitHub's private vulnerability reporting.
- Verified that `LLMModule.api_key` is not leaked through `get_status()`, `get_full_state()`, snapshot dumps, or `metadata.json`.

[Unreleased]: https://github.com/GeoLambdaAI/world-genesis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GeoLambdaAI/world-genesis/releases/tag/v0.1.0
