# Third-Party Data Attributions

World Genesis ships pre-computed Earth-system data derived from public sources.
This document lists every dataset, its license, and the attribution required
when redistributing or citing simulation outputs.

---

## Geographic base data

### Natural Earth (110 m cultural & physical)
- **Files:** `data/ne_110m_land.geojson`, `data/ne_110m_rivers.geojson`,
  `data/ne_110m_lakes.geojson`, `data/landmask.npy` (derived).
- **Source:** [naturalearthdata.com](https://www.naturalearthdata.com/)
- **Licence:** Public domain. No restriction on use.
- **Attribution suggested:** "Made with Natural Earth."

## Climate & atmosphere

### NOAA Mauna Loa CO₂ record
- **Use:** Calibration of `MacroModel.CO2_PRE_INDUSTRIAL`, present-day initial
  condition (`co2_ppm = 427.0` for 2025).
- **Source:** [NOAA Global Monitoring Laboratory](https://gml.noaa.gov/ccgg/trends/)
- **Licence:** US Government public domain.
- **Attribution required:** Cite Friedlingstein et al. (2024) for the Global
  Carbon Budget figures used in emissions calibration.

### NASA GISTEMP v4 (surface temperature)
- **Use:** Present-day temperature anomaly initial condition (+1.19 °C).
- **Source:** [data.giss.nasa.gov/gistemp](https://data.giss.nasa.gov/gistemp/)
- **Licence:** US Government public domain.
- **Attribution suggested:** "GISS Surface Temperature Analysis (GISTEMP),
  version 4."

### IPCC AR6 WG1 calibration values
- **Use:** Climate sensitivity (3.0 °C/2×CO₂), feedback parameter (1.1 W/m²/°C),
  CO₂ forcing coefficient (5.35 W/m²; Myhre et al. 1998).
- **Source:** [IPCC AR6 WG1 Table 7.SM.1](https://www.ipcc.ch/report/ar6/wg1/)
- **Licence:** IPCC reports may be reproduced for non-commercial purposes
  with attribution. Calibration values used here are common scientific
  knowledge and require only standard citation.

## Paleoclimate

### EPICA Antarctic ice core CO₂ (800 ka)
- Petit, J. R. et al. (1999) and EPICA Community Members (2004).
- **Source:** PANGAEA / NOAA Paleoclimatology archives.
- **Licence:** Open scientific data; cite the original papers.

### Spratt & Lisiecki (2016) sea-level stack
- **Use:** Paleoclimate scenario sea level prior to 1850.
- **Citation:** Spratt, R. M. & Lisiecki, L. E. (2016). *Climate of the Past*
  12, 1079–1092. (CC-BY).

## Resources & geology

### USGS Mineral Commodity Summaries / Petroleum Assessment
- **Use:** `data/earth_minerals.npy`, `data/earth_fossil_fuels.npy` —
  province location and rough magnitudes.
- **Source:** [usgs.gov](https://www.usgs.gov/)
- **Licence:** US Government public domain.

### FAO GAEZ (Global Agro-Ecological Zones) — methodology only
- **Use:** Inspiration for `data/earth_fertility.npy` derivation; no FAO
  raster data is shipped, only the methodological approach (biome ×
  precipitation × temperature × known breadbaskets).
- **Citation:** Licker et al. (2010); Mueller et al. (2012).

## Socio-economic

### World Bank Open Data
- **Use:** `data/present_day_countries.json` — GDP, population, area
  per country circa 2024.
- **Source:** [data.worldbank.org](https://data.worldbank.org/)
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Attribution required:** Cite "World Bank Open Data" with retrieval date.
  Files are fetched fresh by `generate_present_day_data.py` — note the
  retrieval date in `present_day_metadata.json`.

### UN WPP / Conflict trackers
- Active-conflict geolocations in `data/present_day_conflicts.json` are
  hand-curated from publicly reported incidents (Ukraine, Gaza, Sudan,
  Myanmar, etc.). For research output, cite ACLED or UCDP for canonical
  conflict datasets — the shipped file is illustrative only.

---

## How to cite the simulation outputs

If you publish results derived from running World Genesis, please cite:

1. **The software** — see [`CITATION.cff`](../CITATION.cff) (auto-rendered
   on GitHub as a "Cite this repository" button).
2. **The relevant primary sources** — every equation in the codebase carries
   an inline citation; the master bibliography is in
   [`paper/paper.bib`](../paper/paper.bib).
3. **The pre-computed data sources above**, especially World Bank if you
   use Scenario B.

## License compatibility

The simulation code is released under **AGPL-3.0-or-later**. The shipped
data files are either public domain (Natural Earth, US Government sources)
or CC BY 4.0 (World Bank), both compatible with AGPL redistribution.
Derived works must preserve attribution to the original data sources
even where the AGPL covers the code.
