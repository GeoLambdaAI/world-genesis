"""
World Genesis — simulation engine.

Manages:
- Real Earth geography with lat/lng coordinate system
- Biome-based terrain from coordinates (climate model)
- Resource spawning and depletion on Earth grid
- Business/economic system
- Society/governance structures
- Spatial indexing for agent interactions
- Complete simulation tick logic
"""

import numpy as np
from typing import Optional
from agents import Agent
from earth import (TerrainType, classify_terrain, get_fertility, is_land,
                   find_land_spawn_points, generate_earth_grid)
from macro import MacroModel
from geopolitics import GeopoliticalSystem
from bridge import MacroAgentBridge
from history import HistoricalSimulation, get_era, get_spawn_locations
from llm_module import LLMModule, LLMConfig
from god_mode import GodMode, GodModeConfig
from scenarios import SCENARIOS, ScenarioLoader, ScenarioConfig
from shared_world_model import SharedWorldModel
from sim_logger import SimulationLogger, LoggerConfig


# ============================================================================
# Resource System (Earth Grid)
# ============================================================================

class ResourceMap:
    """Grid-based resource map over Earth with regeneration."""

    def __init__(self, earth_grid: dict):
        self.rows = earth_grid["rows"]
        self.cols = earth_grid["cols"]
        self.lats = earth_grid["lats"]
        self.lngs = earth_grid["lngs"]
        self.cell_size_deg = earth_grid["cell_size_deg"]
        self.lat_min = earth_grid["lat_min"]
        self.lat_max = earth_grid["lat_max"]
        self.lng_min = earth_grid["lng_min"]
        self.lng_max = earth_grid["lng_max"]

        # Resource layers
        self.food = np.zeros((self.rows, self.cols))
        self.minerals = np.zeros((self.rows, self.cols))
        self.wood = np.zeros((self.rows, self.cols))
        self.water = np.zeros((self.rows, self.cols))

        # Regeneration rates
        self.food_regen = np.zeros((self.rows, self.cols))
        self.minerals_regen = np.zeros((self.rows, self.cols))
        self.wood_regen = np.zeros((self.rows, self.cols))
        self.water_regen = np.zeros((self.rows, self.cols))

    def initialize_from_terrain(self, terrain: np.ndarray, fertility: np.ndarray,
                               minerals_grid=None, freshwater_grid=None,
                               fossil_grid=None):
        """
        Set initial resources based on terrain, fertility, and Earth system data.

        Uses pre-computed mineral, freshwater, and fossil fuel grids when available.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                t = terrain[r, c]
                f = fertility[r, c]

                # Mineral richness from Earth system data or terrain default
                m = minerals_grid[r, c] if minerals_grid is not None else (0.5 if t == 3 else 0.2)
                # Freshwater from Earth system data or terrain default
                w = freshwater_grid[r, c] if freshwater_grid is not None else 0.5

                if t == TerrainType.PLAINS:
                    self.food[r, c] = 80 * f
                    self.food_regen[r, c] = 2.0 * f
                    self.water[r, c] = 60 * w
                    self.water_regen[r, c] = 1.0 * w
                    self.minerals[r, c] = 30 * m
                    self.minerals_regen[r, c] = 0.2 * m
                elif t == TerrainType.FOREST:
                    self.food[r, c] = 40 * f
                    self.food_regen[r, c] = 1.0 * f
                    self.wood[r, c] = 100 * f
                    self.wood_regen[r, c] = 1.5 * f
                    self.water[r, c] = 70 * w
                    self.water_regen[r, c] = 1.5 * w
                    self.minerals[r, c] = 20 * m
                elif t == TerrainType.MOUNTAINS:
                    self.minerals[r, c] = 100 * m
                    self.minerals_regen[r, c] = 0.5 * m
                    self.water[r, c] = 40 * w
                    self.water_regen[r, c] = 0.5 * w
                    self.food[r, c] = 10 * f
                    self.food_regen[r, c] = 0.2 * f
                elif t == TerrainType.DESERT:
                    self.minerals[r, c] = 50 * m
                    self.minerals_regen[r, c] = 0.3 * m
                    self.water[r, c] = 5 * w
                    self.water_regen[r, c] = 0.1 * w
                    self.food[r, c] = 5 * f
                    self.food_regen[r, c] = 0.1 * f
                elif t == TerrainType.TUNDRA:
                    self.food[r, c] = 10 * f
                    self.food_regen[r, c] = 0.3 * f
                    self.water[r, c] = 50 * w
                    self.water_regen[r, c] = 0.5 * w
                    self.minerals[r, c] = 40 * m
                    self.minerals_regen[r, c] = 0.3 * m

    def get_cell(self, lat: float, lng: float) -> tuple[int, int]:
        """Convert lat/lng to grid row/col."""
        r = int(np.clip((self.lat_max - lat) / self.cell_size_deg,
                        0, self.rows - 1))
        c = int(np.clip((lng - self.lng_min) / self.cell_size_deg,
                        0, self.cols - 1))
        return r, c

    def harvest(self, lat: float, lng: float, resource_type: str, amount: float) -> float:
        r, c = self.get_cell(lat, lng)
        layer = getattr(self, resource_type, None)
        if layer is None:
            return 0.0
        available = layer[r, c]
        taken = min(available, amount)
        layer[r, c] -= taken
        return float(taken)

    def get_local(self, lat: float, lng: float) -> dict:
        r, c = self.get_cell(lat, lng)
        return {
            "food": float(self.food[r, c]),
            "minerals": float(self.minerals[r, c]),
            "wood": float(self.wood[r, c]),
            "water": float(self.water[r, c]),
        }

    def regenerate(self):
        """Regenerate resources each tick."""
        self.food = np.minimum(self.food + self.food_regen * 0.1, 100.0)
        self.minerals = np.minimum(self.minerals + self.minerals_regen * 0.05, 100.0)
        self.wood = np.minimum(self.wood + self.wood_regen * 0.08, 100.0)
        self.water = np.minimum(self.water + self.water_regen * 0.1, 100.0)

    def to_grid_data(self) -> dict:
        """Serialize for UI (only non-zero cells to save bandwidth)."""
        return {
            "food": self.food.tolist(),
            "minerals": self.minerals.tolist(),
            "wood": self.wood.tolist(),
            "water": self.water.tolist(),
            "cell_size_deg": self.cell_size_deg,
            "cols": self.cols,
            "rows": self.rows,
        }


# ============================================================================
# Business / Economy
# ============================================================================

class Business:
    _next_id = 0

    def __init__(self, owner_id: int, lat: float, lng: float,
                 business_type: str, capital: float):
        Business._next_id += 1
        self.id = Business._next_id
        self.owner_id = owner_id
        self.lat = lat
        self.lng = lng
        self.business_type = business_type
        self.capital = capital
        self.revenue = 0.0
        self.employees: list[int] = []
        self.age = 0
        self.reputation = 0.5
        self.active = True

    def operate(self, world) -> float:
        if not self.active:
            return 0.0
        self.age += 1

        workforce = len(self.employees) + 1
        base_revenue = workforce * 2.0 * self.reputation

        if self.business_type in ("farming", "mining", "crafting"):
            resource_type = {"farming": "food", "mining": "minerals",
                             "crafting": "wood"}.get(self.business_type, "food")
            harvested = world.harvest_resource(self.lat, self.lng, resource_type, workforce * 3)
            base_revenue += harvested * 1.5

        revenue = base_revenue * (1.0 + self.capital * 0.001)

        wage_costs = len(self.employees) * 1.5
        operating_costs = 0.5

        profit = revenue - wage_costs - operating_costs
        self.capital += profit
        self.revenue = revenue
        self.reputation = min(1.0, self.reputation + 0.001 * (1 if profit > 0 else -1))

        if self.capital <= 0:
            self.active = False

        return profit

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "lat": round(self.lat, 4),
            "lng": round(self.lng, 4),
            "type": self.business_type,
            "capital": round(self.capital, 1),
            "revenue": round(self.revenue, 1),
            "employees": len(self.employees),
            "age": self.age,
            "reputation": round(self.reputation, 3),
            "active": self.active,
        }


# ============================================================================
# Society / Governance
# ============================================================================

class Settlement:
    """A cluster of agents forming a community."""
    _next_id = 0

    def __init__(self, lat: float, lng: float, founder_id: int):
        Settlement._next_id += 1
        self.id = Settlement._next_id
        self.lat = lat
        self.lng = lng
        self.founder_id = founder_id
        self.members: set[int] = {founder_id}
        self.name = self._generate_name()
        self.population = 1
        self.culture_values: dict[str, float] = {
            "cooperation": 0.5,
            "innovation": 0.5,
            "tradition": 0.5,
            "militarism": 0.2,
            "trade_openness": 0.5,
        }
        self.governance_type = "tribal"
        self.leader_id: Optional[int] = founder_id
        self.laws: list[str] = []
        self.tax_rate = 0.05
        self.treasury = 0.0
        self.age = 0

    def _generate_name(self) -> str:
        prefixes = ["New", "Fort", "Port", "Lake", "Mount", "Green", "Iron",
                     "Gold", "Silver", "Crystal", "Shadow", "Sun", "Star"]
        suffixes = ["haven", "burg", "ton", "dale", "ford", "bridge", "gate",
                     "wood", "field", "peak", "vale", "shore", "hollow"]
        return np.random.choice(prefixes) + np.random.choice(suffixes)

    def update(self, agents: list):
        self.age += 1
        living_members = [a for a in agents if a.id in self.members and a.alive]
        self.population = len(living_members)

        if self.population == 0:
            return

        for trait_name, culture_key in [
            ("cooperation", "cooperation"),
            ("creativity", "innovation"),
            ("risk_tolerance", "militarism"),
        ]:
            avg_trait = np.mean([a.traits[trait_name] for a in living_members])
            self.culture_values[culture_key] = (
                0.95 * self.culture_values[culture_key] + 0.05 * avg_trait
            )

        if self.population >= 20 and self.governance_type == "tribal":
            self.governance_type = "council"
        elif self.population >= 50 and self.governance_type == "council":
            self.governance_type = "republic"
        elif self.population >= 100 and self.governance_type == "republic":
            if self.culture_values["cooperation"] > 0.6:
                self.governance_type = "democracy"

        for agent in living_members:
            tax = agent.wealth * self.tax_rate * 0.01
            agent.wealth -= tax
            self.treasury += tax

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "lat": round(self.lat, 4),
            "lng": round(self.lng, 4),
            "population": self.population,
            "governance": self.governance_type,
            "culture": {k: round(v, 3) for k, v in self.culture_values.items()},
            "leader_id": self.leader_id,
            "treasury": round(self.treasury, 1),
            "tax_rate": self.tax_rate,
            "age": self.age,
        }


# ============================================================================
# World Engine
# ============================================================================

class World:
    """
    Earth simulation managing terrain, agents, resources, businesses,
    and settlements using real-world coordinates. Planet-agnostic by
    design — alternative scenarios (e.g. Mars) reuse this engine with
    different terrain and climate inputs.
    """

    def __init__(self, seed: int = 42, cell_size_deg: float = 2.0,
                 config: dict = None, scenario_id: str = "historical"):
        self.seed = seed
        self.config = config or {}
        self.rng = np.random.RandomState(seed)
        self.tick = 0
        self.cell_size_deg = cell_size_deg

        # Scenario
        self.scenario = SCENARIOS.get(scenario_id, SCENARIOS["historical"])
        self.scenario_loader = ScenarioLoader()
        self.macro_always_active = self.scenario.macro_active_from_start

        # Generate Earth terrain grid
        self.earth_grid = generate_earth_grid(
            lat_min=-60, lat_max=75,
            lng_min=-180, lng_max=180,
            cell_size_deg=cell_size_deg,
            seed=seed
        )

        self.terrain = self.earth_grid["terrain"]
        self.elevation = self.earth_grid["elevation"]
        self.fertility = self.earth_grid["fertility"]

        # Resources — initialized from Earth system data
        self.resources = ResourceMap(self.earth_grid)
        self.resources.initialize_from_terrain(
            self.terrain, self.fertility,
            minerals_grid=self.earth_grid.get("minerals"),
            freshwater_grid=self.earth_grid.get("freshwater"),
            fossil_grid=self.earth_grid.get("fossil_fuels"),
        )

        # Entities
        self.agents: list[Agent] = []
        self.businesses: list[Business] = []
        self.settlements: list[Settlement] = []

        # Statistics
        self.stats_history: list[dict] = []

        # Spatial index: cKDTree for fast neighbor lookups
        from scipy.spatial import cKDTree
        self._kdtree: Optional[cKDTree] = None
        self._kdtree_alive: Optional[list] = None  # Alive agents for index mapping

        # Historical simulation (70,000 years of human civilization)
        self.start_year_bp = self.config.get("start_year_bp", 70000)
        self.history = HistoricalSimulation(start_year_bp=self.start_year_bp)

        # Macro dynamics (Club of Rome / Earth4All) — activates in Industrial+ era
        self.macro = MacroModel(config={"dt_years": 1.0 / 12.0})
        self.geopolitics = GeopoliticalSystem(rng=self.rng)
        self.bridge = MacroAgentBridge()
        self.macro_update_interval = 10  # ticks between macro updates

        # Shared JEPA world model — all agents use this single model
        self.shared_world_model = SharedWorldModel(obs_dim=40, action_dim=8, latent_dim=24)
        import agents as _agents_module
        _agents_module._shared_world_model = self.shared_world_model

        # LLM Social Cognition (disabled by default — toggle via UI)
        self.llm = LLMModule(LLMConfig(enabled=False))

        # God Mode interventions (disabled by default)
        self.god_mode = GodMode(GodModeConfig(enabled=False))

        # Recent dialogues ring buffer for UI
        self.recent_dialogues: list[dict] = []

        # Scientific logger
        self.logger = SimulationLogger(LoggerConfig(enabled=True))

    # ------------------------------------------------------------------
    # Agent Management
    # ------------------------------------------------------------------

    def spawn_initial_agents(self, count: int = 25):
        """Spawn initial population based on scenario."""
        if self.scenario.id == "present_day":
            self.scenario_loader.configure_world(self, self.scenario)
            self._rebuild_spatial_grid()
            return

        year_bp = self.history.year_bp

        if year_bp > 5000:
            # Historical mode: spawn near migration waypoints for current era
            spawn_points = get_spawn_locations(year_bp, count)
            # Validate all points are on habitable land (not ice, not ocean)
            from earth import is_land
            validated = []
            for lat, lng in spawn_points:
                if is_land(lat, lng) and not self.history.paleoclimate.get_ice_mask(year_bp, lat, lng):
                    validated.append((lat, lng))
                else:
                    # Retry near origin point
                    for _ in range(10):
                        jlat = lat + self.rng.normal(0, 5)
                        jlng = lng + self.rng.normal(0, 5)
                        if is_land(jlat, jlng) and not self.history.paleoclimate.get_ice_mask(year_bp, jlat, jlng):
                            validated.append((jlat, jlng))
                            break
            spawn_points = validated[:count]
        else:
            # Modern era: spread across habitable land
            spawn_points = find_land_spawn_points(count, self.seed)

        for lat, lng in spawn_points:
            agent = Agent(lat, lng)
            agent.energy = 80 + self.rng.random() * 20
            agent.wealth = 10 + self.rng.random() * 20
            self.agents.append(agent)

        self._rebuild_spatial_grid()

        # Start scientific logger
        self.logger.start_run(self)

    def add_agent(self, agent: Agent):
        self.agents.append(agent)
        nearest = self._find_nearest_settlement(agent.lat, agent.lng)
        if nearest and self._distance_deg(agent.lat, agent.lng, nearest.lat, nearest.lng) < 3.0:
            nearest.members.add(agent.id)

    def _rebuild_spatial_grid(self):
        """Rebuild cKDTree from alive agent positions. O(N log N)."""
        from scipy.spatial import cKDTree
        alive = [a for a in self.agents if a.alive]
        self._kdtree_alive = alive
        if alive:
            positions = np.array([[a.lat, a.lng] for a in alive])
            self._kdtree = cKDTree(positions)
        else:
            self._kdtree = None

    def get_nearby_agents(self, lat: float, lng: float, radius: float) -> list:
        """Fast spatial query using cKDTree. O(log N) per query."""
        if self._kdtree is None or not self._kdtree_alive:
            return []
        indices = self._kdtree.query_ball_point([lat, lng], radius)
        return [self._kdtree_alive[i] for i in indices]

    def get_local_state(self, lat: float, lng: float) -> dict:
        """Get the local world state visible to an agent."""
        resources = self.resources.get_local(lat, lng)
        nearby = self.get_nearby_agents(lat, lng, 5.0)  # ~5 degrees radius
        r, c = self.resources.get_cell(lat, lng)
        r = min(r, self.terrain.shape[0] - 1)
        c = min(c, self.terrain.shape[1] - 1)

        avg_wealth = np.mean([a.wealth for a in nearby]) if nearby else 0
        social_trust = np.mean([a.traits["cooperation"] for a in nearby]) if nearby else 0.5

        settlement = self._find_nearest_settlement(lat, lng)
        gov_stability = 0.5
        if settlement and self._distance_deg(lat, lng, settlement.lat, settlement.lng) < 5.0:
            gov_stability = min(1.0, settlement.age / 200.0 + 0.3)

        # Nearest agent direction
        nearest_dx, nearest_dy = 0.0, 0.0
        others = [a for a in nearby if self._distance_deg(lat, lng, a.lat, a.lng) > 0.01]
        if others:
            nearest = min(others, key=lambda a: self._distance_deg(lat, lng, a.lat, a.lng))
            dist = self._distance_deg(lat, lng, nearest.lat, nearest.lng)
            if dist > 0:
                nearest_dx = (nearest.lng - lng) / dist
                nearest_dy = (nearest.lat - lat) / dist

        state = {
            "local_food": resources["food"],
            "local_minerals": resources["minerals"],
            "local_wood": resources["wood"],
            "local_water": resources["water"],
            "nearby_agents": len(nearby),
            "local_demand": max(0, len(nearby) * 2 - resources["food"]) / 50.0,
            "local_supply": sum(resources.values()) / 200.0,
            "avg_wealth": avg_wealth,
            "terrain_type": int(self.terrain[r, c]),
            "fertility": float(self.fertility[r, c]),
            "elevation": float(self.elevation[r, c]),
            "social_trust": social_trust,
            "governance_stability": gov_stability,
            "nearest_agent_dx": nearest_dx,
            "nearest_agent_dy": nearest_dy,
        }

        # Add macro-derived signals for agent observations
        macro_state = self.bridge.get_macro_local_state(
            self.macro.state, lat, lng, self.geopolitics, self
        )
        state.update(macro_state)

        return state

    def harvest_resource(self, lat: float, lng: float, resource_type: str, amount: float) -> float:
        return self.resources.harvest(lat, lng, resource_type, amount)

    # ------------------------------------------------------------------
    # Business Management
    # ------------------------------------------------------------------

    def create_business(self, owner_id: int, lat: float, lng: float,
                        business_type: str, capital: float) -> dict:
        biz = Business(owner_id, lat, lng, business_type, capital)
        self.businesses.append(biz)
        return biz.to_dict()

    # ------------------------------------------------------------------
    # Settlement Management
    # ------------------------------------------------------------------

    def _find_nearest_settlement(self, lat: float, lng: float) -> Optional[Settlement]:
        if not self.settlements:
            return None
        dists = [self._distance_deg(lat, lng, s.lat, s.lng) for s in self.settlements]
        return self.settlements[int(np.argmin(dists))]

    def _check_settlement_formation(self):
        if self.tick % 50 != 0:
            return

        for agent in self.agents:
            if not agent.alive:
                continue
            in_settlement = any(agent.id in s.members for s in self.settlements)
            if in_settlement:
                continue

            nearby = self.get_nearby_agents(agent.lat, agent.lng, 3.0)
            nearby_unaffiliated = [
                a for a in nearby
                if a.id != agent.id
                and not any(a.id in s.members for s in self.settlements)
            ]

            if len(nearby_unaffiliated) >= 4:
                cx = np.mean([a.lat for a in nearby_unaffiliated + [agent]])
                cy = np.mean([a.lng for a in nearby_unaffiliated + [agent]])
                settlement = Settlement(cx, cy, agent.id)
                for a in nearby_unaffiliated:
                    settlement.members.add(a.id)
                self.settlements.append(settlement)

    def _spawn_migration_frontier(self):
        """
        Spawn agents at the frontier of human migration based on
        historical migration waves from history.py.

        Represents population growth at migration frontiers — new groups
        splitting off from existing populations and pushing into new territories.
        """
        from history import MIGRATION_WAVES
        from earth import is_land

        year_bp = self.history.year_bp
        alive = [a for a in self.agents if a.alive]

        # Don't spawn if population is already large
        if len(alive) > 200:
            return

        # Find migration waves that should have reached by now
        for wave in MIGRATION_WAVES:
            if year_bp > wave["year_bp"]:
                continue  # Not yet reached

            # Check if any agent is already near this wave point
            wave_lat, wave_lng = wave["lat"], wave["lng"]
            nearby = self.get_nearby_agents(wave_lat, wave_lng, 10.0)
            if nearby:
                continue  # Already have agents there

            # Spawn 1-3 frontier agents near this wave point
            n_spawn = self.rng.randint(1, 4)
            for _ in range(n_spawn):
                lat = wave_lat + self.rng.normal(0, 3)
                lng = wave_lng + self.rng.normal(0, 3)
                if is_land(lat, lng) and not self.history.paleoclimate.get_ice_mask(year_bp, lat, lng):
                    agent = Agent(lat, lng)
                    agent.energy = 80 + self.rng.random() * 20
                    agent.wealth = 5 + self.rng.random() * 10
                    self.agents.append(agent)

    def _apply_ice_age_effects(self):
        """Apply paleoclimate ice sheet coverage to terrain and resources."""
        year_bp = self.history.year_bp
        climate = self.history.paleoclimate.get_climate(year_bp)

        for r in range(self.resources.rows):
            lat = self.resources.lat_max - (r + 0.5) * self.resources.cell_size_deg
            for c in range(self.resources.cols):
                lng = self.resources.lng_min + (c + 0.5) * self.resources.cell_size_deg

                if self.history.paleoclimate.get_ice_mask(year_bp, lat, lng):
                    # Under ice: zero resources
                    self.resources.food[r, c] = 0
                    self.resources.food_regen[r, c] = 0
                    self.resources.wood[r, c] = 0
                    self.resources.wood_regen[r, c] = 0
                    self.resources.water[r, c] = 0
                else:
                    # Apply global temperature offset to fertility
                    temp_offset = climate["temperature_anomaly"]
                    if temp_offset < -2:
                        # Cold era: reduced food regeneration
                        cold_factor = max(0.3, 1.0 + temp_offset * 0.08)
                        self.resources.food_regen[r, c] *= cold_factor

    @staticmethod
    def _distance_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Distance in degrees (approximate, fast)."""
        return float(np.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2))

    # ------------------------------------------------------------------
    # Simulation Tick
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """Execute one simulation tick."""
        self.tick += 1
        events = []

        # Start tick timer for logger
        self.logger.tick_start()

        # Reset LLM tick counter
        if self.llm.config.enabled:
            self.llm.reset_tick_counter()

        # Process God Mode active effects
        if self.god_mode.config.enabled:
            self.god_mode.update(self)

        # Rebuild spatial grid every tick for fast lookups
        self._rebuild_spatial_grid()

        # Update all agents
        for agent in self.agents:
            result = agent.update(self)
            if result:
                events.append(result)
                # Track actions for macro feedback
                if result.get("event") == "action":
                    self.bridge.record_agent_action(result.get("action", ""))

        # Operate businesses
        for biz in self.businesses:
            biz.operate(self)

        # Update settlements
        for settlement in self.settlements:
            settlement.update(self.agents)

        # Check for new settlements
        self._check_settlement_formation()

        # Regenerate resources
        self.resources.regenerate()

        # Remove dead businesses
        self.businesses = [b for b in self.businesses if b.active or b.age < 100]

        # Train shared world model centrally (every 20 ticks)
        if self.tick % 20 == 0 and len(self.shared_world_model.experience_buffer) > 32:
            self.shared_world_model.train_step(
                batch_size=min(64, len(self.shared_world_model.experience_buffer))
            )

        # Collect agent dialogues for UI
        for agent in self.agents:
            if agent.alive and agent.last_dialogue:
                self.recent_dialogues.append({
                    "tick": self.tick,
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "text": agent.last_dialogue[:200],
                    "action": agent.current_action,
                })
                agent.last_dialogue = None
        if len(self.recent_dialogues) > 50:
            self.recent_dialogues = self.recent_dialogues[-50:]

        # ---- Advance historical timeline ----
        history_result = self.history.advance_time(1)
        current_era = self.history.get_current_era()

        # Set era time scale so agents can scale their movement
        self._era_time_scale = current_era.time_scale

        # Migration wave spawning: periodically add agents at frontier locations
        # This represents the leading edge of human migration across continents
        if self.tick % 20 == 0 and self.history.year_bp > 5000:
            self._spawn_migration_frontier()

        # Apply paleoclimate ice effects to resources periodically
        if self.tick % 50 == 0:
            self._apply_ice_age_effects()

        # ---- Macro + Geopolitics integration (every N ticks) ----
        # Macro ODE system: active from start in present_day scenario, else Industrial+ era
        is_modern = self.macro_always_active or self.history.year_bp < 200
        if self.tick % self.macro_update_interval == 0 and is_modern:
            # 1. Aggregate agent actions -> macro feedback
            feedback = self.bridge.aggregate_agent_feedback(
                self.agents, self.businesses, self.settlements
            )
            # Inject conflict intensity from geopolitics
            feedback["conflict_intensity"] = self.geopolitics.get_conflict_intensity()

            # 2. Advance macro model by one step
            self.macro.step(feedback)

            # 3. Apply macro effects to world resources
            self.bridge.apply_macro_to_world(
                self.macro.state, self.resources,
                self.terrain, self.fertility, self.elevation
            )

            # 4. Update geopolitics
            self.geopolitics.update(
                self.settlements, self.agents, self.macro.state
            )

            # 5. Apply geopolitical effects to agents
            self.bridge.apply_geopolitics_to_agents(
                self.geopolitics, self.agents, self
            )

            # 6. Reset accumulators
            self.bridge.reset_accumulators()

        # Collect statistics
        alive_agents = [a for a in self.agents if a.alive]
        stats = {
            "tick": self.tick,
            "population": len(alive_agents),
            "total_born": len(self.agents),
            "avg_energy": float(np.mean([a.energy for a in alive_agents])) if alive_agents else 0,
            "avg_wealth": float(np.mean([a.wealth for a in alive_agents])) if alive_agents else 0,
            "avg_happiness": float(np.mean([a.happiness for a in alive_agents])) if alive_agents else 0,
            "avg_age": float(np.mean([a.age for a in alive_agents])) if alive_agents else 0,
            "max_generation": max((a.generation for a in alive_agents), default=0),
            "businesses": len([b for b in self.businesses if b.active]),
            "settlements": len(self.settlements),
            "events": events[:20],
            "history": self.history.get_summary(),
            "macro": self.macro.get_summary() if is_modern else {},
            "geopolitics": self.geopolitics.get_summary(),
            "llm": self.llm.get_status(),
            "god_mode": self.god_mode.get_status(),
        }
        self.stats_history.append(stats)
        if len(self.stats_history) > 2000:
            self.stats_history = self.stats_history[-1000:]

        # Scientific logging
        self.logger.log_tick(self, stats)

        return stats

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_full_state(self) -> dict:
        """Get complete world state for UI rendering."""
        alive_agents = [a for a in self.agents if a.alive]
        return {
            "tick": self.tick,
            "agents": [a.to_dict() for a in alive_agents],
            "businesses": [b.to_dict() for b in self.businesses if b.active],
            "settlements": [s.to_dict() for s in self.settlements],
            "stats": self.stats_history[-1] if self.stats_history else {},
            "stats_history": self.stats_history[-200:],
            "history": self.history.get_summary(),
            "macro": self.macro.get_summary(),
            "geopolitics": self.geopolitics.get_summary(),
            "nations": self.geopolitics.get_nations_list(),
            "conflicts": self.geopolitics.active_conflicts,
            "scenario": {"id": self.scenario.id, "name": self.scenario.name},
            "dialogues": self.recent_dialogues[-20:],
            "llm_status": self.llm.get_status(),
            "god_mode_status": self.god_mode.get_status(),
            "logger": self.logger.get_status(),
        }
