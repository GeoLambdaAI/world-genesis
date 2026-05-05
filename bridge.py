"""
Coupling Bridge — Connects macro dynamics <-> agent world <-> geopolitics.

Three bidirectional data flows every macro tick:
1. AGENTS -> MACRO: Agent actions aggregate into emissions, extraction, tech investment
2. MACRO -> WORLD: Climate/pollution degrades ResourceMap fertility & yields
3. GEOPOLITICS <-> BOTH: Nation policies constrain agents; conflict feeds macro stress

Design principle: the macro model changes the ENVIRONMENT (resources, fertility),
agents then autonomously adapt. No direct agent control from macro layer.

References:
- World3 resource-agriculture feedback (Meadows 2004)
- IPCC AR6 WG2 Chapter 5: food production under climate change
- Schewe et al. 2014: freshwater stress projections
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macro import MacroState
    from world import ResourceMap, Settlement, Business
    from agents import Agent


class MacroAgentBridge:
    """
    Bidirectional coupling between macro ODE system and agent simulation.

    Called every macro_update_interval ticks from World.step().
    """

    def __init__(self):
        # Track cumulative extraction between macro ticks for feedback
        self._harvest_accumulator: float = 0.0
        self._business_ticks: int = 0
        self._research_actions: int = 0
        self._total_actions: int = 0

    def reset_accumulators(self):
        """Reset per-interval accumulators. Called after each macro step."""
        self._harvest_accumulator = 0.0
        self._business_ticks = 0
        self._research_actions = 0
        self._total_actions = 0

    # ------------------------------------------------------------------
    # 1. AGENTS -> MACRO: Aggregate agent activity into macro feedback
    # ------------------------------------------------------------------

    def aggregate_agent_feedback(
        self,
        agents: list,
        businesses: list,
        settlements: list,
    ) -> dict:
        """
        Aggregate agent-level activity into macro-model feedback signals.

        Returns dict consumed by MacroModel.step():
            emission_multiplier: float    - industrial activity scaling
            extraction_multiplier: float  - resource extraction intensity
            renewable_investment: float   - green tech investment fraction
            conflict_intensity: float     - from geopolitics (set externally)
            population_factor: float      - agent reproduction rate signal
            research_boost: float         - R&D investment from agent research actions
        """
        alive = [a for a in agents if a.alive]
        n_alive = len(alive)

        if n_alive == 0:
            return self._default_feedback()

        # Population factor: agents reproducing faster -> higher pop growth
        avg_children = np.mean([a.children_count for a in alive])
        population_factor = 0.8 + 0.4 * min(1.0, avg_children / 3.0)

        # Emission multiplier: active businesses = more industrial emissions
        active_businesses = [b for b in businesses if b.active]
        n_businesses = len(active_businesses)
        # More businesses per capita = more emissions
        business_intensity = n_businesses / max(1, n_alive) * 10.0
        emission_multiplier = 0.5 + 0.5 * min(2.0, business_intensity)

        # Resource extraction: from business types and agent work actions
        extractive_types = {"farming", "mining", "crafting"}
        n_extractive = sum(1 for b in active_businesses
                           if b.business_type in extractive_types)
        extraction_multiplier = 0.7 + 0.6 * min(1.5, n_extractive / max(1, n_alive) * 10)

        # Renewable investment: fraction of businesses that are "green"
        # Research-heavy and trading businesses are proxies for service economy
        service_types = {"research", "trading", "diplomacy", "medicine"}
        n_service = sum(1 for b in active_businesses
                        if b.business_type in service_types)
        renewable_investment = n_service / max(1, n_businesses) if n_businesses > 0 else 0.0

        # Research boost: from agents doing research actions + research skill levels
        avg_research_skill = np.mean([a.skills.get_level("research") for a in alive])
        research_fraction = self._research_actions / max(1, self._total_actions)
        research_boost = (
            0.5 * avg_research_skill +
            0.5 * research_fraction
        ) * min(2.0, n_alive / 25.0)  # Scale with population

        return {
            "emission_multiplier": float(np.clip(emission_multiplier, 0.1, 3.0)),
            "extraction_multiplier": float(np.clip(extraction_multiplier, 0.1, 3.0)),
            "renewable_investment": float(np.clip(renewable_investment, 0.0, 1.0)),
            "conflict_intensity": 0.0,  # Set by geopolitics layer
            "population_factor": float(np.clip(population_factor, 0.5, 2.0)),
            "research_boost": float(np.clip(research_boost, 0.0, 2.0)),
        }

    def _default_feedback(self) -> dict:
        """Default feedback when no agents alive."""
        return {
            "emission_multiplier": 0.1,
            "extraction_multiplier": 0.1,
            "renewable_investment": 0.0,
            "conflict_intensity": 0.0,
            "population_factor": 0.5,
            "research_boost": 0.0,
        }

    def record_agent_action(self, action_type: str):
        """Called from World.step() to track agent actions between macro ticks."""
        self._total_actions += 1
        if action_type == "research":
            self._research_actions += 1

    # ------------------------------------------------------------------
    # 2. MACRO -> WORLD: Apply climate/pollution effects to resources
    # ------------------------------------------------------------------

    def apply_macro_to_world(
        self,
        macro_state: 'MacroState',
        resource_map: 'ResourceMap',
        terrain: np.ndarray,
        fertility: np.ndarray,
        elevation: np.ndarray,
    ):
        """
        Translate macro-level changes into resource map modifications.

        This is how agents FEEL climate change — through the resources they
        depend on becoming scarcer or shifting geographically.

        Effects applied:
        - Temperature -> fertility reduction (latitude-dependent)
        - Pollution -> food regeneration reduction
        - Sea level rise -> coastal cell flooding
        - Freshwater stress -> water regeneration reduction
        - Resource depletion -> global capacity scaling
        """
        T = macro_state.temperature_anomaly
        pollution = macro_state.persistent_pollution
        slr = macro_state.sea_level_rise_m
        freshwater = macro_state.freshwater_stress
        fossil_remaining = macro_state.fossil_fuels
        mineral_remaining = macro_state.minerals_global
        tech = macro_state.technology_level

        rows, cols = terrain.shape

        for r in range(rows):
            lat = resource_map.lat_max - (r + 0.5) * resource_map.cell_size_deg
            abs_lat = abs(lat)

            for c in range(cols):
                t_type = terrain[r, c]

                # Skip ocean
                if t_type == 0:
                    continue

                base_fertility = fertility[r, c]
                elev = elevation[r, c]

                # --- Temperature impact on fertility ---
                # Source: Schlenker & Roberts 2009, IPCC AR6 WG2 Ch5
                if abs_lat < 20:
                    # Tropical: strongly affected above 1.5C
                    temp_factor = max(0.3, 1.0 - 0.15 * max(0, T - 1.5) ** 1.3)
                elif abs_lat < 45:
                    # Temperate: slight benefit then decline
                    # Optimal warming ~1C, damage above 2C
                    if T < 1.5:
                        temp_factor = 1.0 + 0.05 * T  # Slight benefit
                    else:
                        temp_factor = max(0.4, 1.075 - 0.08 * (T - 1.5) ** 1.5)
                elif abs_lat < 60:
                    # Subarctic: benefits from warming initially
                    temp_factor = min(1.3, 1.0 + 0.1 * T)
                else:
                    # Arctic/Antarctic: permafrost thaw effects
                    temp_factor = max(0.5, 1.0 - 0.05 * max(0, T - 2.0))

                # --- Pollution impact on food regen ---
                # Source: World3 pollution-agriculture link (Meadows 2004)
                pollution_factor = max(0.4, 1.0 - 0.4 * pollution)

                # --- Sea level rise: flood low-elevation coastal cells ---
                # Source: IPCC AR6 WG1 Ch9
                if elev < 0.15 and slr > 0.3:
                    flood_prob = min(1.0, (slr - 0.3) / 0.5)
                    if flood_prob > 0.5:
                        # Severe flooding -> reduce all resources dramatically
                        temp_factor *= max(0.1, 1.0 - flood_prob)

                # --- Freshwater stress ---
                # Source: Schewe et al. 2014
                # Arid regions hit hardest
                if t_type == 4:  # Desert
                    water_factor = max(0.2, 1.0 - 0.8 * freshwater)
                elif t_type == 1:  # Plains
                    water_factor = max(0.5, 1.0 - 0.4 * freshwater)
                else:
                    water_factor = max(0.6, 1.0 - 0.2 * freshwater)

                # --- Apply combined modifier to regeneration rates ---
                combined = temp_factor * pollution_factor * water_factor

                # Scale food regeneration
                original_food_regen = base_fertility * 2.0 if t_type == 1 else base_fertility * 1.0
                resource_map.food_regen[r, c] = original_food_regen * combined

                # Water regeneration affected by freshwater stress
                resource_map.water_regen[r, c] *= water_factor

                # Mineral availability scales with global remaining stock
                resource_map.minerals_regen[r, c] *= mineral_remaining

        # --- Global resource capacity scaling ---
        # As global stocks deplete, local max capacity also drops
        # This creates scarcity pressure that agents experience
        resource_map.food = np.minimum(
            resource_map.food,
            100.0 * macro_state.food_production_index
        )
        resource_map.minerals = np.minimum(
            resource_map.minerals,
            100.0 * mineral_remaining
        )

    # ------------------------------------------------------------------
    # 3. GEOPOLITICS -> AGENTS: Apply geopolitical effects
    # ------------------------------------------------------------------

    def apply_geopolitics_to_agents(
        self,
        geopolitics,
        agents: list,
        world,
    ):
        """
        Apply geopolitical effects to individual agents.

        Effects:
        - Conflict zones: agents lose health and wealth
        - Trade agreements: allied nation agents get trade bonuses
        - Sanctions: sanctioned nation agents get trade penalties
        - Technology diffusion: agents in advanced nations learn faster
        """
        if geopolitics is None or not geopolitics.nations:
            return

        alive = [a for a in agents if a.alive]

        # Build nation membership lookup: agent_id -> NationState
        agent_nation = {}
        for nation in geopolitics.nations:
            for sid in nation.settlement_ids:
                for s in world.settlements:
                    if s.id == sid:
                        for member_id in s.members:
                            agent_nation[member_id] = nation

        # Apply conflict effects
        for conflict in geopolitics.active_conflicts:
            zone_lat = conflict.get("lat", 0)
            zone_lng = conflict.get("lng", 0)
            zone_radius = conflict.get("radius", 5.0)
            intensity = conflict.get("intensity", 0.5)

            for agent in alive:
                dist = world._distance_deg(agent.lat, agent.lng, zone_lat, zone_lng)
                if dist < zone_radius:
                    # Proximity-scaled damage
                    proximity = 1.0 - dist / zone_radius
                    agent.health -= 2.0 * intensity * proximity
                    agent.wealth -= 1.0 * intensity * proximity
                    agent.happiness -= 3.0 * intensity * proximity

        # Technology diffusion: agents in nations with high tech benefit
        for agent in alive:
            nation = agent_nation.get(agent.id)
            if nation and nation.technology_level > 1.2:
                # Skill learning bonus from national tech infrastructure
                tech_bonus = (nation.technology_level - 1.0) * 0.01
                agent.skills.practice("research", tech_bonus, 1.0)

    # ------------------------------------------------------------------
    # Helper: Inject macro signals into agent observation
    # ------------------------------------------------------------------

    def get_macro_local_state(
        self,
        macro_state: 'MacroState',
        lat: float,
        lng: float,
        geopolitics=None,
        world=None,
    ) -> dict:
        """
        Get macro-derived fields for get_local_state().
        Extends the agent's observation with global context.
        """
        state = {
            "temperature_anomaly": macro_state.temperature_anomaly,
            "co2_level": macro_state.co2_ppm,
            "social_tension": macro_state.social_tension,
            "pollution_level": macro_state.persistent_pollution,
            "resource_scarcity": 1.0 - macro_state.fossil_fuels,
            "conflict_nearby": 0.0,
            "nation_tech_level": macro_state.technology_level,
            "trade_access": 0.5,
        }

        # Check for nearby conflicts
        if geopolitics and geopolitics.active_conflicts:
            for conflict in geopolitics.active_conflicts:
                dist = np.sqrt(
                    (lat - conflict.get("lat", 0)) ** 2 +
                    (lng - conflict.get("lng", 0)) ** 2
                )
                if dist < conflict.get("radius", 5.0):
                    state["conflict_nearby"] = max(
                        state["conflict_nearby"],
                        conflict.get("intensity", 0.5) * (1.0 - dist / conflict["radius"])
                    )

        # Nation-specific tech and trade access
        if geopolitics and world:
            for nation in geopolitics.nations:
                for sid in nation.settlement_ids:
                    for s in world.settlements:
                        if s.id == sid:
                            dist = np.sqrt((lat - s.lat)**2 + (lng - s.lng)**2)
                            if dist < 5.0:
                                state["nation_tech_level"] = nation.technology_level
                                state["trade_access"] = nation.trade_openness
                                break

        return state
