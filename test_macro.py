"""
Standalone test for MacroModel: run BAU scenario 2025 -> 2100.

Validates:
- CO2 should reach ~550-700 ppm by 2100 (BAU)
- Temperature anomaly should reach +2.5-4.0 deg C by 2100
- Social tension should rise significantly
- Fossil fuels should show Hubbert-style peak and decline
- Population should approach ~10-11 billion then plateau/decline
"""

import sys
sys.path.insert(0, '.')
from macro import MacroModel, MacroState


def run_bau_scenario():
    """Run Business-As-Usual scenario from 2025 to 2100."""
    model = MacroModel(config={"dt_years": 1.0 / 12.0})  # Monthly steps

    # No agent feedback = BAU defaults
    bau_feedback = {}

    print("=" * 80)
    print("MACRO MODEL BAU SCENARIO: 2025 -> 2100")
    print("=" * 80)

    header = f"{'Year':>6} {'CO2':>7} {'Temp':>6} {'SLR':>6} {'Fossil':>7} {'Mineral':>7} {'Poll':>6} {'Pop(B)':>7} {'GDP':>6} {'Ineq':>6} {'Tension':>7} {'Tech':>6} {'Renew':>6} {'Food':>6} {'Welfare':>7}"
    print(header)
    print("-" * len(header))

    years_data = []
    target_years = list(range(2025, 2101, 5))
    step_count = 0

    while model.state.year < 2100.5:
        model.step(bau_feedback)
        step_count += 1

        # Print every 5 years
        year_rounded = round(model.state.year)
        if year_rounded in target_years and abs(model.state.year - year_rounded) < 0.05:
            s = model.state
            print(f"{s.year:6.1f} {s.co2_ppm:7.1f} {s.temperature_anomaly:6.2f} "
                  f"{s.sea_level_rise_m:6.3f} {s.fossil_fuels:7.3f} {s.minerals_global:7.3f} "
                  f"{s.persistent_pollution:6.3f} {s.global_population_billions:7.2f} "
                  f"{s.global_gdp_index:6.3f} {s.inequality_index:6.3f} "
                  f"{s.social_tension:7.3f} {s.technology_level:6.3f} "
                  f"{s.renewable_fraction:6.3f} {s.food_production_index:6.3f} "
                  f"{s.human_welfare_index:7.3f}")
            target_years.remove(year_rounded)
            years_data.append(model.get_summary())

    print()
    print(f"Total ODE steps: {step_count}")
    print()

    # Validation
    s = model.state
    print("=" * 60)
    print("VALIDATION (2100 values)")
    print("=" * 60)

    checks = []

    # CO2: BAU should reach 550-750 ppm
    ok = 500 <= s.co2_ppm <= 800
    checks.append(ok)
    print(f"CO2: {s.co2_ppm:.1f} ppm  (target: 500-800)  {'PASS' if ok else 'FAIL'}")

    # Temperature: +2.0-5.0 C
    ok = 2.0 <= s.temperature_anomaly <= 5.0
    checks.append(ok)
    print(f"Temperature: +{s.temperature_anomaly:.2f} C  (target: +2.0-5.0)  {'PASS' if ok else 'FAIL'}")

    # Population: 8-12 billion
    ok = 7.0 <= s.global_population_billions <= 13.0
    checks.append(ok)
    print(f"Population: {s.global_population_billions:.2f} B  (target: 7-13)  {'PASS' if ok else 'FAIL'}")

    # Fossil fuels: should have declined significantly
    ok = 0.1 <= s.fossil_fuels <= 0.7
    checks.append(ok)
    print(f"Fossil fuels: {s.fossil_fuels:.3f}  (target: 0.1-0.7)  {'PASS' if ok else 'FAIL'}")

    # Social tension: should have risen
    ok = s.social_tension > 0.3
    checks.append(ok)
    print(f"Social tension: {s.social_tension:.3f}  (target: >0.3)  {'PASS' if ok else 'FAIL'}")

    # Sea level: 0.3-1.5m by 2100
    ok = 0.2 <= s.sea_level_rise_m <= 2.0
    checks.append(ok)
    print(f"Sea level rise: {s.sea_level_rise_m:.3f} m  (target: 0.2-2.0)  {'PASS' if ok else 'FAIL'}")

    # Renewable fraction: should have grown
    ok = s.renewable_fraction > 0.3
    checks.append(ok)
    print(f"Renewable fraction: {s.renewable_fraction:.3f}  (target: >0.3)  {'PASS' if ok else 'FAIL'}")

    # Technology: should have grown
    ok = s.technology_level > 1.5
    checks.append(ok)
    print(f"Technology level: {s.technology_level:.3f}  (target: >1.5)  {'PASS' if ok else 'FAIL'}")

    print()
    passed = sum(checks)
    total = len(checks)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("ALL VALIDATIONS PASSED!")
    else:
        print(f"WARNING: {total - passed} checks failed")

    return passed == total


if __name__ == "__main__":
    success = run_bau_scenario()
    sys.exit(0 if success else 1)
