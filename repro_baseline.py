#!/usr/bin/env python3
"""
Minimal baseline reproduction for Step 1: Reproduce baseline
Compare a two-stake sticky (Hysteresis) strategy against a standard strategy
in a small, deterministic setting to observe final BRs and RoR.
"""
import simulation_engine as engine

def main():
    # Minimal two-stake setup: NL20 and NL50
    stakes_data = [
        {"name": "NL20", "bb_size": 0.20, "bb_per_100": 2.0, "ev_bb_per_100": 2.0, "std_dev_per_100": 100.0,
         "sample_hands": 10000, "win_rate_drop": 0.0, "rake_bb_per_100": 0.0},
        {"name": "NL50", "bb_size": 0.50, "bb_per_100": 2.0, "ev_bb_per_100": 2.0, "std_dev_per_100": 100.0,
         "sample_hands": 10000, "win_rate_drop": 0.0, "rake_bb_per_100": 0.0},
    ]

    # Two strategies:
    # - Sticky Demo: Hysteresis strategy with a typical BI buffer
    # - Balanced Demo: A simple standard strategy with a couple of rules
    strategies = {
        "Sticky Demo": {"type": "hysteresis", "num_buy_ins": 40},
        "Balanced Demo": {"type": "standard",
                          "rules": [
                              {"threshold": 10000, "tables": {"NL50": "100%"}},
                              {"threshold": 0, "tables": {"NL20": "100%"}}
                          ]}
    }

    # Lightweight config to run the baseline
    config = {
        "STARTING_BANKROLL_EUR": 1000,
        "TARGET_BANKROLL": 2000,
        "RUIN_THRESHOLD": 0,
        "NUMBER_OF_SIMULATIONS": 500,    # keep small for quick baseline
        "TOTAL_HANDS_TO_SIMULATE": 10000,
        "HANDS_PER_CHECK": 1000,
        "RAKEBACK_PERCENTAGE": 0.0,
        "STOP_LOSS_BB": 0,
        "PRIOR_SAMPLE_SIZE": 1000,
        "ZERO_HANDS_INPUT_WEIGHT": 0.5,
        "SEED": 42,
        "PLOT_PERCENTILE_LIMIT": 99,
        "STAKES_DATA": stakes_data,
        "STRATEGIES_TO_RUN": strategies
    }

    # Run the analysis and print a compact summary
    results = engine.run_full_analysis(config)
    print("Baseline Reproduction: Sticky vs Standard (2-stake, small sample)")
    for name, res in results['results'].items():
        print(f"- {name}: Median BR = {res['median_final_bankroll']:.2f}, RoR = {res['risk_of_ruin']:.2f}%, "
              f"TargetProb = {res['target_prob']:.2f}%, Median Growth = {res['growth_rate']:.4f}")
    # Quick delta check if both present
    try:
        sticky = results['results']['Sticky Demo']
        balanced = results['results']['Balanced Demo']
        print("Sticky vs Balanced med BR diff:", sticky['median_final_bankroll'] - balanced['median_final_bankroll'])
    except Exception:
        pass

if __name__ == "__main__":
    main()