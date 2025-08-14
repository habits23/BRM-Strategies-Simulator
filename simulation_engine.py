import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.stats import t, gaussian_kde
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import io

# =================================================================================
#   BANKROLL MANAGEMENT STRATEGY CLASS
# =================================================================================

class BankrollManagementStrategy:
    """
    A class to represent a bankroll management strategy with defined thresholds.
    """
    def __init__(self, name, strategy_rules, stakes_data):
        """
        Initializes the strategy with a name, a list of rules, and the stakes data for validation.
        """
        self.name = name
        self.rules = sorted(strategy_rules, key=lambda x: x["threshold"], reverse=True)
        self._validate_rules(stakes_data)

    def _validate_rules(self, stakes_data):
        """
        Central validation method to run all checks on the strategy rules.
        """
        self._validate_sorted_thresholds()
        self._validate_rule_content(stakes_data)

    def _validate_sorted_thresholds(self):
        """
        Ensures that the thresholds in the rules are sorted in descending order.
        """
        for i in range(len(self.rules) - 1):
            if self.rules[i]["threshold"] < self.rules[i+1]["threshold"]:
                raise ValueError(
                    f"Strategy '{self.name}' rules are not sorted correctly. "
                    "Thresholds must be in descending order."
                )

    def _validate_rule_content(self, stakes_data):
        """
        Checks that each rule's content is valid.
        """
        valid_stakes = {stake["name"] for stake in stakes_data}
        for rule in self.rules:
            if "threshold" not in rule or "tables" not in rule:
                raise ValueError(f"Strategy '{self.name}' rule is missing 'threshold' or 'tables'.")
            for stake_name, count in rule["tables"].items():
                if stake_name not in valid_stakes:
                    raise ValueError(f"Strategy '{self.name}' references invalid stake: '{stake_name}'. Valid: {valid_stakes}")
                if isinstance(count, str):
                    sanitized_val = count.replace('%', '').replace(' ', '')
                    if "-" in sanitized_val:
                        parts = sanitized_val.split('-')
                        if len(parts) != 2: raise ValueError(f"Invalid percentage range '{count}' in '{self.name}'.")
                        try:
                            low, high = map(float, parts)
                            if not (0 <= low <= high <= 100): raise ValueError(f"Percentage range '{count}' invalid in '{self.name}'.")
                        except ValueError: raise ValueError(f"Invalid percentage range format '{count}' in '{self.name}'.")
                    else:
                        try:
                            if not (0 <= float(sanitized_val) <= 100): raise ValueError(f"Percentage '{count}' invalid in '{self.name}'.")
                        except ValueError: raise ValueError(f"Invalid fixed percentage format '{count}' in '{self.name}'.")
                elif not (isinstance(count, int) and count >= 0):
                     raise ValueError(f"Strategy '{self.name}' has invalid table count for '{stake_name}'. Must be non-negative int or percentage string.")

    def get_table_mix(self, bankroll):
        """
        Finds the correct table mix for a given bankroll.
        If the bankroll is below the lowest threshold, it defaults to the mix of the lowest threshold rule.
        """
        for rule in self.rules:
            if bankroll >= rule["threshold"]:
                return rule["tables"]
        # If no rule is met (bankroll is below the lowest threshold),
        # return the table mix of the lowest threshold rule.
        if self.rules:
            return self.rules[-1]["tables"]
        return {}

    def get_rules_as_vectors(self):
        """
        Returns the thresholds and table mixes as separate lists, sorted.
        """
        thresholds = [rule["threshold"] for rule in self.rules]
        tables = [rule["tables"] for rule in self.rules]
        return thresholds, tables

class HysteresisStrategy(BankrollManagementStrategy):
    """
    A specialized strategy class for hysteresis-based ("sticky") bankroll management.
    """
    def __init__(self, name, num_buy_ins, stakes_data):
        self.num_buy_ins = num_buy_ins
        self.stakes_data = sorted(stakes_data, key=lambda x: x["bb_size"])
        # Generate rules before calling super().__init__ so they are ready for validation
        rules = self._generate_rules()
        super().__init__(name, rules, stakes_data)

    def _generate_rules(self):
        """
        Generate rules for moving between stakes based on buy-in requirements.
        """
        stake_thresholds = []
        for i in range(len(self.stakes_data) - 1, -1, -1):
            current_stake = self.stakes_data[i]
            if isinstance(self.num_buy_ins, dict):
                # Use the specific value, or a default of 40 if a stake is somehow missing.
                buy_ins_for_stake = self.num_buy_ins.get(current_stake['name'], 40)
            else:
                # Use the global integer value.
                buy_ins_for_stake = self.num_buy_ins
            threshold = buy_ins_for_stake * 100 * current_stake["bb_size"]
            tables_config = {s["name"]: "0%" for s in self.stakes_data}
            tables_config[current_stake["name"]] = "100%"
            stake_thresholds.append({
                "threshold": threshold,
                "tables": tables_config,
                "stake_name": current_stake["name"]
            })
        return sorted(stake_thresholds, key=lambda x: x["threshold"], reverse=True)

# =================================================================================
#   SIMULATION LOGIC AND HELPER FUNCTIONS
# =================================================================================

def setup_simulation_parameters(config):
    """Initializes and returns common parameters needed for simulations."""
    rng = np.random.default_rng(config['SEED'])
    all_win_rates = {}
    all_dfs = {}
    all_session_profits_bb = {}

    for i, stake in enumerate(config['STAKES_DATA']):
        name = stake["name"]
        ev_bb_per_100 = stake.get("ev_bb_per_100", stake["bb_per_100"])
        std_dev_per_100 = stake["std_dev_per_100"]
        sample_hands = stake["sample_hands"]
        win_rate_drop = stake["win_rate_drop"]

        if i > 0:
            previous_stake_name = config['STAKES_DATA'][i-1]["name"]
            prior_win_rate = np.mean(all_win_rates[previous_stake_name]) - win_rate_drop
        else:
            prior_win_rate = ev_bb_per_100

        all_win_rates[name] = calculate_effective_win_rate(ev_bb_per_100, std_dev_per_100, sample_hands, rng, prior_win_rate, config)
        all_dfs[name] = calculate_dynamic_df(sample_hands, config)

        df = all_dfs[name]
        t_samples = t.rvs(df, size=(config['NUMBER_OF_SIMULATIONS'], config['TOTAL_SESSIONS_PER_RUN']), random_state=rng)
        t_std_dev = np.sqrt(df / (df - 2))
        normalized_t_samples = t_samples / t_std_dev
        all_session_profits_bb[name] = (normalized_t_samples * std_dev_per_100 + all_win_rates[name][:, np.newaxis])

    return all_session_profits_bb, all_win_rates, rng

def calculate_session_outcome(current_bankrolls, tables_per_stake, all_session_profits_bb, active_mask, session_index, config):
    """Calculates the profit and hands played for a single session across all simulations."""
    session_profits_eur = np.zeros_like(current_bankrolls)
    hands_per_stake_this_session = {stake["name"]: np.zeros_like(current_bankrolls, dtype=int) for stake in config['STAKES_DATA']}
    hands_per_100_blocks = config['HANDS_PER_TABLE_PER_SESSION'] / 100.0

    for stake in config['STAKES_DATA']:
        name = stake["name"]
        bb_size = stake["bb_size"]
        tables_mask = active_mask & (tables_per_stake[name] > 0)
        if np.any(tables_mask):
            profit_bb_per_100 = all_session_profits_bb[name][tables_mask, session_index]
            total_profit_bb = profit_bb_per_100 * (tables_per_stake[name][tables_mask] * hands_per_100_blocks)
            session_profits_eur[tables_mask] += total_profit_bb * bb_size

            hands_for_stake = tables_per_stake[name][tables_mask] * config['HANDS_PER_TABLE_PER_SESSION']
            hands_per_stake_this_session[name][tables_mask] = hands_for_stake

    uncapped_profits = session_profits_eur.copy()

    if config['STOP_LOSS_PERCENTAGE_PER_SESSION'] > 0.0:
        max_loss_eur = current_bankrolls * config['STOP_LOSS_PERCENTAGE_PER_SESSION']
        session_profits_eur = np.maximum(uncapped_profits, -max_loss_eur)

        stop_loss_hit_mask = (uncapped_profits < -max_loss_eur)
        if np.any(stop_loss_hit_mask):
            loss_ratio = np.divide(max_loss_eur[stop_loss_hit_mask], -uncapped_profits[stop_loss_hit_mask], where=uncapped_profits[stop_loss_hit_mask] != 0)
            for name in hands_per_stake_this_session:
                hands_per_stake_this_session[name][stop_loss_hit_mask] = (hands_per_stake_this_session[name][stop_loss_hit_mask] * loss_ratio).astype(int)

    total_rakeback_eur = np.zeros_like(current_bankrolls)
    if config['RAKEBACK_PERCENTAGE'] > 0:
        for stake in config['STAKES_DATA']:
            if "rake_bb_per_100" in stake and np.any(hands_per_stake_this_session[stake["name"]] > 0):
                rake_paid_eur = (hands_per_stake_this_session[stake["name"]] / 100.0) * stake["rake_bb_per_100"] * stake["bb_size"]
                total_rakeback_eur += rake_paid_eur * config['RAKEBACK_PERCENTAGE']

    final_session_profit = session_profits_eur + total_rakeback_eur
    return final_session_profit, hands_per_stake_this_session, total_rakeback_eur

def calculate_effective_win_rate(ev_bb_per_100, std_dev_per_100, sample_hands, rng, prior_win_rate, config):
    """Adjusts the observed EV win rate using a Bayesian-inspired 'shrinkage' method."""
    if sample_hands > 0:
        data_weight = sample_hands / (sample_hands + config['PRIOR_SAMPLE_SIZE'])
        shrunk_win_rate = (data_weight * ev_bb_per_100) + ((1 - data_weight) * prior_win_rate)
    else:
        model_extrapolation = prior_win_rate
        user_estimate = ev_bb_per_100
        shrunk_win_rate = (config['ZERO_HANDS_INPUT_WEIGHT'] * user_estimate) + ((1 - config['ZERO_HANDS_INPUT_WEIGHT']) * model_extrapolation)

    effective_sample_size_for_variance = sample_hands + config['PRIOR_SAMPLE_SIZE']
    N_blocks = max(1.0, effective_sample_size_for_variance / 100.0)
    std_error = std_dev_per_100 / np.sqrt(N_blocks)
    adjustment = rng.normal(loc=0.0, scale=std_error, size=config['NUMBER_OF_SIMULATIONS'])
    return shrunk_win_rate + adjustment

def calculate_dynamic_df(sample_hands, config):
    """Calculates degrees of freedom dynamically based on sample size."""
    df = config['MIN_DEGREES_OF_FREEDOM'] + (config['MAX_DEGREES_OF_FREEDOM'] - config['MIN_DEGREES_OF_FREEDOM']) * \
         (min(sample_hands, config['HANDS_FOR_MAX_DF']) / config['HANDS_FOR_MAX_DF'])
    return df

def calculate_binned_mode(data, ruin_threshold):
    """Calculates the mode of a continuous distribution using Kernel Density Estimation (KDE)."""
    successful_runs = data[data > ruin_threshold]
    if len(successful_runs) < 2:
        return np.median(data)

    p99 = np.percentile(successful_runs, 99)
    filtered_data = successful_runs[successful_runs <= p99]
    if len(filtered_data) < 2:
        return np.median(successful_runs)

    try:
        kde = gaussian_kde(filtered_data)
        grid = np.linspace(min(filtered_data), max(filtered_data), 1000)
        density = kde(grid)
        return grid[np.argmax(density)]
    except (np.linalg.LinAlgError, ValueError):
        return np.median(filtered_data)

def sample_percentage_range(range_str, rng):
    """Given a range string like "40-60", return a random float in that range."""
    parts = range_str.split('-')
    low, high = map(float, parts)
    return rng.uniform(low, high)

def normalize_percentages(percentages):
    """Normalizes a dictionary of percentages so they sum to 1."""
    total = sum(percentages.values())
    if total == 0:
        return {k: 1.0 / len(percentages) for k in percentages}
    return {k: v / total for k, v in percentages.items()}

def round_table_counts(tables_float, total_tables):
    """Rounds float table counts to integers, ensuring the sum equals total_tables."""
    floors = {k: int(v) for k, v in tables_float.items()}
    remainder = total_tables - sum(floors.values())
    remainders = {k: v - floors[k] for k, v in tables_float.items()}
    sorted_stakes = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    counts = floors.copy()
    for i in range(remainder):
        stake = sorted_stakes[i % len(sorted_stakes)][0]
        counts[stake] += 1
    return counts

def resolve_table_mix(rule, total_tables, rng):
    """Resolves a strategy rule into a concrete dictionary of integer table counts."""
    percentages = {}
    fixed_ratios = {}
    for stake, val in rule.items():
        if isinstance(val, str):
            sanitized_val = val.replace('%', '').replace(' ', '')
            if "-" in sanitized_val:
                percentages[stake] = sample_percentage_range(sanitized_val, rng)
            else:
                percentages[stake] = float(sanitized_val)
        elif isinstance(val, int) and val > 0:
            fixed_ratios[stake] = val

    if fixed_ratios:
        total_ratio = sum(fixed_ratios.values())
        if total_ratio > 0:
            for stake, ratio_val in fixed_ratios.items():
                percentages[stake] = (ratio_val / total_ratio) * 100.0

    if percentages:
        normalized = normalize_percentages(percentages)
        tables_float = {k: v * total_tables for k, v in normalized.items()}
        return round_table_counts(tables_float, total_tables)
    return {}

def get_initial_table_mix_string(strategy, config):
    """Helper function to describe the starting table mix as a string."""
    rule = strategy.get_table_mix(config['STARTING_BANKROLL_EUR'])
    if not rule:
        return "No Play"
    mix_parts = [f"{stake}: {value}" for stake, value in sorted(rule.items())]
    return ", ".join(mix_parts) if mix_parts else "No Play"

def _calculate_percentile_win_rates(final_bankrolls, all_win_rates, hands_per_stake_histories, rakeback_histories, config, bb_size_map):
    """Calculates assigned and realized win rates for simulations closest to key percentiles."""
    percentile_win_rates = {}
    percentiles_to_find = {'2.5th': 2.5, '5th': 5, '25th': 25, 'Median': 50, '75th': 75, '95th': 95, '97.5th': 97.5}
    final_rakeback = rakeback_histories[:, -1]

    for name, p_val in sorted(percentiles_to_find.items(), key=lambda item: item[1]):
        percentile_bankroll = np.percentile(final_bankrolls, p_val)
        closest_sim_index = np.argmin(np.abs(final_bankrolls - percentile_bankroll))
        
        stake_wrs = {'p_val': p_val}
        total_hands_for_sim, assigned_weighted_wr_sum, weighted_bb_size_sum = 0, 0, 0

        for stake_name, wr_array in all_win_rates.items():
            exact_wr = wr_array[closest_sim_index]
            stake_wrs[stake_name] = f"{exact_wr:.2f}"
            hands_at_stake = hands_per_stake_histories[stake_name][closest_sim_index, -1]
            total_hands_for_sim += hands_at_stake
            assigned_weighted_wr_sum += exact_wr * hands_at_stake
            weighted_bb_size_sum += hands_at_stake * bb_size_map[stake_name]

        if total_hands_for_sim > 0:
            avg_bb_size = weighted_bb_size_sum / total_hands_for_sim
            total_profit_eur = final_bankrolls[closest_sim_index] - config['STARTING_BANKROLL_EUR']
            profit_from_play_eur = total_profit_eur - final_rakeback[closest_sim_index]
            stake_wrs['Assigned WR'] = f"{assigned_weighted_wr_sum / total_hands_for_sim:.2f}"
            stake_wrs['Realized WR (Play)'] = f"{(profit_from_play_eur / avg_bb_size) / (total_hands_for_sim / 100):.2f}"
            stake_wrs['Rakeback (bb/100)'] = f"{(final_rakeback[closest_sim_index] / avg_bb_size) / (total_hands_for_sim / 100):.2f}"
        
        percentile_win_rates[f"{name} Percentile"] = stake_wrs
    return percentile_win_rates

def analyze_strategy_results(strategy_name, strategy_obj, bankroll_histories, hands_per_stake_histories, rakeback_histories, all_win_rates, rng, peak_stake_levels, demotion_flags, stake_level_map, stake_name_map, max_drawdowns, config):
    """Takes the raw simulation output and calculates all the necessary metrics and analytics."""
    bb_size_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
    total_hands_histories = np.sum(list(hands_per_stake_histories.values()), axis=0)
    final_bankrolls = bankroll_histories[:, -1]

    # --- Calculate Weighted Assigned WR for each simulation ---
    # This is crucial for understanding the distribution of "luck" (assigned win rates)
    final_hands_per_stake = {name: history[:, -1] for name, history in hands_per_stake_histories.items()}
    total_hands_per_sim = np.sum(list(final_hands_per_stake.values()), axis=0)

    weighted_assigned_wr_sum = np.zeros(config['NUMBER_OF_SIMULATIONS'])
    for stake_name, wr_array in all_win_rates.items():
        weighted_assigned_wr_sum += wr_array * final_hands_per_stake[stake_name]

    # Avoid division by zero for sims with no hands played
    avg_assigned_wr_per_sim = np.divide(
        weighted_assigned_wr_sum,
        total_hands_per_sim,
        out=np.zeros_like(weighted_assigned_wr_sum),
        where=total_hands_per_sim != 0
    )

    # Find the Assigned WR for the run that resulted in the median final bankroll
    median_final_bankroll_val = np.percentile(final_bankrolls, 50)
    median_sim_index = np.argmin(np.abs(final_bankrolls - median_final_bankroll_val))
    median_run_assigned_wr = avg_assigned_wr_per_sim[median_sim_index]

    average_assigned_win_rates = {name: np.mean(wr_array) for name, wr_array in all_win_rates.items()}

    total_hands_per_stake = {name: np.sum(history[:, -1]) for name, history in hands_per_stake_histories.items()}
    grand_total_hands = sum(total_hands_per_stake.values())
    hands_distribution_pct = {name: (total / grand_total_hands) * 100 for name, total in total_hands_per_stake.items()} if grand_total_hands > 0 else {}

    risk_of_demotion = {}
    for level, stake_name in stake_name_map.items():
        if level > 0:
            sims_that_reached_peak = np.sum(peak_stake_levels >= level)
            if sims_that_reached_peak > 0:
                sims_demoted_from_peak = np.sum(demotion_flags[level])
                demotion_prob = (sims_demoted_from_peak / sims_that_reached_peak) * 100
                risk_of_demotion[stake_name] = {'prob': demotion_prob, 'reached_count': sims_that_reached_peak}

    final_stake_counts = defaultdict(int)
    final_highest_stake_counts = defaultdict(int)
    stake_order_map = {stake['name']: i for i, stake in enumerate(sorted(config['STAKES_DATA'], key=lambda s: s['bb_size']))}

    for br in final_bankrolls:
        table_mix = strategy_obj.get_table_mix(br)
        # This part is for the detailed PDF report, so it's still needed.
        mix_str = ", ".join(f"{s}: {v}" for s, v in sorted(table_mix.items())) if table_mix else "No Play"
        final_stake_counts[mix_str] += 1

        # This is the more direct calculation for the UI summary.
        if not table_mix:
            final_highest_stake_counts["No Play"] += 1
        else:
            highest_stake = max(table_mix.keys(), key=lambda s: stake_order_map.get(s, -1))
            final_highest_stake_counts[highest_stake] += 1

    final_stake_distribution = {mix_str: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for mix_str, count in final_stake_counts.items()}
    final_highest_stake_distribution = {stake: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for stake, count in final_highest_stake_counts.items()}

    percentile_win_rates = _calculate_percentile_win_rates(final_bankrolls, all_win_rates, hands_per_stake_histories, rakeback_histories, config, bb_size_map)

    median_max_drawdown = np.median(max_drawdowns)
    p95_max_drawdown = np.percentile(max_drawdowns, 95)

    # Calculate median rakeback
    final_rakeback = rakeback_histories[:, -1]
    median_rakeback_eur = np.median(final_rakeback)

    final_bankroll_mode = calculate_binned_mode(final_bankrolls, config['RUIN_THRESHOLD'])
    target_achieved_count = np.sum(np.any(bankroll_histories >= config['TARGET_BANKROLL'], axis=1))
    busted_runs = np.sum(np.any(bankroll_histories <= config['RUIN_THRESHOLD'], axis=1))
    risk_of_ruin_percent = (busted_runs / config['NUMBER_OF_SIMULATIONS']) * 100
    percentiles = {p: np.percentile(final_bankrolls, p) for p in [2.5, 5, 25, 50, 75, 95, 97.5]}
    median_growth_rate = (percentiles[50] - config['STARTING_BANKROLL_EUR']) / config['STARTING_BANKROLL_EUR'] if config['STARTING_BANKROLL_EUR'] > 0 else 0.0

    return {
        'final_bankrolls': final_bankrolls, 'median_final_bankroll': percentiles[50],
        'final_bankroll_mode': final_bankroll_mode, 'growth_rate': median_growth_rate,
        'risk_of_ruin': risk_of_ruin_percent, 'p5': percentiles[5], 'p2_5': percentiles[2.5],
        'bankroll_histories': bankroll_histories, 'hands_histories': total_hands_histories,
        'median_history': np.median(bankroll_histories, axis=0),
        'hands_history': np.mean(total_hands_histories, axis=0),
        'target_prob': (target_achieved_count / config['NUMBER_OF_SIMULATIONS']) * 100,
        'above_threshold_hit_counts': {rule["threshold"]: np.sum(np.any(bankroll_histories >= rule["threshold"], axis=1))
                                      for rule in strategy_obj.rules if rule["threshold"] > config['STARTING_BANKROLL_EUR']},
        'below_threshold_drop_counts': {rule["threshold"]: np.sum(np.any(bankroll_histories <= rule["threshold"], axis=1))
                                       for rule in strategy_obj.rules if rule["threshold"] < config['STARTING_BANKROLL_EUR']},
        'percentile_win_rates': percentile_win_rates, 'risk_of_demotion': risk_of_demotion,
        'hands_distribution_pct': hands_distribution_pct,
        'final_stake_distribution': final_stake_distribution,
        'final_highest_stake_distribution': final_highest_stake_distribution,
        'median_max_drawdown': median_max_drawdown,
        'p95_max_drawdown': p95_max_drawdown,
        'median_rakeback_eur': median_rakeback_eur,
        'average_assigned_win_rates': average_assigned_win_rates,
        'avg_assigned_wr_per_sim': avg_assigned_wr_per_sim,
        'median_run_assigned_wr': median_run_assigned_wr,
    }
def run_multiple_simulations_vectorized(strategy, all_session_profits_bb, rng, stake_level_map, config):
    """
    Runs all simulations at once using vectorized NumPy operations for speed.
    This version dynamically resolves table mixes for each session.
    """
    bankroll_history = np.full((config['NUMBER_OF_SIMULATIONS'], config['TOTAL_SESSIONS_PER_RUN'] + 1), config['STARTING_BANKROLL_EUR'], dtype=float)
    hands_per_stake_histories = {stake['name']: np.zeros((config['NUMBER_OF_SIMULATIONS'], config['TOTAL_SESSIONS_PER_RUN'] + 1), dtype=int) for stake in config['STAKES_DATA']}
    rakeback_histories = np.zeros((config['NUMBER_OF_SIMULATIONS'], config['TOTAL_SESSIONS_PER_RUN'] + 1), dtype=float)

    # --- Maximum Drawdown Initialization ---
    peak_bankrolls_so_far = np.full(config['NUMBER_OF_SIMULATIONS'], config['STARTING_BANKROLL_EUR'], dtype=float)
    max_drawdowns_so_far = np.zeros(config['NUMBER_OF_SIMULATIONS'], dtype=float)


    # --- Demotion Tracking Initialization ---
    initial_rule = strategy.get_table_mix(config['STARTING_BANKROLL_EUR'])
    initial_stakes_with_tables = [stake for stake, count in initial_rule.items() if (isinstance(count, int) and count > 0) or (isinstance(count, str) and float(count.replace('%','').split('-')[0]) > 0)]
    initial_level = max([stake_level_map[s] for s in initial_stakes_with_tables]) if initial_stakes_with_tables else -1
    peak_stake_levels = np.full(config['NUMBER_OF_SIMULATIONS'], initial_level, dtype=int)
    demotion_flags = {level: np.zeros(config['NUMBER_OF_SIMULATIONS'], dtype=bool) for level in stake_level_map.values()}

    thresholds, rules = strategy.get_rules_as_vectors()

    for i in range(config['TOTAL_SESSIONS_PER_RUN']):
        current_bankrolls = bankroll_history[:, i]
        
        # Store previous peak levels to detect demotions
        previous_peak_levels = peak_stake_levels.copy()

        # Determine the table mix for each simulation based on its current bankroll
        tables_per_stake = {stake["name"]: np.zeros_like(current_bankrolls, dtype=int) for stake in config['STAKES_DATA']}
        session_total_tables = rng.integers(config['MIN_TABLES_PER_SESSION'], config['MAX_TABLES_PER_SESSION'] + 1, size=config['NUMBER_OF_SIMULATIONS'])
        
        remaining_mask = np.ones_like(current_bankrolls, dtype=bool)
        for threshold, rule in zip(thresholds, rules):
            current_mask = (current_bankrolls >= threshold) & remaining_mask
            if not np.any(current_mask):
                continue
            
            indices = np.where(current_mask)[0]
            for sim_idx in indices:
                resolved_mix = resolve_table_mix(rule, session_total_tables[sim_idx], rng)
                for stake_name, count in resolved_mix.items():
                    tables_per_stake[stake_name][sim_idx] = count
            remaining_mask[current_mask] = False

        # Handle simulations with bankroll below the lowest threshold by applying the lowest rule
        if np.any(remaining_mask) and rules:
            lowest_rule = rules[-1] # The last rule is the lowest threshold
            indices = np.where(remaining_mask)[0]
            for sim_idx in indices:
                resolved_mix = resolve_table_mix(lowest_rule, session_total_tables[sim_idx], rng)
                for stake_name, count in resolved_mix.items():
                    tables_per_stake[stake_name][sim_idx] = count

        # --- Demotion Tracking Logic ---
        current_levels = np.full(config['NUMBER_OF_SIMULATIONS'], -1, dtype=int)
        for stake_name, level in stake_level_map.items():
            has_tables_mask = tables_per_stake[stake_name] > 0
            current_levels[has_tables_mask] = np.maximum(current_levels[has_tables_mask], level)

        demotion_from_peak_mask = current_levels < previous_peak_levels
        for level in stake_level_map.values():
            if level > 0:
                demoted_this_session_mask = (previous_peak_levels == level) & demotion_from_peak_mask
                demotion_flags[level][demoted_this_session_mask] = True

        # Update peak levels for the next session
        peak_stake_levels = np.maximum(previous_peak_levels, current_levels)

        total_tables = sum(tables_per_stake.values())
        active_mask = (current_bankrolls >= config['RUIN_THRESHOLD']) & (total_tables > 0)
        if not np.any(active_mask):
            bankroll_history[:, i+1:] = bankroll_history[:, i][:, np.newaxis]
            for stake_name in hands_per_stake_histories:
                hands_per_stake_histories[stake_name][:, i+1:] = hands_per_stake_histories[stake_name][:, i][:, np.newaxis]
            break

        session_profits_eur, hands_per_stake_this_session, session_rakeback_eur = calculate_session_outcome(
            current_bankrolls, tables_per_stake, all_session_profits_bb, active_mask, i, config
        )

        new_bankrolls = current_bankrolls + session_profits_eur
        bankroll_history[:, i+1] = np.where(active_mask, new_bankrolls, current_bankrolls)

        # --- Maximum Drawdown Calculation ---
        peak_bankrolls_so_far = np.maximum(peak_bankrolls_so_far, bankroll_history[:, i+1])
        current_drawdowns = peak_bankrolls_so_far - bankroll_history[:, i+1]
        max_drawdowns_so_far = np.maximum(max_drawdowns_so_far, current_drawdowns)

        rakeback_histories[:, i+1] = rakeback_histories[:, i] + np.where(active_mask, session_rakeback_eur, 0)
        for stake_name, hands_array in hands_per_stake_this_session.items():
            hands_per_stake_histories[stake_name][:, i+1] = hands_per_stake_histories[stake_name][:, i] + np.where(active_mask, hands_array, 0)

    return bankroll_history, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns_so_far

def run_sticky_simulation_vectorized(strategy, all_session_profits_bb, rng, stake_level_map, config):
    """
    Runs a simulation with a specific 'sticky' bankroll management strategy.
    This version correctly handles multiple stakes by implementing a proper state machine.
    """
    bankroll_history = np.full((config['NUMBER_OF_SIMULATIONS'], config['TOTAL_SESSIONS_PER_RUN'] + 1), config['STARTING_BANKROLL_EUR'], dtype=float)
    hands_per_stake_histories = {stake['name']: np.zeros((config['NUMBER_OF_SIMULATIONS'], config['TOTAL_SESSIONS_PER_RUN'] + 1), dtype=int) for stake in config['STAKES_DATA']}
    rakeback_histories = np.zeros((config['NUMBER_OF_SIMULATIONS'], config['TOTAL_SESSIONS_PER_RUN'] + 1), dtype=float)

    # --- Maximum Drawdown Initialization ---
    peak_bankrolls_so_far = np.full(config['NUMBER_OF_SIMULATIONS'], config['STARTING_BANKROLL_EUR'], dtype=float)
    max_drawdowns_so_far = np.zeros(config['NUMBER_OF_SIMULATIONS'], dtype=float)


    # --- Demotion Tracking Initialization ---
    stake_rules = sorted(strategy.rules, key=lambda r: r['threshold'])
    rule_index_to_level = {i: stake_level_map[rule['stake_name']] for i, rule in enumerate(stake_rules)}

    current_stake_indices = np.zeros(config['NUMBER_OF_SIMULATIONS'], dtype=int)
    for i in range(len(stake_rules) - 1, -1, -1):
        threshold = stake_rules[i]['threshold']
        current_stake_indices[config['STARTING_BANKROLL_EUR'] >= threshold] = i

    initial_levels = np.vectorize(rule_index_to_level.get)(current_stake_indices)
    peak_stake_levels = initial_levels.copy()
    demotion_flags = {level: np.zeros(config['NUMBER_OF_SIMULATIONS'], dtype=bool) for level in stake_level_map.values()}

    for session_idx in range(config['TOTAL_SESSIONS_PER_RUN']):
        current_bankrolls = bankroll_history[:, session_idx]

        previous_peak_levels = peak_stake_levels.copy()


        previous_stake_indices = current_stake_indices.copy()

        for i in range(len(stake_rules) - 1):
            move_up_threshold = stake_rules[i+1]['threshold']
            can_move_up_mask = (current_stake_indices == i) & (current_bankrolls >= move_up_threshold)
            current_stake_indices[can_move_up_mask] = i + 1

        for i in range(len(stake_rules) - 1, 0, -1):
            # Hysteresis move-down: A player at stake 'i' only moves down to 'i-1'
            # if their bankroll drops below the entry threshold for stake 'i-1'.
            # This creates a "sticky" buffer zone.
            move_down_threshold = stake_rules[i-1]['threshold']
            must_move_down_mask = (current_stake_indices == i) & (current_bankrolls < move_down_threshold)
            current_stake_indices[must_move_down_mask] = i - 1

        # --- Demotion Tracking Logic ---
        current_levels = np.vectorize(rule_index_to_level.get)(current_stake_indices)

        demotion_from_peak_mask = current_levels < previous_peak_levels
        for level in stake_level_map.values():
            demoted_this_session_mask = (previous_peak_levels == level) & demotion_from_peak_mask
            demotion_flags[level][demoted_this_session_mask] = True

        # Update peak levels for the next session
        peak_stake_levels = np.maximum(previous_peak_levels, current_levels)

        tables_per_stake = {stake["name"]: np.zeros(config['NUMBER_OF_SIMULATIONS'], dtype=int) for stake in config['STAKES_DATA']}
        session_total_tables = rng.integers(config['MIN_TABLES_PER_SESSION'], config['MAX_TABLES_PER_SESSION'] + 1, size=config['NUMBER_OF_SIMULATIONS'])

        for i in range(len(stake_rules)):
            at_this_stake_mask = (current_stake_indices == i)
            if not np.any(at_this_stake_mask):
                continue
            # Since the sticky strategy is always 100% of one stake, we can vectorize this.
            stake_name_for_rule = stake_rules[i]['stake_name']
            tables_per_stake[stake_name_for_rule][at_this_stake_mask] = session_total_tables[at_this_stake_mask]

        total_tables = sum(tables_per_stake.values())
        active_mask = (current_bankrolls >= config['RUIN_THRESHOLD']) & (total_tables > 0)

        session_profits_eur, hands_per_stake_this_session, session_rakeback_eur = calculate_session_outcome(
            current_bankrolls, tables_per_stake, all_session_profits_bb, active_mask, session_idx, config
        )

        new_bankrolls = current_bankrolls + session_profits_eur
        bankroll_history[:, session_idx+1] = np.where(active_mask, new_bankrolls, current_bankrolls)

        # --- Maximum Drawdown Calculation ---
        peak_bankrolls_so_far = np.maximum(peak_bankrolls_so_far, bankroll_history[:, session_idx+1])
        current_drawdowns = peak_bankrolls_so_far - bankroll_history[:, session_idx+1]
        max_drawdowns_so_far = np.maximum(max_drawdowns_so_far, current_drawdowns)

        rakeback_histories[:, session_idx+1] = rakeback_histories[:, session_idx] + np.where(active_mask, session_rakeback_eur, 0)
        for stake_name, hands_array in hands_per_stake_this_session.items():
            hands_per_stake_histories[stake_name][:, session_idx+1] = hands_per_stake_histories[stake_name][:, session_idx] + np.where(active_mask, hands_array, 0)

    return bankroll_history, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns_so_far
# =================================================================================
#   PLOTTING AND REPORTING FUNCTIONS
# =================================================================================

def plot_strategy_progression(bankroll_histories, hands_histories, strategy_name, config, pdf=None):
    """Plots the median progression with a shaded area for the 25th-75th percentile range."""
    fig, ax = plt.subplots(figsize=(8, 5)) # Further reduced size for a more compact app layout
    median_history = np.median(bankroll_histories, axis=0)
    y_upper_limit = np.percentile(bankroll_histories, 95) * 1.1 # Add 10% for padding
    lower_percentile = np.percentile(bankroll_histories, 25, axis=0)
    upper_percentile = np.percentile(bankroll_histories, 75, axis=0)
    median_hands = np.median(hands_histories, axis=0)

    for i in range(min(50, len(bankroll_histories))):
        ax.plot(hands_histories[i], bankroll_histories[i], color='gray', alpha=0.1)

    ax.fill_between(median_hands, lower_percentile, upper_percentile,
                     color='lightblue', alpha=0.4, label='25th-75th Percentile Range')
    ax.plot(median_hands, median_history, 'b-', linewidth=3, label='Median Progression')
    ax.axhline(config['STARTING_BANKROLL_EUR'], color='black', linewidth=2, label='Starting Bankroll')
    ax.axhline(config['TARGET_BANKROLL'], color='gold', linestyle='-.', label=f"Target: €{config['TARGET_BANKROLL']}")
    ax.axhline(config['RUIN_THRESHOLD'], color='red', linestyle='--', linewidth=2, label=f"Ruin Threshold: €{config['RUIN_THRESHOLD']}")
    ax.set_title(f'Bankroll Progression: {strategy_name}', fontsize=14)
    ax.set_xlabel('Total Hands Played', fontsize=12)
    ax.set_ylabel('Bankroll (EUR)', fontsize=12)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0, top=y_upper_limit)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_median_progression_comparison(all_results, config, pdf=None):
    """Compares the median bankroll progression for all strategies on a single plot."""
    fig, ax = plt.subplots(figsize=(8, 5)) # Made consistent with other compact plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for i, (strategy_name, result) in enumerate(all_results.items()):
        ax.plot(result['hands_history'], result['median_history'], label=strategy_name, linewidth=2.5, color=colors[i])

    ax.axhline(config['STARTING_BANKROLL_EUR'], color='gray', linestyle='--', label='Starting Bankroll')
    ax.axhline(config['TARGET_BANKROLL'], color='gold', linestyle='-.', label=f"Target: €{config['TARGET_BANKROLL']}")
    ax.axhline(config['RUIN_THRESHOLD'], color='red', linestyle='--', linewidth=2, label=f"Ruin Threshold: €{config['RUIN_THRESHOLD']}")

    ax.set_title('Median Bankroll Progression Comparison Across All Strategies', fontsize=16)
    ax.set_xlabel('Total Hands Played', fontsize=12)
    ax.set_ylabel('Bankroll (EUR)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_final_bankroll_distribution(final_bankrolls, result, strategy_name, config, pdf=None):
    """Creates a histogram of final bankrolls with key metrics highlighted."""
    # Filter out bankrolls that are too high to improve the visibility of the main data cluster.
    max_x_limit = np.percentile(final_bankrolls, 99.0)
    filtered_bankrolls = final_bankrolls[final_bankrolls <= max_x_limit]

    median_val = result['median_final_bankroll']
    # We only need the 5th and 95th percentiles for this cleaner plot.
    percentiles = {p: np.percentile(final_bankrolls, p) for p in [5, 95]}

    fig, ax = plt.subplots(figsize=(8, 5)) # Further reduced size for a more compact app layout
    ax.hist(filtered_bankrolls, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

    # Key metrics
    ax.axvline(median_val, color='red', linestyle='dashed', linewidth=2, label=f'Median: €{median_val:,.2f}')
    ax.axvline(percentiles[5], color='darkred', linestyle=':', linewidth=2, label=f'5th Percentile: €{percentiles[5]:,.2f}')
    ax.axvline(percentiles[95], color='green', linestyle=':', linewidth=2, label=f'95th Percentile: €{percentiles[95]:,.2f}')

    # Contextual lines
    ax.axvline(config["STARTING_BANKROLL_EUR"], color='black', linewidth=2, label=f'Starting: €{config["STARTING_BANKROLL_EUR"]:,.0f}')
    ax.axvline(config["TARGET_BANKROLL"], color='gold', linestyle='-.', linewidth=2, label=f'Target: €{config["TARGET_BANKROLL"]:,.0f}')
    ax.axvline(config["RUIN_THRESHOLD"], color='red', linestyle='--', linewidth=2, label=f'Ruin Threshold: €{config["RUIN_THRESHOLD"]:,.0f}')
    ax.set_title(f'Final Bankroll Distribution for {strategy_name}')
    ax.set_xlabel('Final Bankroll (EUR)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_assigned_wr_distribution(avg_assigned_wr_per_sim, median_run_assigned_wr, average_input_wr, strategy_name, pdf=None):
    """
    Plots the distribution of the weighted average 'Assigned WR' for all simulations.
    This visualizes the 'luck' distribution and highlights where the median-outcome run falls.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filter out extreme outliers for better visualization
    p1 = np.percentile(avg_assigned_wr_per_sim, 1)
    p99 = np.percentile(avg_assigned_wr_per_sim, 99)
    filtered_data = avg_assigned_wr_per_sim[(avg_assigned_wr_per_sim >= p1) & (avg_assigned_wr_per_sim <= p99)]

    ax.hist(filtered_data, bins=50, color='c', edgecolor='black', alpha=0.6, label='Distribution of All Runs\' Luck')

    ax.axvline(average_input_wr, color='blue', linestyle='--', linewidth=2,
               label=f'Your Avg. Input WR: {average_input_wr:.2f}')

    ax.axvline(median_run_assigned_wr, color='red', linestyle='-', linewidth=2.5,
               label=f'Luck of Median-Outcome Run: {median_run_assigned_wr:.2f}')

    ax.set_title(f'Distribution of Assigned Luck (WR) for {strategy_name}', fontsize=14)
    ax.set_xlabel('Weighted Average Assigned Win Rate (bb/100)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Simulations)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Ensure vertical lines are always visible ---
    # Get the current limits and expand them if the lines are outside the visible range
    xmin, xmax = ax.get_xlim()
    new_xmin = min(xmin, median_run_assigned_wr, average_input_wr)
    new_xmax = max(xmax, median_run_assigned_wr, average_input_wr)
    padding = (new_xmax - new_xmin) * 0.05 # Add 5% padding
    ax.set_xlim(new_xmin - padding, new_xmax + padding)


    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def create_title_page(pdf, timestamp):
    """Creates a title page for the PDF report."""
    fig = plt.figure(figsize=(11, 8.5))
    title = "Poker Bankroll Management Simulation Report"
    subtitle = f"Generated on: {timestamp}"
    fig.text(0.5, 0.6, title, ha='center', va='center', size=20, weight='bold')
    fig.text(0.5, 0.5, subtitle, ha='center', va='center', size=14)
    fig.text(0.5, 0.1, "Simulator: Final BR Simulator v1.5", ha='center', va='center', size=10, color='gray')
    pdf.savefig(fig)
    plt.close(fig)

def get_strategy_report_lines(strategy_name, result, strategy_obj, config):
    """Gathers all the text lines for a strategy's detailed report."""
    report_lines = [
        f"--- Detailed Report for: {strategy_name} ---",
        f"Initial Table Mix: {get_initial_table_mix_string(strategy_obj, config)}",
        ""
    ]
    # Helper functions for different sections of the report
    def _write_threshold_analysis(res):
        lines = ["--- Strategy Threshold Analysis ---"]
        if 'above_threshold_hit_counts' in res and res['above_threshold_hit_counts']:
            lines.append("Probability of hitting upper thresholds:")
            for threshold, count in sorted(res['above_threshold_hit_counts'].items(), reverse=True):
                lines.append(f"   - €{threshold:,.0f}: {(count / config['NUMBER_OF_SIMULATIONS']) * 100:.2f}%")
        if 'below_threshold_drop_counts' in res and res['below_threshold_drop_counts']:
            lines.append("Probability of dropping to lower thresholds:")
            for threshold, count in sorted(res['below_threshold_drop_counts'].items(), reverse=True):
                lines.append(f"   - €{threshold:,.0f}: {(count / config['NUMBER_OF_SIMULATIONS']) * 100:.2f}%")
        return lines

    def _write_hands_distribution(res):
        lines = []
        if 'hands_distribution_pct' in res and res['hands_distribution_pct']:
            lines.extend(["", "--- Hands Played Distribution ---", "Percentage of total hands played at each stake (averaged over all simulations):"])
            stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
            sorted_stakes = sorted(res['hands_distribution_pct'].items(), key=lambda item: stake_order_map.get(item[0], float('inf')))
            for stake_name, pct in sorted_stakes:
                if pct > 0.01:
                    lines.append(f"   - {stake_name}: {pct:.2f}%")
        return lines

    def _write_demotion_analysis(res):
        lines = []
        if 'risk_of_demotion' in res and res['risk_of_demotion']:
            lines.extend(["", "--- Risk of Demotion Analysis ---"])
            lines.append("Demotion Risk: Percentage of simulations that dropped to a lower stake after reaching a peak.")
            stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
            sorted_demotions = sorted(res['risk_of_demotion'].items(), key=lambda item: stake_order_map.get(item[0], float('inf')), reverse=True)
            for stake_name, data in sorted_demotions:
                lines.append(f"   - From {stake_name}: {data['prob']:.2f}% (based on {data['reached_count']} simulations reaching this peak)")
        return lines

    def _write_drawdown_analysis(res):
        lines = []
        if 'median_max_drawdown' in res:
            lines.extend(["", "--- Maximum Drawdown (MDD) Analysis ---"])
            lines.append("MDD is the largest single peak-to-trough drop in bankroll during a simulation.")
            median_mdd = res['median_max_drawdown']
            p95_mdd = res['p95_max_drawdown']
            lines.append(f"   - Median MDD: €{median_mdd:.2f}")
            lines.append(f"   - 95th Percentile MDD: €{p95_mdd:.2f} (5% of runs had a worse drawdown)")
        return lines

    def _write_final_stake_distribution(res):
        lines = []
        if 'final_stake_distribution' in res and res['final_stake_distribution']:
            lines.extend(["", "--- Final Stake Distribution ---", "Percentage of simulations ending at each stake/table mix:"])
            sorted_dist = sorted(res['final_stake_distribution'].items(), key=lambda item: item[1], reverse=True)
            for mix_str, pct in sorted_dist:
                if pct > 0.01:
                    lines.append(f"   - {mix_str}: {pct:.2f}%")
        return lines

    def _write_final_highest_stake_distribution(res):
        lines = []
        if 'final_highest_stake_distribution' in res and res['final_highest_stake_distribution']:
            lines.extend(["", "--- Final Highest Stake Played ---", "Percentage of simulations ending with this as their highest active stake:"])
            stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
            sorted_dist = sorted(res['final_highest_stake_distribution'].items(), key=lambda item: stake_order_map.get(item[0], -1), reverse=True)
            for stake, pct in sorted_dist:
                if pct > 0.01:
                    lines.append(f"   - {stake}: {pct:.2f}%")
        return lines

    def _write_final_bankroll_metrics(res):
        p2_5 = res.get('p2_5', 0.0)
        p5 = res.get('p5', 0.0)
        median = res.get('median_final_bankroll', 0.0)
        p95 = np.percentile(res.get('final_bankrolls', [0]), 95)
        p97_5 = np.percentile(res.get('final_bankrolls', [0]), 97.5)
        return [
            "", "--- Final Bankroll Metrics ---",
            f"Risk of Ruin: {res.get('risk_of_ruin', 0.0):.2f}% (Percentage of runs that dropped to or below €{config['RUIN_THRESHOLD']})",
            f"Probability of reaching €{config['TARGET_BANKROLL']}: {res.get('target_prob', 0.0):.2f}%",
            f"2.5th Percentile: €{p2_5:.2f} (97.5% of runs finished above this value)",
            f"5th Percentile:   €{p5:.2f} (95% of runs finished above this value)",
            f"50th Percentile:  €{median:.2f} (Median)",
            f"95th Percentile:  €{p95:.2f} (5% of runs finished above this value)",
            f"97.5th Percentile:€{p97_5:.2f} (2.5% of runs finished above this value)"
        ]

    def _write_win_rate_analysis(res):
        lines = ["", "--- Effective Win Rate Analysis (bb/100) ---", "Win rates for simulations ending at specific percentiles:"]
        for percentile_name, wr_data in sorted(res['percentile_win_rates'].items(), key=lambda item: item[1]['p_val']):
            play_wr = wr_data.get('Realized WR (Play)', 'N/A')
            rb_wr = wr_data.get('Rakeback (bb/100)', 'N/A')
            assigned_wr = wr_data.get('Assigned WR', 'N/A')
            primary_wr_str = f"Realized (Play): {play_wr}, Rakeback: {rb_wr}, Assigned: {assigned_wr}"
            stake_wr_string = ", ".join([f"{stake}: {wr}" for stake, wr in wr_data.items() if stake not in ['Realized WR (Play)', 'Rakeback (bb/100)', 'Assigned WR', 'p_val']])
            lines.extend([f"   - {percentile_name}:", f"     {primary_wr_str}", f"     (Assigned Stake Breakdown: {stake_wr_string})" if stake_wr_string else ""])
        return lines

    report_lines.extend(_write_threshold_analysis(result))
    report_lines.extend(_write_hands_distribution(result))
    report_lines.extend(_write_demotion_analysis(result))
    report_lines.extend(_write_drawdown_analysis(result))
    report_lines.extend(_write_final_bankroll_metrics(result))
    report_lines.extend(_write_final_stake_distribution(result))
    report_lines.extend(_write_final_highest_stake_distribution(result))
    if 'percentile_win_rates' in result and result['percentile_win_rates']:
        report_lines.extend(_write_win_rate_analysis(result))
    return report_lines

def write_strategy_report_to_pdf(pdf, report_lines, page_number_info=None):
    """Writes the detailed text report for a single strategy to a PDF page, handling pagination."""
    lines_per_page = 45
    pages_of_lines = [report_lines[i:i + lines_per_page] for i in range(0, len(report_lines), lines_per_page)]

    for i, page_lines in enumerate(pages_of_lines):
        report_text = "\n".join(page_lines)
        fig = plt.figure(figsize=(11, 8.5))
        if page_number_info:
            fig.text(0.95, 0.05, f"Page {page_number_info['current'] + i}", transform=fig.transFigure, size=8, va='bottom', ha='right', color='gray')
        fig.text(0.05, 0.95, report_text, transform=fig.transFigure, size=10, va='top', ha='left', fontfamily='monospace')
        pdf.savefig(fig)
        plt.close(fig)

def save_summary_table_to_pdf(pdf, all_results, strategy_page_map, config):
    """Creates a table of the main summary results and saves it to a PDF page."""
    header = ['Strategy', 'Page', 'Median Final BR', 'Mode Final BR', 'Median Growth', 'Median Rakeback', 'RoR (%)', 'Target Prob (%)', '5th %ile', '2.5th %ile']
    cell_text = []
    for strategy_name, result in all_results.items():
        cell_text.append([
            strategy_name,
            str(strategy_page_map.get(strategy_name, '-')),
            f"€{result['median_final_bankroll']:,.2f}",
            f"€{result['final_bankroll_mode']:,.2f}",
            f"{result['growth_rate']:.2%}",
            f"€{result.get('median_rakeback_eur', 0.0):,.2f}",
            f"{result['risk_of_ruin']:.2f}",
            f"{result['target_prob']:.2f}",
            f"€{result['p5']:,.2f}",
            f"€{result['p2_5']:,.2f}"
        ])

    if not cell_text: return

    col_widths = [len(h) for h in header]
    for row in cell_text:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    total_width = sum(col_widths)
    col_widths_ratio = [w / total_width for w in col_widths]

    fig, ax = plt.subplots(figsize=(16, 2 + len(cell_text) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=cell_text, colLabels=header, loc='center', cellLoc='center', colWidths=col_widths_ratio)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.5)
    ax.set_title('Strategy Comparison Summary', fontweight="bold", pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def generate_pdf_report(all_results, config, timestamp_str):
    """
    Generates the entire multi-page PDF report, writing to an in-memory buffer.
    """
    pdf_buffer = io.BytesIO()
    strategy_page_map = {}
    current_page_count = 3
    lines_per_page = 45

    for strategy_name, result in all_results.items():
        strategy_config = config['STRATEGIES_TO_RUN'][strategy_name]
        strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA'])
        strategy_page_map[strategy_name] = current_page_count + 1
        report_lines = get_strategy_report_lines(strategy_name, result, strategy_obj, config)
        num_text_pages = (len(report_lines) + lines_per_page - 1) // lines_per_page # Ceiling division
        # There are now 3 plots per strategy
        num_plot_pages = 3
        current_page_count += num_text_pages + num_plot_pages

    # Calculate a representative input win rate for the plot's label
    total_sample_hands = sum(s['sample_hands'] for s in config['STAKES_DATA'])
    if total_sample_hands > 0:
        weighted_input_wr = sum(s['ev_bb_per_100'] * s['sample_hands'] for s in config['STAKES_DATA']) / total_sample_hands
    else:
        # Fallback if no sample hands are provided
        weighted_input_wr = config['STAKES_DATA'][0]['ev_bb_per_100'] if config['STAKES_DATA'] else 1.5

    with PdfPages(pdf_buffer) as pdf:
        create_title_page(pdf, timestamp_str)
        save_summary_table_to_pdf(pdf, all_results, strategy_page_map, config)
        plot_median_progression_comparison(all_results, config, pdf=pdf)

        for strategy_name, result in all_results.items():
            strategy_config = config['STRATEGIES_TO_RUN'][strategy_name]
            strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA'])
            page_num = strategy_page_map.get(strategy_name, 0)
            report_lines_for_writing = get_strategy_report_lines(strategy_name, result, strategy_obj, config)
            write_strategy_report_to_pdf(pdf, report_lines_for_writing, page_number_info={'current': page_num})
            plot_strategy_progression(result['bankroll_histories'], result['hands_histories'], strategy_name, config, pdf=pdf)
            plot_final_bankroll_distribution(result['final_bankrolls'], result, strategy_name, config, pdf=pdf)
            plot_assigned_wr_distribution(
                result['avg_assigned_wr_per_sim'],
                result['median_run_assigned_wr'],
                weighted_input_wr,
                strategy_name,
                pdf=pdf)

    pdf_buffer.seek(0)
    return pdf_buffer
# =================================================================================
#   MAIN CONTROLLER FUNCTIONS
# =================================================================================

def initialize_strategy(strategy_name, strategy_config, stakes_data):
    """Initializes a strategy object from its configuration."""
    strategy_type = strategy_config.get("type", "standard")
    if strategy_type == "hysteresis":
        return HysteresisStrategy(strategy_name, strategy_config["num_buy_ins"], stakes_data)
    elif strategy_type == "standard":
        return BankrollManagementStrategy(strategy_name, strategy_config["rules"], stakes_data)
    else:
        raise ValueError(f"Unknown strategy type '{strategy_type}' for '{strategy_name}'")

def run_full_analysis(config):
    """
    The main entry point for running the entire simulation analysis.
    This function is called by the Streamlit app.
    """
    all_results = {}
    
    # --- Setup common data for all simulations ---
    all_session_profits_bb, all_win_rates, rng = setup_simulation_parameters(config)
    stake_level_map = {stake['name']: i for i, stake in enumerate(sorted(config['STAKES_DATA'], key=lambda s: s['bb_size']))}
    stake_name_map = {v: k for k, v in stake_level_map.items()}

    for strategy_name, strategy_config in config['STRATEGIES_TO_RUN'].items():
        strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA'])

        # Determine which simulation function to use based on the class type
        if isinstance(strategy_obj, HysteresisStrategy):
            bankroll_histories, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns = run_sticky_simulation_vectorized(strategy_obj, all_session_profits_bb, rng, stake_level_map, config)
        else:
            bankroll_histories, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns = run_multiple_simulations_vectorized(strategy_obj, all_session_profits_bb, rng, stake_level_map, config)

        # Analyze the results and store them
        all_results[strategy_name] = analyze_strategy_results(
            strategy_name, strategy_obj, bankroll_histories, hands_per_stake_histories, rakeback_histories, all_win_rates, rng,
            peak_stake_levels, demotion_flags, stake_level_map, stake_name_map, max_drawdowns, config
        )

    return all_results
