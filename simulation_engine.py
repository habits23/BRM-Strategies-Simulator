import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.stats import t, gaussian_kde
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import pandas as pd
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

def setup_simulation_parameters(config, seed):
    """Initializes and returns common parameters needed for simulations."""
    rng = np.random.default_rng(seed)
    all_win_rates = {}
    # This is the critical step: Generate one "long-term luck factor" per simulation run.
    long_term_luck_factors = rng.normal(loc=0.0, scale=1.0, size=config['NUMBER_OF_SIMULATIONS'])

    for i, stake in enumerate(config['STAKES_DATA']):
        name = stake["name"]
        ev_bb_per_100 = stake.get("ev_bb_per_100", stake["bb_per_100"])
        std_dev_per_100 = stake["std_dev_per_100"]
        sample_hands = stake["sample_hands"]
        win_rate_drop = stake["win_rate_drop"]

        if i > 0:
            previous_stake_name = config['STAKES_DATA'][i-1]["name"]
            # The prior is the entire array of win rates from the previous stake.
            # This is CRITICAL to preserve the per-simulation luck correlation across stakes.
            prior_win_rate = all_win_rates[previous_stake_name] - win_rate_drop
        else:
            prior_win_rate = ev_bb_per_100

        all_win_rates[name] = calculate_effective_win_rate(ev_bb_per_100, std_dev_per_100, sample_hands, long_term_luck_factors, prior_win_rate, config)

    return all_win_rates, rng


def calculate_hand_block_outcome(current_bankrolls, proportions_per_stake, all_win_rates, rng, active_mask, config):
    """Calculates the profit and hands played for a block of hands across all simulations."""
    # This function models SHORT-TERM LUCK (session-to-session variance).
    # It takes the pre-calculated `all_win_rates` (the Assigned WR, which includes long-term luck)
    # and adds a new random variance component for this specific block of hands.
    # The final result of the simulation, after accumulating all these short-term results,
    # is the "Play WR" or "Realized WR".

    session_profits_eur = np.zeros_like(current_bankrolls)
    hands_per_stake_this_session = {stake["name"]: np.zeros_like(current_bankrolls, dtype=int) for stake in config['STAKES_DATA']}

    for stake in config['STAKES_DATA']:
        name = stake["name"]
        bb_size = stake["bb_size"]
        std_dev_per_100 = stake["std_dev_per_100"]

        # Define the mask based on which simulations are active AND have a proportion for this stake
        proportions_mask = active_mask & (proportions_per_stake[name] > 0)
        if np.any(proportions_mask):
            proportions = proportions_per_stake[name][proportions_mask]
            hands_for_stake = (proportions * config['HANDS_PER_CHECK']).astype(int)
            hands_per_stake_this_session[name][proportions_mask] = hands_for_stake

            num_100_hand_blocks = hands_for_stake / 100.0

            # 1. Profit from Skill + Long-Term Luck (the pre-calculated Assigned WR)
            # This component is deterministic for a given simulation run.
            profit_from_ev_bb = all_win_rates[name][proportions_mask] * num_100_hand_blocks

            # 2. Profit from Short-Term Variance (the "Session Luck Factor")
            # A new random number is generated for every block to simulate session-to-session luck.
            random_noise_component = rng.normal(loc=0.0, scale=1.0, size=np.sum(proportions_mask))
            profit_from_variance_bb = random_noise_component * std_dev_per_100 * np.sqrt(num_100_hand_blocks)

            total_profit_bb = profit_from_ev_bb + profit_from_variance_bb
            session_profits_eur[proportions_mask] += total_profit_bb * bb_size

    total_rakeback_eur = np.zeros_like(current_bankrolls)
    if config['RAKEBACK_PERCENTAGE'] > 0:
        for stake in config['STAKES_DATA']:
            if "rake_bb_per_100" in stake and np.any(hands_per_stake_this_session[stake["name"]] > 0):
                rake_paid_eur = (hands_per_stake_this_session[stake["name"]] / 100.0) * stake["rake_bb_per_100"] * stake["bb_size"]
                total_rakeback_eur += rake_paid_eur * config['RAKEBACK_PERCENTAGE']

    final_session_profit = session_profits_eur + total_rakeback_eur
    return final_session_profit, hands_per_stake_this_session, total_rakeback_eur

def _calculate_withdrawal_amounts(current_bankrolls, bankroll_at_start_of_month, settings, due_mask, ruin_threshold):
    """
    Calculates the withdrawal amount for each simulation based on its strategy.
    This is a helper function for the main simulation loops.
    """
    withdrawal_amounts = np.zeros_like(current_bankrolls)

    # Safety check: only withdraw if bankroll is above the minimum threshold specified in the UI
    eligible_mask = due_mask & (current_bankrolls >= settings["min_bankroll"])
    if not np.any(eligible_mask):
        return withdrawal_amounts

    strategy = settings["strategy"]
    value = settings["value"]

    calculated_withdrawals = np.zeros_like(current_bankrolls)

    if "Fixed Amount" in strategy:
        calculated_withdrawals[eligible_mask] = value

    elif "Percentage of Profits" in strategy:
        profits = current_bankrolls - bankroll_at_start_of_month
        # Only withdraw from positive profits
        positive_profit_mask = eligible_mask & (profits > 0)
        if np.any(positive_profit_mask):
            calculated_withdrawals[positive_profit_mask] = profits[positive_profit_mask] * (value / 100.0)

    elif "Withdraw Down to Threshold" in strategy:
        amount_over_threshold = current_bankrolls - value
        # Only withdraw if bankroll is over the threshold
        over_threshold_mask = eligible_mask & (amount_over_threshold > 0)
        if np.any(over_threshold_mask):
            calculated_withdrawals[over_threshold_mask] = amount_over_threshold[over_threshold_mask]

    # Final check: Don't let withdrawal cause bankroll to go below ruin threshold.
    # This is a safety net, though min_br_for_withdrawal should usually prevent this.
    max_possible_withdrawal = current_bankrolls - ruin_threshold
    final_withdrawals = np.minimum(calculated_withdrawals, max_possible_withdrawal)
    final_withdrawals[final_withdrawals < 0] = 0 # Ensure withdrawals are not negative

    withdrawal_amounts[eligible_mask] = final_withdrawals[eligible_mask]
    return withdrawal_amounts

def calculate_effective_win_rate(ev_bb_per_100, std_dev_per_100, sample_hands, long_term_luck_factors, prior_win_rate, config):
    """Adjusts the observed EV win rate using a Bayesian-inspired 'shrinkage' method."""
    # This function calculates the "Assigned WR" which includes LONG-TERM LUCK.
    # It's called once per stake at the beginning of the simulation.
    # 1. It creates a "skill_estimate_wr" by blending the user's input with a prior, weighted by sample size.
    # 2. It calculates the uncertainty ("std_error") in that skill estimate.
    # 3. It applies the `long_term_luck_factors` (one persistent random number per simulation) scaled by the uncertainty.
    # The result is the "true" win rate (Assigned WR) for each individual simulation run.

    if sample_hands > 0:
        # --- The "Shrinkage" Effect ---
        # This determines the center of our win rate estimate. We "shrink" the user's
        # observed win rate (ev_bb_per_100) towards a more conservative prior estimate.
        # The amount of shrinkage depends on the weight of the evidence (sample_hands)
        # compared to the model's skepticism (PRIOR_SAMPLE_SIZE).
        #
        # - High sample_hands: data_weight is high -> skill_estimate_wr is close to user's input.
        # - Low sample_hands: data_weight is low -> skill_estimate_wr is "shrunk" towards the prior.
        data_weight = sample_hands / (sample_hands + config['PRIOR_SAMPLE_SIZE'])
        skill_estimate_wr = (data_weight * ev_bb_per_100) + ((1 - data_weight) * prior_win_rate)
    else:
        # If there are no hands, we blend the user's guess with the model's extrapolation.
        model_extrapolation = prior_win_rate
        user_estimate = ev_bb_per_100
        skill_estimate_wr = (config['ZERO_HANDS_INPUT_WEIGHT'] * user_estimate) + ((1 - config['ZERO_HANDS_INPUT_WEIGHT']) * model_extrapolation)

    # For a true sanity check, we can bypass the luck factor entirely.
    if config.get('PRIOR_SAMPLE_SIZE') >= 10_000_000:
        # The issue is that for the first stake, skill_estimate_wr is a float.
        # We must ensure it's an array of the correct size for all simulations.
        if isinstance(skill_estimate_wr, (int, float)):
            return np.full_like(long_term_luck_factors, skill_estimate_wr)
        return skill_estimate_wr # It's already an array for subsequent stakes

    # --- The "Uncertainty" Effect ---
    # This determines the *width* of the distribution of long-term luck.
    # We create a total "pool of evidence" by combining the user's sample with the prior.
    # A larger pool means more certainty and thus a smaller standard error.
    #
    # - High sample_hands or high PRIOR_SAMPLE_SIZE: Large evidence pool -> low std_error -> narrow luck distribution.
    # - Low sample_hands and low PRIOR_SAMPLE_SIZE: Small evidence pool -> high std_error -> wide luck distribution.
    #
    # This is why a low PRIOR_SAMPLE_SIZE leads to a wider, more "lucky" range of outcomes.
    # The model has a small evidence pool, so it is very uncertain about the true skill,
    # and it expresses this uncertainty by simulating a wider range of possibilities.
    effective_sample_size_for_variance = sample_hands + config['PRIOR_SAMPLE_SIZE']
    N_blocks = max(1.0, effective_sample_size_for_variance / 100.0)
    std_error = std_dev_per_100 / np.sqrt(N_blocks)

    # Apply the luck factor. `long_term_luck_factors` is an array of N(0,1) random numbers,
    # one for each simulation. This is the "long-term luck" component that persists for
    # the entire duration of a single simulation run.
    long_term_luck_adjustment = long_term_luck_factors * std_error

    # The final Assigned WR is the shrunken skill estimate plus the long-term luck adjustment.
    return skill_estimate_wr + long_term_luck_adjustment

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

def resolve_proportions(rule, rng):
    """Resolves a strategy rule into a dictionary of proportions (floats summing to 1)."""
    percentages = {}
    fixed_ratios = {}
    for stake, val in rule.items():
        if isinstance(val, str):
            sanitized_val = val.replace('%', '').replace(' ', '')
            if "-" in sanitized_val:
                percentages[stake] = sample_percentage_range(sanitized_val, rng)
            else:
                try:
                    percentages[stake] = float(sanitized_val)
                except ValueError:
                    pass
        elif isinstance(val, int) and val > 0:
            fixed_ratios[stake] = val

    if fixed_ratios:
        return normalize_percentages(fixed_ratios)
    if percentages:
        return normalize_percentages(percentages)
    return {}

def _initialize_simulation_state(num_sims, num_checks, config):
    """Initializes all common numpy arrays for a simulation run."""
    bankroll_history = np.full((num_sims, num_checks + 1), config['STARTING_BANKROLL_EUR'], dtype=float)
    hands_per_stake_histories = {stake['name']: np.zeros((num_sims, num_checks + 1), dtype=int) for stake in config['STAKES_DATA']}
    rakeback_histories = np.zeros((num_sims, num_checks + 1), dtype=float)

    # Withdrawal state
    withdrawal_settings = config.get("WITHDRAWAL_SETTINGS", {"enabled": False})
    total_withdrawn_histories = np.zeros((num_sims, num_checks + 1), dtype=float)
    hands_since_last_withdrawal = np.zeros(num_sims, dtype=int) if withdrawal_settings.get("enabled") else None
    bankroll_at_last_withdrawal = np.full(num_sims, config['STARTING_BANKROLL_EUR'], dtype=float) if withdrawal_settings.get("enabled") else None

    # Drawdown state
    peak_bankrolls_so_far = np.full(num_sims, config['STARTING_BANKROLL_EUR'], dtype=float)
    max_drawdowns_so_far = np.zeros(num_sims, dtype=float)

    # Stop-loss state
    is_stopped_out = np.zeros(num_sims, dtype=bool)
    stop_loss_triggers = np.zeros(num_sims, dtype=int)

    # Underwater state
    underwater_hands_count = np.zeros(num_sims, dtype=int)
    integrated_drawdown = np.zeros(num_sims, dtype=float)
    
    # --- Downswing Extent/Stretch Analysis Initialization ---
    depth_thresholds_bb = np.array(config.get('DOWNSWING_DEPTH_THRESHOLDS_BB', []))
    duration_thresholds_hands = np.array(config.get('DOWNSWING_DURATION_THRESHOLDS_HANDS', []))
    # We will store a boolean flag for each sim and each threshold
    downswing_depth_exceeded = np.zeros((num_sims, len(depth_thresholds_bb)), dtype=bool)
    downswing_duration_exceeded = np.zeros((num_sims, len(duration_thresholds_hands)), dtype=bool)
    # Track the BB size at the time of the peak for accurate downswing depth calculation
    bb_size_at_peak = np.zeros(num_sims, dtype=float)
    current_underwater_stretch_hands = np.zeros(num_sims, dtype=int)

    return (
        bankroll_history, hands_per_stake_histories, rakeback_histories,
        total_withdrawn_histories, hands_since_last_withdrawal, bankroll_at_last_withdrawal,
        peak_bankrolls_so_far, max_drawdowns_so_far,
        is_stopped_out, stop_loss_triggers, 
        downswing_depth_exceeded, downswing_duration_exceeded, bb_size_at_peak, current_underwater_stretch_hands, # New
        underwater_hands_count, integrated_drawdown
    )

def _process_simulation_block(
    # --- Loop variables ---
    i, rng,
    # --- Static config & data ---
    config, all_win_rates, stake_bb_size_map,
    # --- Per-block calculated data ---
    proportions_per_stake, bb_sizes_eur,
    # Mutable state variables
    bankroll_history, hands_per_stake_histories, rakeback_histories,
    total_withdrawn_histories, hands_since_last_withdrawal, bankroll_at_last_withdrawal,
    peak_bankrolls_so_far, max_drawdowns_so_far,
    is_stopped_out, stop_loss_triggers,
    underwater_hands_count, integrated_drawdown,
    downswing_depth_exceeded, downswing_duration_exceeded, bb_size_at_peak, current_underwater_stretch_hands
):
    """
    Processes a single block/step of the simulation for all runs.
    This function contains the logic common to both standard and sticky strategies.
    It mutates the state arrays in place and returns a boolean indicating if the loop should continue.
    """
    num_sims = bankroll_history.shape[0]
    current_bankrolls = bankroll_history[:, i]
    total_proportions = sum(proportions_per_stake.values())
    # Determine who is active this block (not ruined, has a table mix, and is not stopped out)
    active_mask = (current_bankrolls >= config['RUIN_THRESHOLD']) & (total_proportions > 0) & ~is_stopped_out

    # Reset the stopped out flag for the next loop. Any sim that was stopped out can now play again.
    is_stopped_out.fill(False)

    if not np.any(active_mask):
        bankroll_history[:, i+1:] = bankroll_history[:, i][:, np.newaxis]
        for stake_name in hands_per_stake_histories:
            hands_per_stake_histories[stake_name][:, i+1:] = hands_per_stake_histories[stake_name][:, i][:, np.newaxis]
        return False # Signal to break the main loop

    block_profits_eur, hands_per_stake_this_block, block_rakeback_eur = calculate_hand_block_outcome(
        current_bankrolls, proportions_per_stake, all_win_rates, rng, active_mask, config
    )
    # --- Stop-Loss Logic ---
    if config.get("STOP_LOSS_BB", 0) > 0:
        highest_bb_size = np.zeros(num_sims)
        for stake_name, bb_size in stake_bb_size_map.items():
            played_this_stake_mask = proportions_per_stake[stake_name] > 0
            highest_bb_size[played_this_stake_mask] = np.maximum(highest_bb_size[played_this_stake_mask], bb_size)
        stop_loss_eur = config["STOP_LOSS_BB"] * highest_bb_size
        valid_stop_loss_mask = active_mask & (stop_loss_eur > 0)
        profit_from_play = block_profits_eur - block_rakeback_eur
        triggered_mask = (profit_from_play < -stop_loss_eur) & valid_stop_loss_mask
        if np.any(triggered_mask):
            is_stopped_out[triggered_mask] = True
            stop_loss_triggers[triggered_mask] += 1
    
    # Update bankrolls with play profits first, before withdrawal calculations
    temp_bankrolls = current_bankrolls + block_profits_eur

    # --- Withdrawal Logic ---
    withdrawal_settings = config.get("WITHDRAWAL_SETTINGS", {"enabled": False})
    withdrawal_amounts_this_block = np.zeros(num_sims, dtype=float)
    if withdrawal_settings.get("enabled"):
        total_hands_this_block = np.sum(list(hands_per_stake_this_block.values()), axis=0)
        hands_since_last_withdrawal += np.where(active_mask, total_hands_this_block, 0)
        due_for_withdrawal_mask = (hands_since_last_withdrawal >= withdrawal_settings["monthly_volume"]) & active_mask
        if np.any(due_for_withdrawal_mask):
            withdrawal_amounts_this_block = _calculate_withdrawal_amounts(
                temp_bankrolls, bankroll_at_last_withdrawal, withdrawal_settings,
                due_for_withdrawal_mask, config['RUIN_THRESHOLD']
            )
            temp_bankrolls -= withdrawal_amounts_this_block
            hands_since_last_withdrawal[due_for_withdrawal_mask] = 0
            bankroll_at_last_withdrawal[due_for_withdrawal_mask] = temp_bankrolls[due_for_withdrawal_mask]

    total_withdrawn_histories[:, i+1] = total_withdrawn_histories[:, i] + withdrawal_amounts_this_block
    new_bankrolls = temp_bankrolls
    bankroll_history[:, i+1] = np.where(active_mask, new_bankrolls, current_bankrolls)

    # --- Downswing Analysis (Duration Part 1: Update current stretch) ---
    # Identify which simulations are currently underwater and add hands to their current stretch
    is_underwater_mask = new_bankrolls < peak_bankrolls_so_far
    current_underwater_stretch_hands[is_underwater_mask] += config['HANDS_PER_CHECK']

    # --- Underwater Time Calculation ---
    underwater_mask = (bankroll_history[:, i+1] < peak_bankrolls_so_far) & active_mask
    if np.any(underwater_mask):
        underwater_hands_count[underwater_mask] += config['HANDS_PER_CHECK']
        drawdown_amount = peak_bankrolls_so_far - bankroll_history[:, i+1]
        integrated_drawdown[underwater_mask] += drawdown_amount[underwater_mask] * config['HANDS_PER_CHECK']

    # Identify simulations that made a new peak in this block
    made_new_peak_mask = new_bankrolls > peak_bankrolls_so_far

    # --- Downswing Analysis (Duration Part 2: Check thresholds for ended stretches) ---
    duration_thresholds_hands = np.array(config.get('DOWNSWING_DURATION_THRESHOLDS_HANDS', []))
    if np.any(made_new_peak_mask) and len(duration_thresholds_hands) > 0:
        ended_stretch_durations = current_underwater_stretch_hands[made_new_peak_mask]
        duration_check = ended_stretch_durations[:, np.newaxis] >= duration_thresholds_hands
        downswing_duration_exceeded[made_new_peak_mask] |= duration_check

    # Reset the stretch counter for sims that made a new peak
    current_underwater_stretch_hands[made_new_peak_mask] = 0

    # --- Downswing Analysis (Depth) ---
    # This must be done BEFORE updating the peak bankroll for the current block.
    depth_thresholds_bb = np.array(config.get('DOWNSWING_DEPTH_THRESHOLDS_BB', []))
    if len(depth_thresholds_bb) > 0:
        # Update the BB size at the time of the peak for any sims that made a new high
        bb_size_at_peak[made_new_peak_mask] = bb_sizes_eur[made_new_peak_mask]
        # Calculate current downswing in EUR from the PREVIOUS peak
        current_downswings_eur = peak_bankrolls_so_far - new_bankrolls
        valid_peak_mask = bb_size_at_peak > 0
        downswings_bb = np.divide(current_downswings_eur, bb_size_at_peak, out=np.zeros_like(current_downswings_eur), where=valid_peak_mask)
        depth_check = downswings_bb[:, np.newaxis] >= depth_thresholds_bb
        downswing_depth_exceeded |= depth_check

    # --- Maximum Drawdown Calculation ---
    # Now we can update the peak bankroll with the results from the current block.
    np.maximum(peak_bankrolls_so_far, bankroll_history[:, i+1], out=peak_bankrolls_so_far)
    current_drawdowns = peak_bankrolls_so_far - bankroll_history[:, i+1]
    np.maximum(max_drawdowns_so_far, current_drawdowns, out=max_drawdowns_so_far)

    # --- History Updates ---
    rakeback_histories[:, i+1] = rakeback_histories[:, i] + np.where(active_mask, block_rakeback_eur, 0)
    for stake_name, hands_array in hands_per_stake_this_block.items():
        hands_per_stake_histories[stake_name][:, i+1] = hands_per_stake_histories[stake_name][:, i] + np.where(active_mask, hands_array, 0)
    
    return True # Signal to continue loop

def run_multiple_simulations_vectorized(strategy, all_win_rates, rng, stake_level_map, config):
    """
    Runs all simulations at once using vectorized NumPy operations for speed.
    This version dynamically resolves table mixes for each session.
    """
    num_sims = config['NUMBER_OF_SIMULATIONS']
    num_checks = int(np.ceil(config['TOTAL_HANDS_TO_SIMULATE'] / config['HANDS_PER_CHECK']))

    (
        bankroll_history, hands_per_stake_histories, rakeback_histories,
        total_withdrawn_histories, hands_since_last_withdrawal, bankroll_at_last_withdrawal,
        peak_bankrolls_so_far, max_drawdowns_so_far,
        is_stopped_out, stop_loss_triggers, 
        downswing_depth_exceeded, downswing_duration_exceeded, bb_size_at_peak, current_underwater_stretch_hands, # New
        underwater_hands_count, integrated_drawdown
    ) = _initialize_simulation_state(num_sims, num_checks, config)

    # --- Demotion Tracking Initialization ---
    initial_rule = strategy.get_table_mix(config['STARTING_BANKROLL_EUR'])
    initial_stakes_with_tables = [stake for stake, count in initial_rule.items() if (isinstance(count, int) and count > 0) or (isinstance(count, str) and float(count.replace('%','').split('-')[0]) > 0)]
    initial_level = max([stake_level_map[s] for s in initial_stakes_with_tables]) if initial_stakes_with_tables else -1
    peak_stake_levels = np.full(num_sims, initial_level, dtype=int)
    demotion_flags = {level: np.zeros(num_sims, dtype=bool) for level in stake_level_map.values()}

    stake_bb_size_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
    thresholds, rules = strategy.get_rules_as_vectors()

    for i in range(num_checks):
        current_bankrolls = bankroll_history[:, i]

        # Store previous peak levels to detect demotions
        previous_peak_levels = peak_stake_levels.copy()

        # Determine the table mix for each simulation based on its current bankroll
        proportions_per_stake = {stake["name"]: np.zeros(num_sims, dtype=float) for stake in config['STAKES_DATA']}

        remaining_mask = np.ones_like(current_bankrolls, dtype=bool)
        for threshold, rule in zip(thresholds, rules):
            current_mask = (current_bankrolls >= threshold) & remaining_mask
            if not np.any(current_mask):
                continue

            indices = np.where(current_mask)[0]
            for sim_idx in indices:
                resolved_proportions = resolve_proportions(rule, rng)
                for stake_name, prop in resolved_proportions.items():
                    proportions_per_stake[stake_name][sim_idx] = prop
            remaining_mask[current_mask] = False

        # Handle simulations with bankroll below the lowest threshold by applying the lowest rule
        if np.any(remaining_mask) and rules:
            lowest_rule = rules[-1] # The last rule is the lowest threshold
            indices = np.where(remaining_mask)[0]
            for sim_idx in indices:
                resolved_proportions = resolve_proportions(lowest_rule, rng)
                for stake_name, prop in resolved_proportions.items():
                    proportions_per_stake[stake_name][sim_idx] = prop

        # --- Demotion Tracking Logic for the current block ---
        current_levels = np.full(config['NUMBER_OF_SIMULATIONS'], -1, dtype=int)
        for stake_name, level in stake_level_map.items():
            has_tables_mask = proportions_per_stake[stake_name] > 0
            current_levels[has_tables_mask] = np.maximum(current_levels[has_tables_mask], level)

        demotion_from_peak_mask = current_levels < previous_peak_levels
        for level in stake_level_map.values():
            if level > 0:
                demoted_this_session_mask = (previous_peak_levels == level) & demotion_from_peak_mask
                demotion_flags[level][demoted_this_session_mask] = True

        # Update peak levels for the next session
        peak_stake_levels = np.maximum(previous_peak_levels, current_levels)

        # Calculate bb_sizes_eur for the current block
        bb_sizes_eur = np.zeros(num_sims, dtype=float)
        for stake_name, prop_array in proportions_per_stake.items():
            bb_sizes_eur += prop_array * stake_bb_size_map[stake_name]

        should_continue = _process_simulation_block(
            i, rng,
            config, all_win_rates, stake_bb_size_map,
            proportions_per_stake, bb_sizes_eur,
            bankroll_history, hands_per_stake_histories, rakeback_histories,
            total_withdrawn_histories, hands_since_last_withdrawal, bankroll_at_last_withdrawal,
            peak_bankrolls_so_far, max_drawdowns_so_far,
            is_stopped_out, stop_loss_triggers,
            underwater_hands_count, integrated_drawdown,
            downswing_depth_exceeded, downswing_duration_exceeded, bb_size_at_peak, current_underwater_stretch_hands
        )
        if not should_continue:
            break

    # --- Final Downswing Analysis Check ---
    # For sims that are still underwater at the very end, check their final stretch duration
    duration_thresholds_hands = np.array(config.get('DOWNSWING_DURATION_THRESHOLDS_HANDS', []))
    if len(duration_thresholds_hands) > 0:
        final_underwater_mask = bankroll_history[:, -1] < peak_bankrolls_so_far
        if np.any(final_underwater_mask):
            final_stretch_durations = current_underwater_stretch_hands[final_underwater_mask]
            duration_check = final_stretch_durations[:, np.newaxis] >= duration_thresholds_hands
            downswing_duration_exceeded[final_underwater_mask] |= duration_check

    return (
        bankroll_history, hands_per_stake_histories, rakeback_histories, peak_stake_levels,
        demotion_flags, max_drawdowns_so_far, stop_loss_triggers, underwater_hands_count,
        integrated_drawdown, total_withdrawn_histories,
        downswing_depth_exceeded, downswing_duration_exceeded # New return values
    )

# =================================================================================
#   RESULTS ANALYSIS FUNCTIONS
# =================================================================================

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

def _calculate_percentile_win_rates(final_bankrolls, all_win_rates, hands_per_stake_histories, rakeback_histories, total_withdrawn_histories, config, bb_size_map):
    """Calculates assigned and realized win rates for simulations closest to key percentiles."""
    percentile_win_rates = {}
    percentiles_to_find = {'2.5th': 2.5, '5th': 5, '25th': 25, 'Median': 50, '75th': 75, '95th': 95, '97.5th': 97.5}
    final_withdrawn = total_withdrawn_histories[:, -1]
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
            stake_wrs['Assigned WR'] = f"{assigned_weighted_wr_sum / total_hands_for_sim:.2f}"
            avg_bb_size = weighted_bb_size_sum / total_hands_for_sim

            if avg_bb_size > 0:
                # CRITICAL: Total profit must include money that was withdrawn.
                # Total value generated = (Final BR - Start BR) + Total Withdrawn
                total_profit_eur = (final_bankrolls[closest_sim_index] + final_withdrawn[closest_sim_index]) - config['STARTING_BANKROLL_EUR']
                profit_from_play_eur = total_profit_eur - final_rakeback[closest_sim_index]
                realized_wr_val = (profit_from_play_eur / avg_bb_size) / (total_hands_for_sim / 100)
                assigned_wr_val = assigned_weighted_wr_sum / total_hands_for_sim
                variance_impact_val = realized_wr_val - assigned_wr_val

                stake_wrs['Realized WR (Play)'] = f"{realized_wr_val:.2f}"
                stake_wrs['Rakeback (bb/100)'] = f"{(final_rakeback[closest_sim_index] / avg_bb_size) / (total_hands_for_sim / 100):.2f}"
                stake_wrs['Variance Impact'] = f"{variance_impact_val:+.2f}" # Use + to show positive/negative impact
            else: # Handle case where avg_bb_size is 0
                stake_wrs['Realized WR (Play)'] = "N/A"
                stake_wrs['Rakeback (bb/100)'] = "N/A"
                stake_wrs['Variance Impact'] = "N/A"

        percentile_win_rates[f"{name} Percentile"] = stake_wrs
    return percentile_win_rates

def analyze_strategy_results(strategy_name, strategy_obj, bankroll_histories, hands_per_stake_histories, rakeback_histories, all_win_rates, rng, peak_stake_levels, demotion_flags, stake_level_map, stake_name_map, max_drawdowns, stop_loss_triggers, underwater_hands_count, integrated_drawdown, total_withdrawn_histories, downswing_depth_exceeded, downswing_duration_exceeded, config):
    """Takes the raw simulation output and calculates all the necessary metrics and analytics."""
    bb_size_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
    total_hands_histories = np.sum(list(hands_per_stake_histories.values()), axis=0)
    final_withdrawn = total_withdrawn_histories[:, -1]
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
        # Sort by string representation of keys to prevent TypeErrors with mixed-type keys (e.g., 'NL50' and NaN from empty rows)
        sorted_items = sorted(table_mix.items(), key=lambda item: str(item[0])) if table_mix else []
        mix_str = ", ".join(f"{s}: {v}" for s, v in sorted_items) if sorted_items else "No Play"
        final_stake_counts[mix_str] += 1

        # This is the more direct calculation for the UI summary.
        if not table_mix:
            final_highest_stake_counts["No Play"] += 1
        else:
            # Filter the mix to only include stakes that are actually played.
            # This is critical for the Hysteresis strategy, which includes all stakes in its
            # rules, with most set to "0%". The original code would incorrectly find the
            # highest stake overall (e.g., NL200) instead of the one with a "100%" allocation.
            active_stakes_in_mix = {
                stake: value for stake, value in table_mix.items()
                if (isinstance(value, str) and float(value.replace('%','').split('-')[0]) > 0) or (isinstance(value, int) and value > 0)
            }

            if not active_stakes_in_mix:
                final_highest_stake_counts["No Play"] += 1
            else:
                highest_stake = max(active_stakes_in_mix.keys(), key=lambda s: stake_order_map.get(s, -1))
                final_highest_stake_counts[highest_stake] += 1

    final_stake_distribution = {mix_str: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for mix_str, count in final_stake_counts.items()}
    final_highest_stake_distribution = {stake: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for stake, count in final_highest_stake_counts.items()}

    percentile_win_rates = _calculate_percentile_win_rates(final_bankrolls, all_win_rates, hands_per_stake_histories, rakeback_histories, total_withdrawn_histories, config, bb_size_map)

    median_max_downswing = np.median(max_drawdowns)
    p95_max_downswing = np.percentile(max_drawdowns, 95)

    # Calculate median rakeback
    final_rakeback = rakeback_histories[:, -1]
    median_rakeback_eur = np.median(final_rakeback)

    # Calculate median profit from play
    # This now correctly includes withdrawn amounts as part of the total profit generated.
    total_profit_per_sim = (final_bankrolls + final_withdrawn) - config['STARTING_BANKROLL_EUR']
    profit_from_play_per_sim = total_profit_per_sim - final_rakeback
    median_profit_from_play_eur = np.median(profit_from_play_per_sim)

    # Calculate median stop-losses
    median_stop_losses = np.median(stop_loss_triggers) if stop_loss_triggers is not None else 0

    # Calculate median hands played
    median_hands_played = np.median(total_hands_per_sim)

    # Calculate median time spent underwater
    time_underwater_pct = np.divide(
        underwater_hands_count,
        total_hands_per_sim,
        out=np.zeros_like(underwater_hands_count, dtype=float),
        where=total_hands_per_sim != 0
    ) * 100
    median_time_underwater_pct = np.median(time_underwater_pct)

    # Calculate median integrated drawdown
    median_integrated_drawdown = np.median(integrated_drawdown)

    final_bankroll_mode = calculate_binned_mode(final_bankrolls, config['RUIN_THRESHOLD'])
    target_achieved_count = np.sum(np.any(bankroll_histories >= config['TARGET_BANKROLL'], axis=1))
    busted_runs = np.sum(np.any(bankroll_histories <= config['RUIN_THRESHOLD'], axis=1))
    risk_of_ruin_percent = (busted_runs / config['NUMBER_OF_SIMULATIONS']) * 100
    percentiles = {p: np.percentile(final_bankrolls, p) for p in [2.5, 5, 25, 50, 75, 95, 97.5]}
    median_growth_rate = (percentiles[50] - config['STARTING_BANKROLL_EUR']) / config['STARTING_BANKROLL_EUR'] if config['STARTING_BANKROLL_EUR'] > 0 else 0.0

    # --- Calculate Withdrawal Metrics ---
    median_total_withdrawn = np.median(final_withdrawn)
    p95_total_withdrawn = np.percentile(final_withdrawn, 95)

    # --- Calculate Total Return ---
    # Total return is the change in bankroll plus everything that was taken out.
    total_return_per_sim = (final_bankrolls - config['STARTING_BANKROLL_EUR']) + final_withdrawn
    median_total_return = np.median(total_return_per_sim)


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
        'median_max_downswing': median_max_downswing,
        'p95_max_downswing': p95_max_downswing,
        'max_downswings': max_drawdowns,
        'median_rakeback_eur': median_rakeback_eur,
        'median_profit_from_play_eur': median_profit_from_play_eur,
        'median_hands_played': median_hands_played,
        'median_stop_losses': median_stop_losses,
        'median_time_underwater_pct': median_time_underwater_pct,
        'median_integrated_drawdown': median_integrated_drawdown,
        'average_assigned_win_rates': average_assigned_win_rates,
        'avg_assigned_wr_per_sim': avg_assigned_wr_per_sim,
        'median_run_assigned_wr': median_run_assigned_wr,
        'median_total_withdrawn': median_total_withdrawn,
        'p95_total_withdrawn': p95_total_withdrawn,
        'median_total_return': median_total_return,
        'total_withdrawn_histories': total_withdrawn_histories,
        'downswing_analysis': (lambda: {
            'depth_probabilities': {
                int(t): p for t, p in zip(
                    config.get('DOWNSWING_DEPTH_THRESHOLDS_BB', []),
                    np.mean(downswing_depth_exceeded, axis=0) * 100
                )
            } if downswing_depth_exceeded is not None and len(config.get('DOWNSWING_DEPTH_THRESHOLDS_BB', [])) > 0 else {},
            'duration_probabilities': {
                int(t): p for t, p in zip(
                    config.get('DOWNSWING_DURATION_THRESHOLDS_HANDS', []),
                    np.mean(downswing_duration_exceeded, axis=0) * 100
                )
            } if downswing_duration_exceeded is not None and len(config.get('DOWNSWING_DURATION_THRESHOLDS_HANDS', [])) > 0 else {}
        })(),
    }
def run_sticky_simulation_vectorized(strategy, all_win_rates, rng, stake_level_map, config):
    """
    Runs a simulation with a specific 'sticky' bankroll management strategy.
    This version correctly handles multiple stakes by implementing a proper state machine.
    """
    num_sims = config['NUMBER_OF_SIMULATIONS']
    num_checks = int(np.ceil(config['TOTAL_HANDS_TO_SIMULATE'] / config['HANDS_PER_CHECK']))

    (
        bankroll_history, hands_per_stake_histories, rakeback_histories,
        total_withdrawn_histories, hands_since_last_withdrawal, bankroll_at_last_withdrawal,
        peak_bankrolls_so_far, max_drawdowns_so_far,
        is_stopped_out, stop_loss_triggers, # Corrected order
        downswing_depth_exceeded, downswing_duration_exceeded, bb_size_at_peak, current_underwater_stretch_hands,
        underwater_hands_count, integrated_drawdown
    ) = _initialize_simulation_state(num_sims, num_checks, config)

    # --- Demotion Tracking Initialization ---
    stake_rules = sorted(strategy.rules, key=lambda r: r['threshold'])
    rule_index_to_level = {i: stake_level_map[rule['stake_name']] for i, rule in enumerate(stake_rules)}

    current_stake_indices = np.zeros(num_sims, dtype=int)
    for idx in range(len(stake_rules) - 1, -1, -1):
        threshold = stake_rules[idx]['threshold']
        current_stake_indices[config['STARTING_BANKROLL_EUR'] >= threshold] = idx

    initial_levels = np.vectorize(rule_index_to_level.get)(current_stake_indices)
    peak_stake_levels = initial_levels.copy()
    demotion_flags = {level: np.zeros(num_sims, dtype=bool) for level in stake_level_map.values()}
    stake_bb_size_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}

    for i in range(num_checks):
        current_bankrolls = bankroll_history[:, i]

        previous_peak_levels = peak_stake_levels.copy()

        for idx in range(len(stake_rules) - 1):
            move_up_threshold = stake_rules[idx+1]['threshold']
            can_move_up_mask = (current_stake_indices == idx) & (current_bankrolls >= move_up_threshold)
            current_stake_indices[can_move_up_mask] = idx + 1

        for j in range(len(stake_rules) - 1, 0, -1):
            # Hysteresis move-down: A player at stake 'j' only moves down to 'j-1'
            # if their bankroll drops below the entry threshold for stake 'i-1'.
            # This creates a "sticky" buffer zone.
            move_down_threshold = stake_rules[j-1]['threshold']
            must_move_down_mask = (current_stake_indices == j) & (current_bankrolls < move_down_threshold)
            current_stake_indices[must_move_down_mask] = j - 1

        # --- Demotion Tracking Logic ---
        current_levels = np.vectorize(rule_index_to_level.get)(current_stake_indices)

        demotion_from_peak_mask = current_levels < previous_peak_levels
        for level in stake_level_map.values():
            demoted_this_block_mask = (previous_peak_levels == level) & demotion_from_peak_mask
            demotion_flags[level][demoted_this_block_mask] = True

        # Update peak levels for the next session
        peak_stake_levels = np.maximum(previous_peak_levels, current_levels)

        proportions_per_stake = {stake["name"]: np.zeros(num_sims, dtype=float) for stake in config['STAKES_DATA']}

        for rule_idx in range(len(stake_rules)):
            at_this_stake_mask = (current_stake_indices == rule_idx)
            if not np.any(at_this_stake_mask):
                continue

            rule = stake_rules[rule_idx]['tables']
            resolved_proportions = resolve_proportions(rule, rng)
            for stake_name, prop in resolved_proportions.items():
                proportions_per_stake[stake_name][at_this_stake_mask] = prop

        # Calculate bb_sizes_eur for the current block
        bb_sizes_eur = np.zeros(num_sims, dtype=float)
        for stake_name, prop_array in proportions_per_stake.items():
            bb_sizes_eur += prop_array * stake_bb_size_map[stake_name]

        should_continue = _process_simulation_block(
            i, rng,
            config, all_win_rates, stake_bb_size_map,
            proportions_per_stake, bb_sizes_eur,
            bankroll_history, hands_per_stake_histories, rakeback_histories,
            total_withdrawn_histories, hands_since_last_withdrawal, bankroll_at_last_withdrawal,
            peak_bankrolls_so_far, max_drawdowns_so_far,
            is_stopped_out, stop_loss_triggers,
            underwater_hands_count, integrated_drawdown,
            downswing_depth_exceeded, downswing_duration_exceeded, bb_size_at_peak, current_underwater_stretch_hands
        )
        if not should_continue:
            break

    # --- Final Downswing Analysis Check ---
    duration_thresholds_hands = np.array(config.get('DOWNSWING_DURATION_THRESHOLDS_HANDS', []))
    if len(duration_thresholds_hands) > 0:
        final_underwater_mask = bankroll_history[:, -1] < peak_bankrolls_so_far
        if np.any(final_underwater_mask):
            final_stretch_durations = current_underwater_stretch_hands[final_underwater_mask]
            duration_check = final_stretch_durations[:, np.newaxis] >= duration_thresholds_hands
            downswing_duration_exceeded[final_underwater_mask] |= duration_check

    return (
        bankroll_history, hands_per_stake_histories, rakeback_histories, peak_stake_levels,
        demotion_flags, max_drawdowns_so_far, stop_loss_triggers, underwater_hands_count,
        integrated_drawdown, total_withdrawn_histories,
        downswing_depth_exceeded, downswing_duration_exceeded # New return values
    )
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

def plot_final_bankroll_comparison(all_results, config, color_map=None, pdf=None):
    """
    Creates an overlapping density plot to compare the final bankroll distributions of all strategies.
    """
    # Dynamic height to prevent legend from squishing the plot with many strategies
    num_strategies = len(all_results)
    # A small increase in height per strategy to accommodate the legend
    fig_height = max(5, 4.0 + num_strategies * 0.25)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    if color_map is None:
        # Fallback for generating colors internally if no map is provided
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    # --- Dynamically determine the x-axis range for a clearer plot --- #
    # Combine all results to find a global range that fits all strategies well. #
    all_final_bankrolls = np.concatenate([res['final_bankrolls'] for res in all_results.values() if len(res['final_bankrolls']) > 0])

    # Filter out ruined runs to focus the plot on the distribution of successful outcomes.
    successful_runs = all_final_bankrolls[all_final_bankrolls > config['RUIN_THRESHOLD']]

    if len(successful_runs) > 5: # Need a few points to calculate percentiles robustly
        # Use the user-defined percentile to control the plot's "zoom".
        # This makes the zoom symmetrical (e.g., 95 shows 5th-95th, 99 shows 1st-99th).
        upper_percentile = config.get('PLOT_PERCENTILE_LIMIT', 99)
        lower_percentile = 100 - upper_percentile

        x_min = np.percentile(successful_runs, lower_percentile)
        x_max = np.percentile(successful_runs, upper_percentile)

        # Ensure the starting bankroll is always visible for context.
        x_min = min(x_min, config['STARTING_BANKROLL_EUR'])
        x_max = max(x_max, config['STARTING_BANKROLL_EUR'])

        # Add some visual padding to the limits.
        padding = (x_max - x_min) * 0.05
        x_min -= padding
        x_max += padding
    else:
        # Fallback for cases with very few successful runs (e.g., high risk of ruin)
        x_min = 0
        x_max = config.get('TARGET_BANKROLL', 5000) * 1.5

    x_grid = np.linspace(x_min, x_max, 1000)

    for strategy_name, result in all_results.items():
        final_bankrolls = result['final_bankrolls']
        color = color_map.get(strategy_name)
        if len(final_bankrolls) > 1:
            try:
                # FIX: Filter the data before calculating the KDE.
                # This prevents extreme outliers (the top 1% we've already
                # excluded from the x-axis) from skewing the density plot.
                # The result is a plot that better represents the main body of the distribution.
                filtered_bankrolls = final_bankrolls[(final_bankrolls >= x_min) & (final_bankrolls <= x_max) & (final_bankrolls > config['RUIN_THRESHOLD'])]
                if len(filtered_bankrolls) < 2: # Need at least 2 points for KDE
                    ax.hist(final_bankrolls, bins=50, density=True, alpha=0.5, label=f"{strategy_name} (hist)", color=color)
                    continue

                kde = gaussian_kde(filtered_bankrolls)
                density = kde(x_grid)
                ax.plot(x_grid, density, label=strategy_name, color=color, linewidth=2)
                ax.fill_between(x_grid, density, color=color, alpha=0.1)
            except (np.linalg.LinAlgError, ValueError):
                # Fallback for datasets that are not suitable for KDE
                ax.hist(final_bankrolls, bins=50, density=True, alpha=0.5, label=f"{strategy_name} (hist)", color=color)

    ax.set_title('Comparison of Final Bankroll Distributions', fontsize=16)
    ax.set_xlabel('Final Bankroll (EUR)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(left=x_min, right=x_max)
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_final_bankroll_distribution(final_bankrolls, result, strategy_name, config, pdf=None, color_map=None):
    """
    Plots the final bankroll distribution for a single strategy, highlighting median and mode.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use the strategy's assigned color if a map is provided
    plot_color = 'skyblue'
    if color_map and strategy_name in color_map:
        plot_color = color_map[strategy_name]

    # Filter out ruined runs to focus the plot on the distribution of successful outcomes.
    successful_runs = final_bankrolls[final_bankrolls > config['RUIN_THRESHOLD']]

    if len(successful_runs) > 5:
        upper_percentile = config.get('PLOT_PERCENTILE_LIMIT', 99)
        lower_percentile = 100 - upper_percentile

        x_min = np.percentile(successful_runs, lower_percentile)
        x_max = np.percentile(successful_runs, upper_percentile)

        x_min = min(x_min, config['STARTING_BANKROLL_EUR'])
        x_max = max(x_max, config['STARTING_BANKROLL_EUR'])

        padding = (x_max - x_min) * 0.05
        x_min -= padding
        x_max += padding
    else:
        x_min = 0
        x_max = config.get('TARGET_BANKROLL', 5000) * 1.5

    if len(successful_runs) > 0:
        # Filter the data for the histogram to match the x-axis limits for a cleaner plot
        filtered_for_hist = successful_runs[(successful_runs >= x_min) & (successful_runs <= x_max)]
        ax.hist(filtered_for_hist, bins=50, color=plot_color, edgecolor='black', alpha=0.7)

    # Add median and mode lines
    median_br = result.get('median_final_bankroll')
    mode_br = result.get('final_bankroll_mode')
    p5_br = result.get('p5')
    p95_br = result.get('p95') # Use pre-calculated percentile for consistency
    target_br = config.get('TARGET_BANKROLL')

    if median_br is not None:
        ax.axvline(median_br, color='darkgreen', linestyle='--', linewidth=2, label=f'Median: €{median_br:,.0f}')
    if mode_br is not None:
        ax.axvline(mode_br, color='darkred', linestyle=':', linewidth=2, label=f'Mode: €{mode_br:,.0f}')
    if p5_br is not None:
        ax.axvline(p5_br, color='orangered', linestyle='--', linewidth=1.5, label=f'5th %ile: €{p5_br:,.0f}')
    if p95_br is not None:
        ax.axvline(p95_br, color='purple', linestyle='--', linewidth=1.5, label=f'95th %ile: €{p95_br:,.0f}')
    if target_br is not None:
        ax.axvline(target_br, color='gold', linestyle='-.', linewidth=2, label=f'Target: €{target_br:,.0f}')

    ax.set_title(f'Final Bankroll Distribution for {strategy_name}', fontsize=16)
    ax.set_xlabel('Final Bankroll (EUR)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Simulations)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(left=x_min, right=x_max)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_median_progression_comparison(all_results, config, color_map=None, pdf=None):
    """Compares the median bankroll progression for all strategies on a single plot."""
    # Dynamic height to prevent legend from squishing the plot with many strategies
    num_strategies = len(all_results)
    # A small increase in height per strategy to accommodate the legend
    fig_height = max(5, 4.0 + num_strategies * 0.25)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    if color_map is None:
        # Fallback for generating colors internally if no map is provided
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    for strategy_name, result in all_results.items():
        color = color_map.get(strategy_name)
        ax.plot(result['hands_history'], result['median_history'], label=strategy_name, linewidth=2.5, color=color)

    ax.axhline(config['STARTING_BANKROLL_EUR'], color='gray', linestyle='--', label='Starting Bankroll')
    ax.axhline(config['TARGET_BANKROLL'], color='gold', linestyle='-.', label=f"Target: €{config['TARGET_BANKROLL']}")

    ax.set_title('Median Bankroll Progression Comparison Across All Strategies', fontsize=16)
    ax.set_xlabel('Total Hands Played', fontsize=12)
    ax.set_ylabel('Bankroll (EUR)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_max_downswing_distribution(max_downswings, result, strategy_name, pdf=None, color_map=None):
    """Creates a histogram of maximum downswings with key metrics highlighted."""
    if max_downswings is None or len(max_downswings) == 0:
        return plt.figure() # Return an empty figure if no data

    # Use the strategy's assigned color if a map is provided
    hist_color = 'salmon'
    if color_map and strategy_name in color_map:
        hist_color = color_map[strategy_name]

    # Filter out extreme outliers for better visibility
    max_x_limit = np.percentile(max_downswings, 99.0)
    filtered_downswings = max_downswings[max_downswings <= max_x_limit]

    median_downswing = result['median_max_downswing']
    p95_downswing = result['p95_max_downswing']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(filtered_downswings, bins=50, color=hist_color, edgecolor='black', alpha=0.7)

    ax.axvline(median_downswing, color='darkred', linestyle='dashed', linewidth=2, label=f'Median Downswing: €{median_downswing:,.2f}')
    ax.axvline(p95_downswing, color='purple', linestyle=':', linewidth=2, label=f'95th Pct. Downswing: €{p95_downswing:,.2f}')

    ax.set_title(f'Maximum Downswing Distribution for {strategy_name}')
    ax.set_xlabel('Maximum Downswing (EUR)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_time_underwater_comparison(all_results, config, color_map=None, pdf=None):
    """
    Creates a bar chart comparing the median percentage of time each strategy
    spends 'underwater' (below a previous bankroll peak).
    """
    strategy_names = list(all_results.keys())
    underwater_pcts = [res.get('median_time_underwater_pct', 0) for res in all_results.values()]

    if color_map is None:
        # Fallback for generating colors internally if no map is provided
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    # Get colors in the correct order for the plot
    plot_colors = [color_map.get(name) for name in strategy_names]

    if not strategy_names:
        return plt.figure()

    # Dynamic height to ensure bars are not too squished with many strategies
    num_strategies = len(strategy_names)
    fig_height = max(4, 2.0 + num_strategies * 0.7) # Base height + per-strategy height, with a minimum
    fig, ax = plt.subplots(figsize=(8, fig_height))

    # Horizontal bar chart for better readability of strategy names
    bars = ax.barh(strategy_names, underwater_pcts, color=plot_colors)

    ax.set_xlabel('Median Time Spent "Underwater" (%)', fontsize=12)
    ax.set_title('Psychological Cost: Time Spent Below Bankroll Peak', fontsize=16)
    ax.invert_yaxis()  # Puts the first strategy at the top
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add percentage labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')

    ax.set_xlim(right=max(underwater_pcts) * 1.15 if underwater_pcts else 100)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_total_withdrawn_comparison(all_results, config, color_map=None, pdf=None):
    """
    Creates a bar chart comparing the median total amount withdrawn for each strategy.
    Returns None if withdrawals were not enabled, so the UI can skip rendering it.
    """
    # Only show this plot if withdrawals were enabled for the simulation run.
    if not config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
        return None

    strategy_names = list(all_results.keys())
    withdrawn_amounts = [res.get('median_total_withdrawn', 0) for res in all_results.values()]

    if color_map is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    plot_colors = [color_map.get(name) for name in strategy_names]

    if not strategy_names:
        return plt.figure()

    # Dynamic height to ensure bars are not too squished with many strategies
    num_strategies = len(strategy_names)
    # Use the same dynamic height formula for consistency with the other h-bar chart
    fig_height = max(4, 2.0 + num_strategies * 0.7)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    bars = ax.barh(strategy_names, withdrawn_amounts, color=plot_colors)

    ax.set_xlabel('Median Total Withdrawn (€)', fontsize=12)
    ax.set_title('Income Generation: Median Total Withdrawn', fontsize=16)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, f'€{width:,.0f}', va='center', ha='left')

    ax.set_xlim(right=max(withdrawn_amounts) * 1.2 if withdrawn_amounts and max(withdrawn_amounts) > 0 else 100)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_downswing_comparison(all_results, config, color_map=None, pdf=None):
    """
    Creates a bar chart comparing the 95th percentile maximum downswing for each strategy.
    """
    strategy_names = list(all_results.keys())
    # Use the P95 max downswing as the key risk metric for comparison
    downswing_values = [res.get('p95_max_downswing', 0) for res in all_results.values()]

    if color_map is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    plot_colors = [color_map.get(name) for name in strategy_names]

    if not strategy_names:
        return plt.figure()

    # Dynamic height to ensure bars are not too squished with many strategies
    num_strategies = len(strategy_names)
    fig_height = max(4, 2.0 + num_strategies * 0.7)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    bars = ax.barh(strategy_names, downswing_values, color=plot_colors)

    ax.set_xlabel('95th Percentile Maximum Downswing (€)', fontsize=12)
    ax.set_title('Volatility Comparison: Maximum Downswing Risk', fontsize=16)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, f'€{width:,.0f}', va='center', ha='left')

    if downswing_values and max(downswing_values) > 0:
        ax.set_xlim(right=max(downswing_values) * 1.2)

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

def plot_risk_reward_scatter(all_results, config, color_map=None, pdf=None):
    """
    Creates a scatter plot to visualize the risk vs. reward trade-off for each strategy.
    """
    strategy_names = list(all_results.keys())

    # Define the metrics for the axes
    risk_metric_key = 'risk_of_ruin'
    reward_metric_key = 'median_final_bankroll'

    # --- Data Extraction and Cleaning ---
    # Extract data and filter out any non-finite values (inf, -inf, nan) to prevent plotting errors.
    # This is a critical robustness improvement.
    clean_data = []
    for name in strategy_names:
        res = all_results[name]
        risk = res.get(risk_metric_key)
        reward = res.get(reward_metric_key)
        if risk is not None and reward is not None and np.isfinite(risk) and np.isfinite(reward):
            clean_data.append({'name': name, 'risk': risk, 'reward': reward})

    if len(clean_data) < 2:
        return plt.figure() # Don't plot if there's not enough valid data to compare

    # Unpack the cleaned data for plotting
    plot_names = [d['name'] for d in clean_data]
    risk_values = [d['risk'] for d in clean_data]
    reward_values = [d['reward'] for d in clean_data]
    risk_label = 'Risk of Ruin (%)'
    reward_label = f"Median Final Bankroll (€)"

    # Use a fixed figure size for the scatter plot for stability.
    fig, ax = plt.subplots(figsize=(8, 6))

    if color_map is None:
        # Fallback for generating colors internally if no map is provided
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    # Get colors for the valid, plotted strategies
    plot_colors = [color_map.get(name) for name in plot_names]

    ax.scatter(risk_values, reward_values, s=150, c=plot_colors, alpha=0.7, edgecolors='w', zorder=10)

    # Annotate each point with the strategy name using a robust offset
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.02  # A small 2% offset based on the y-axis range
    for i, name in enumerate(plot_names):
        ax.text(risk_values[i], reward_values[i] + y_offset, name, fontsize=9, ha='center')

    # Add interpretation quadrants based on the average of the valid data
    if risk_values and reward_values:
        avg_risk = np.mean(risk_values)
        avg_reward = np.mean(reward_values)
        ax.axvline(avg_risk, color='gray', linestyle='--', linewidth=0.8)
        ax.axhline(avg_reward, color='gray', linestyle='--', linewidth=0.8)

    ax.set_xlabel(risk_label, fontsize=12)
    ax.set_ylabel(reward_label, fontsize=12)
    ax.set_title('Risk vs. Reward Analysis', fontsize=16)
    ax.grid(True, linestyle=':', alpha=0.5)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_hands_distribution_table(result, strategy_name, pdf=None):
    """
    Creates a table summarizing the hands distribution and win rates per stake
    and saves it to the PDF.
    """
    hands_dist = result.get('hands_distribution_pct')
    avg_wr = result.get('avg_assigned_wr_per_stake')
    trust = result.get('trust_per_stake')

    if not hands_dist or not avg_wr or not trust:
        return None # Not enough data to plot

    # Sort stakes by their 'bb_size' from the config for a logical order
    stakes_data = result.get('stakes_data_for_report', [])
    stake_order = {s['name']: s['bb_size'] for s in stakes_data}
    stake_names_sorted = sorted(hands_dist.keys(), key=lambda s: stake_order.get(s, float('inf')))

    # Filter out potential NaN/None stakes from empty config rows and ensure they were played.
    stake_names = [s for s in stake_names_sorted if pd.notna(s) and hands_dist.get(s, 0) > 0]
    
    if not stake_names:
        return None

    # Prepare data for the table
    cell_text = []
    for stake in stake_names:
        row = [
            f"{hands_dist.get(stake, 0):.1f}%",
            f"{avg_wr.get(stake, 0):.2f}",
            f"{trust.get(stake, 0):.1f}%"
        ]
        cell_text.append(row)

    columns = ['% of Hands', 'Avg. Assigned WR', 'Input Trust']
    rows = [str(s) for s in stake_names] # Ensure all labels are strings

    fig, ax = plt.subplots(figsize=(8, max(1.5, len(rows) * 0.5))) # Dynamic height
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5) # Adjust column width and row height

    ax.set_title(f'Hands Distribution & Trust for {strategy_name}', pad=20, fontsize=14)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_percentile_wr_analysis_table(result, strategy_name, pdf=None):
    """
    Creates a table showing the win rate breakdown for key percentile runs
    and saves it to the PDF.
    """
    percentile_wrs = result.get('percentile_win_rates')
    if not percentile_wrs:
        return None

    # Define the order of rows (percentiles)
    row_order = [p for p in ['p5', 'p25', 'p50_median', 'p75', 'p95'] if p in percentile_wrs]
    if not row_order:
        return None

    # Prepare data for the table
    cell_text = []
    for percentile_key in row_order:
        data = percentile_wrs[percentile_key]
        row = [
            f"{data.get('assigned_wr', 0):.2f}",
            f"{data.get('play_wr', 0):.2f}",
            f"{data.get('rakeback_wr', 0):.2f}",
            f"{data.get('variance_impact', 0):.2f}"
        ]
        cell_text.append(row)

    columns = ['Assigned WR', 'Play WR', 'Rakeback WR', 'Variance Impact']
    row_labels = {
        'p5': '5th Percentile', 'p25': '25th Percentile', 'p50_median': 'Median Run',
        'p75': '75th Percentile', 'p95': '95th Percentile'
    }
    rows = [row_labels.get(key, key) for key in row_order]

    fig, ax = plt.subplots(figsize=(8, max(1.5, len(rows) * 0.5))) # Dynamic height
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title(f'Percentile Win Rate Analysis (bb/100) for {strategy_name}', pad=20, fontsize=14)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def add_text_page(text, pdf, title=""):
    """
    Creates a matplotlib figure with wrapped text and adds it to the PDF.
    Useful for adding analysis summaries or other text blocks.
    """
    fig = plt.figure(figsize=(8.27, 11.69)) # A4 size in inches
    fig.clf()
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)

    # Place text with wrapping
    fig.text(0.05, 0.9, text, va='top', ha='left', wrap=True, fontsize=10)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_summary_table(all_results, strategy_page_map, config, pdf=None):
    """
    Creates a comprehensive summary table as an image, mirroring the UI's detail,
    and saves it to the PDF.
    """
    header = [
        'Strategy', 'Page', 'Median Final BR', 'Mode Final BR', 'Median Growth', 'Median Hands',
        'Median Rakeback', 'RoR (%)', 'Target Prob (%)', '5th %ile BR', 'P95 Downswing'
    ]
    cell_text = []

    for strategy_name, res in all_results.items():
        row = [
            strategy_name,
            str(strategy_page_map.get(strategy_name, '-')),
            f"€{res.get('median_final_bankroll', 0):,.0f}",
            f"€{res.get('final_bankroll_mode', 0):,.0f}",
            f"{res.get('growth_rate', 0):.2%}",
            f"{res.get('median_hands_played', 0):,.0f}",
            f"€{res.get('median_rakeback_eur', 0):,.2f}",
            f"{res.get('risk_of_ruin', 0):.2f}%",
            f"{res.get('target_prob', 0):.2f}%",
            f"€{res.get('p5', 0):,.0f}",
            f"€{res.get('p95_max_downswing', 0):,.0f}"
        ]
        cell_text.append(row)

    if not cell_text:
        return None

    fig, ax = plt.subplots(figsize=(11.69, 8.27)) # A4 landscape
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=header, cellLoc='center', loc='center')
    table.automake_col_widths()
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    ax.set_title('Simulation Summary', pad=20, fontsize=16)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_risk_of_demotion_table(result, strategy_name, pdf=None):
    """
    Creates a table showing the risk of demotion from each stake level.
    """
    risk_data = result.get('risk_of_demotion')
    if not risk_data:
        return None

    # Sort stakes by their 'bb_size' from the config for a logical order
    stakes_data = result.get('stakes_data_for_report', [])
    stake_order = {s['name']: s['bb_size'] for s in stakes_data}
    stake_names_sorted = sorted(risk_data.keys(), key=lambda s: stake_order.get(s, float('inf')))

    # Filter out potential NaN/None stakes that can result from empty rows in config
    stake_names = [s for s in stake_names_sorted if pd.notna(s)]
    
    if not stake_names:
        return None

    cell_text = []
    for stake in stake_names:
        data = risk_data[stake]
        row = [
            f"{int(data.get('reached_count', 0))}",
            f"{data.get('prob', 0):.1f}%"
        ]
        cell_text.append(row)

    columns = ['Sims Reaching Stake', 'Demotion Probability']
    rows = [str(s) for s in stake_names] # Ensure all labels are strings

    fig, ax = plt.subplots(figsize=(8, max(1.5, len(rows) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title(f'Risk of Demotion Analysis for {strategy_name}', pad=20, fontsize=14)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_sanity_check_analysis(result, config, pdf=None):
    """
    Creates a text-based plot comparing analytical expectations vs. simulation results
    for the single-stake sanity check.
    """
    start_br = config['STARTING_BANKROLL_EUR']
    total_hands = config['TOTAL_HANDS_TO_SIMULATE']
    stake_data = config['STAKES_DATA'][0]
    ev_wr = stake_data['ev_bb_per_100']
    std_dev = stake_data['std_dev_per_100']
    bb_size = stake_data['bb_size']

    # Analytical calculations
    expected_profit = (total_hands / 100) * ev_wr * bb_size
    expected_final_br = start_br + expected_profit
    expected_std_dev_eur = (std_dev / np.sqrt(100)) * np.sqrt(total_hands) * bb_size

    # Actual results from simulation
    actual_median_br = result['median_final_bankroll']
    actual_std_dev_br = np.std(result['final_bankrolls'])

    # Calculate percentage differences
    median_diff_pct = ((actual_median_br - expected_final_br) / expected_final_br) * 100 if expected_final_br != 0 else 0
    std_dev_diff_pct = ((actual_std_dev_br - expected_std_dev_eur) / expected_std_dev_eur) * 100 if expected_std_dev_eur != 0 else 0

    text_content = (
        "This report compares the simulation's output against known mathematical formulas for a simple, single-stake scenario.\n"
        "The 'Actual' values from the simulation should be very close to the 'Expected' values calculated analytically.\n\n"
        "--- Expected (Analytical) ---\n"
        f"Expected Final Median Bankroll: €{expected_final_br:,.2f}\n"
        f"Expected Std. Dev. of Final Bankroll: €{expected_std_dev_eur:,.2f}\n\n"
        "--- Actual (from Simulation) ---\n"
        f"Actual Final Median Bankroll: €{actual_median_br:,.2f}\n"
        f"Actual Std. Dev. of Final Bankroll: €{actual_std_dev_br:,.2f}\n\n"
        "--- Comparison ---\n"
        f"Difference in Median: {median_diff_pct:+.2f}%\n"
        f"Difference in Std. Dev.: {std_dev_diff_pct:+.2f}%"
    )

    # Use the generic text page function to render this content
    return add_text_page(text_content, pdf, title="Sanity Check Analysis")

def plot_downswing_analysis_tables(result, strategy_name, pdf=None):
    """
    Creates two table images for downswing depth and duration probabilities,
    and adds them to the PDF.
    """
    downswing_analysis = result.get('downswing_analysis', {})
    depth_data = downswing_analysis.get('depth_probabilities', {})
    duration_data = downswing_analysis.get('duration_probabilities', {})

    # Don't create plots if there's no data
    if not depth_data and not duration_data:
        return

    # Create a figure that can hold two tables side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 5)) # A4 landscape half-page
    fig.suptitle(f'Downswing Probabilities for {strategy_name}', fontsize=16, y=0.98)

    for ax, data, title, cols in [
        (axes[0], depth_data, 'Depth (BB)', ['Depth', 'Probability']),
        (axes[1], duration_data, 'Duration (Hands)', ['Duration', 'Probability'])
    ]:
        ax.axis('tight')
        ax.axis('off')
        ax.set_title(title, pad=20, fontsize=12)

        if data:
            cell_data = [[f"{k:,}+", f"{v:.2f}%"] for k, v in sorted(data.items())]
            table = ax.table(cellText=cell_data, colLabels=cols, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def get_initial_table_mix_string(strategy, config):
    """Helper function to describe the starting table mix as a string."""
    rule = strategy.get_table_mix(config['STARTING_BANKROLL_EUR'])
    if not rule:
        return "No Play"
    # Sort by the string representation of the key to prevent TypeErrors
    # if a strategy returns mixed-type keys (e.g., strings and numbers).
    sorted_items = sorted(rule.items(), key=lambda item: str(item[0]))
    mix_parts = [f"{stake}: {value}" for stake, value in sorted_items]
    return ", ".join(mix_parts) if mix_parts else "No Play"

def write_analysis_report_to_pdf(pdf, analysis_report):
    """Writes the qualitative analysis report to a PDF page."""
    fig = plt.figure(figsize=(11, 8.5))

    lines = analysis_report.split('\n')
    y_pos = 0.90

    for line in lines:
        line = line.strip()
        if not line: # Add vertical space for empty lines
            y_pos -= 0.02
            continue

        font_size = 11
        font_weight = 'normal'
        x_pos = 0.05

        # Simple markdown parsing for PDF
        if line.startswith('### '):
            font_size = 16
            font_weight = 'bold'
            line = line[4:]
            y_pos -= 0.02 # Extra space before header
        elif line.startswith('- '):
            x_pos = 0.07
            line = f"• {line[2:]}" # Use a bullet point character

        # Strip emoji and bold markers for cleaner text
        line = line.replace('**', '').replace('🏆','').replace('📉','').replace('🛡️','').replace('🎲','').replace('🚀','').replace('😌','').replace('🎢','').replace('💰','').replace('⚠️','').replace('🧠','').replace('💸','').replace('⚡','')

        fig.text(x_pos, y_pos, line, transform=fig.transFigure, size=font_size, weight=font_weight, va='top', ha='left', wrap=True)
        y_pos -= 0.04 # Fixed spacing between lines

    pdf.savefig(fig)
    plt.close(fig)

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

    def _write_downswing_analysis(res):
        lines = []
        if 'median_max_downswing' in res:
            lines.extend(["", "--- Maximum Downswing Analysis ---"])
            lines.append("A downswing is the largest single peak-to-trough drop in bankroll during a simulation.")
            median_downswing = res['median_max_downswing']
            p95_downswing = res['p95_max_downswing']
            median_integrated_drawdown = res.get('median_integrated_drawdown', 0)
            lines.append(f"   - Median Downswing Depth: €{median_downswing:,.2f}")
            lines.append(f"   - 95th Percentile Downswing Depth: €{p95_downswing:,.2f} (5% of runs had a worse downswing)")
            if median_integrated_drawdown > 0:
                lines.append(f"   - Median Integrated Drawdown: {median_integrated_drawdown:,.0f} Euro-Hands")
                lines.append("     (This measures the total 'pain' by combining the size and duration of all time spent underwater.)")
        return lines

    def _write_income_analysis(res):
        lines = []
        # Only add this section if withdrawals were enabled for the run
        if config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
            lines.extend(["", "--- Income & Total Return Analysis ---"])
            lines.append("Metrics related to money taken out of the bankroll and overall value generated.")
            median_withdrawn = res.get('median_total_withdrawn', 0.0)
            p95_withdrawn = res.get('p95_total_withdrawn', 0.0)
            median_return = res.get('median_total_return', 0.0)
            lines.append(f"   - Median Total Withdrawn: €{median_withdrawn:,.2f}")
            lines.append(f"   - 95th Percentile Withdrawn: €{p95_withdrawn:,.2f} (5% of runs withdrew more than this)")
            lines.append(f"   - Median Total Return: €{median_return:,.2f} ((Final BR - Start BR) + Withdrawn)")
        return lines

    def _write_final_stake_distribution(res):
        lines = []
        if 'final_stake_distribution' in res and res['final_stake_distribution']:
            lines.extend(["", "--- Final Stake Distribution ---", "Percentage of simulations ending at each stake/table mix:"])
            # Sort by percentage (value) descending, then by mix string (key) ascending as a tie-breaker.
            sorted_dist = sorted(res['final_stake_distribution'].items(), key=lambda item: (-item[1], str(item[0])))
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
    report_lines.extend(_write_downswing_analysis(result))
    report_lines.extend(_write_income_analysis(result))
    report_lines.extend(_write_final_bankroll_metrics(result))
    report_lines.extend(_write_final_stake_distribution(result))
    report_lines.extend(_write_final_highest_stake_distribution(result))
    if 'percentile_win_rates' in result and result['percentile_win_rates']:
        report_lines.extend(_write_win_rate_analysis(result))
    return report_lines

def write_strategy_report_to_pdf(pdf, report_lines, start_page_num=None):
    """Writes the detailed text report for a single strategy to a PDF page, handling pagination."""
    lines_per_page = 45
    pages_of_lines = [report_lines[i:i + lines_per_page] for i in range(0, len(report_lines), lines_per_page)]

    for i, page_lines in enumerate(pages_of_lines):
        report_text = "\n".join(page_lines)
        fig = plt.figure(figsize=(11, 8.5))
        if start_page_num is not None:
            fig.text(0.95, 0.05, f"Page {start_page_num + i}", transform=fig.transFigure, size=8, va='bottom', ha='right', color='gray')
        fig.text(0.05, 0.95, report_text, transform=fig.transFigure, size=10, va='top', ha='left', fontfamily='monospace')
        pdf.savefig(fig)
        plt.close(fig)

def save_summary_table_to_pdf(pdf, all_results, strategy_page_map, config):
    """Creates a table of the main summary results and saves it to a PDF page."""
    header = ['Strategy', 'Page', 'Median Final BR', 'Med Withdrawn', 'Med Total Return', 'Median Growth', 'Median Rakeback', 'RoR (%)', 'Target Prob (%)', '5th %ile']

    # Conditionally add the stop-loss header
    if config.get("STOP_LOSS_BB", 0) > 0:
        header.insert(7, 'Median SL') # Insert after Median Rakeback

    cell_text = []
    for strategy_name, result in all_results.items():
        row = [
            strategy_name,
            str(strategy_page_map.get(strategy_name, '-')),
            f"€{result['median_final_bankroll']:,.2f}",
            f"€{result.get('median_total_withdrawn', 0.0):,.2f}",
            f"€{result.get('median_total_return', 0.0):,.2f}",
            f"{result['growth_rate']:.2%}",
            f"€{result.get('median_rakeback_eur', 0.0):,.2f}",
            f"{result['risk_of_ruin']:.2f}",
            f"{result['target_prob']:.2f}",
            f"€{result['p5']:,.2f}"
        ]

        # Conditionally add the stop-loss value to the row
        if config.get("STOP_LOSS_BB", 0) > 0:
            row.insert(7, f"{result.get('median_stop_losses', 0):.1f}")

        cell_text.append(row)

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

def _get_plot_callables(all_results, result, strategy_name, config, color_map, weighted_input_wr, pdf):
    """Helper to map plot names to their actual plotting function calls with correct arguments."""
    
    # Note: The keys here must match the strings in the plot lists in generate_pdf_report
    comparison_plot_map = {
        'Median Progression': lambda: plot_median_progression_comparison(all_results, config, color_map=color_map, pdf=pdf),
        'Final Bankroll Distribution': lambda: plot_final_bankroll_comparison(all_results, config, color_map=color_map, pdf=pdf),
        'Total Withdrawn': lambda: plot_total_withdrawn_comparison(all_results, config, color_map=color_map, pdf=pdf),
        'Time Underwater': lambda: plot_time_underwater_comparison(all_results, config, color_map=color_map, pdf=pdf),
        'Risk vs Reward': lambda: plot_risk_reward_scatter(all_results, config, color_map=color_map, pdf=pdf),
    }

    strategy_plot_map = {
        'Strategy Progression': lambda: plot_strategy_progression(result.get('bankroll_histories'), result.get('hands_histories'), strategy_name, config, pdf=pdf),
        'Final Bankroll Distribution': lambda: plot_final_bankroll_distribution(result.get('final_bankrolls'), result, strategy_name, config, pdf=pdf, color_map=color_map),
        'Assigned WR Distribution': lambda: plot_assigned_wr_distribution(result.get('avg_assigned_wr_per_sim'), result.get('median_run_assigned_wr'), weighted_input_wr, strategy_name, pdf=pdf),
        'Max Downswing Distribution': lambda: plot_max_downswing_distribution(result.get('max_downswings'), result, strategy_name, pdf=pdf, color_map=color_map),
        'Downswing Probabilities': lambda: plot_downswing_analysis_tables(result, strategy_name, pdf=pdf),
    }
    return comparison_plot_map, strategy_plot_map

def generate_pdf_report(all_results, analysis_report, config, timestamp_str):
    """
    Generates the entire multi-page PDF report, writing to an in-memory buffer.
    """
    pdf_buffer = io.BytesIO()
    strategy_page_map = {}

    # --- Define the plots to be generated to avoid "magic numbers" for page counting. ---
    # These lists explicitly define which plots are included in the PDF report.
    # If a plot is added or removed, only these lists need to be updated.
    comparison_plots = [
        'Median Progression', 'Final Bankroll Distribution', 'Time Underwater', 'Risk vs Reward'
    ]
    if config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
        comparison_plots.append('Total Withdrawn')
    num_comparison_plots = len(comparison_plots)

    # Plots generated for each individual strategy's detailed report section.
    strategy_detail_plots = [
        'Strategy Progression', 'Final Bankroll Distribution', 'Assigned WR Distribution',
        'Max Downswing Distribution', 'Downswing Probabilities'
    ]
    num_plot_pages_per_strategy = len(strategy_detail_plots)

    # --- Calculate the starting page number for the detailed reports section. ---
    # This calculation is now dynamic based on the defined plot lists.
    # 1 (Title Page)
    # + 1 (Summary Table Page)
    # + num_comparison_plots (for the comparison charts)
    # + (1 if analysis_report else 0) (for the qualitative analysis page)
    page_counter_for_map = 2 + num_comparison_plots + (1 if analysis_report else 0)
    lines_per_page = 45

    for strategy_name, result in all_results.items():
        strategy_config = config['STRATEGIES_TO_RUN'][strategy_name]
        strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA']) # noqa
        # The page this strategy's report starts on
        strategy_page_map[strategy_name] = page_counter_for_map + 1

        report_lines = get_strategy_report_lines(strategy_name, result, strategy_obj, config)
        num_text_pages = (len(report_lines) + lines_per_page - 1) // lines_per_page # Ceiling division
        page_counter_for_map += num_text_pages + num_plot_pages_per_strategy

    # Calculate a representative input win rate for the plot's label
    total_sample_hands = sum(s['sample_hands'] for s in config['STAKES_DATA'])
    if total_sample_hands > 0:
        weighted_input_wr = sum(s['ev_bb_per_100'] * s['sample_hands'] for s in config['STAKES_DATA']) / total_sample_hands
    else:
        # Fallback if no sample hands are provided
        weighted_input_wr = config['STAKES_DATA'][0]['ev_bb_per_100'] if config['STAKES_DATA'] else 1.5

    # Create the color map once for the entire PDF generation process
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    with PdfPages(pdf_buffer) as pdf:
        create_title_page(pdf, timestamp_str)
        save_summary_table_to_pdf(pdf, all_results, strategy_page_map, config)

        if analysis_report:
            write_analysis_report_to_pdf(pdf, analysis_report)

        # --- Generate Comparison Plots Dynamically ---
        # This loop is driven by the `comparison_plots` list defined above.
        comparison_plot_calls, _ = _get_plot_callables(all_results, None, None, config, color_map, None, pdf)
        for plot_name in comparison_plots:
            if plot_name in comparison_plot_calls:
                comparison_plot_calls[plot_name]()

        for strategy_name, result in all_results.items():
            strategy_config = config['STRATEGIES_TO_RUN'][strategy_name]
            strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA'])
            page_num = strategy_page_map.get(strategy_name, 0)
            report_lines_for_writing = get_strategy_report_lines(strategy_name, result, strategy_obj, config)
            write_strategy_report_to_pdf(pdf, report_lines_for_writing, start_page_num=page_num)
            
            # --- Generate Strategy-Specific Plots Dynamically ---
            _, strategy_plot_calls = _get_plot_callables(all_results, result, strategy_name, config, color_map, weighted_input_wr, pdf)
            for plot_name in strategy_detail_plots:
                if plot_name in strategy_plot_calls:
                    strategy_plot_calls[plot_name]()

    pdf_buffer.seek(0)
    return pdf_buffer
# =================================================================================
#   QUALITATIVE ANALYSIS FUNCTIONS
# =================================================================================

def generate_qualitative_analysis(all_results, config):
    """Generates a human-readable analysis comparing the performance of different strategies."""
    insights = []
    num_strategies = len(all_results)

    if num_strategies < 2:
        insights.append("### Overall Summary")
        insights.append("Run at least two strategies to generate a comparative analysis.")
        return "\n".join(insights)

    # --- New Metric Calculation: Efficiency Score ---
    for name, res in all_results.items():
        # Calculate a risk-adjusted return score. Higher is better.
        # Use Median Total Return if withdrawals are on, otherwise use simple growth.
        if config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
            total_value_generated = res.get('median_total_return', 0)
        else:
            total_value_generated = res['median_final_bankroll'] - config['STARTING_BANKROLL_EUR']

        pain = res['p95_max_downswing']
        if pain > 1: # Avoid division by zero or tiny numbers
            # We add 1 to value to avoid issues with 0 and to slightly penalize negative values.
            res['efficiency_score'] = (total_value_generated + 1) / pain
        else:
            # If there's no significant downswing, it's extremely efficient. Give it a very high score, proportional to value.
            res['efficiency_score'] = total_value_generated if total_value_generated > 0 else 1.0

    def find_best_worst_with_ties(metric_key, higher_is_better=True):
        """Helper to find the best and worst performing strategies, handling ties for 'best'."""
        valid_results = {name: res for name, res in all_results.items() if metric_key in res}
        if not valid_results: return [], None

        sorted_strategies = sorted(valid_results.items(), key=lambda item: item[1][metric_key], reverse=higher_is_better)

        best_value = sorted_strategies[0][1][metric_key]
        best_strategies = [name for name, res in sorted_strategies if res[metric_key] == best_value]

        worst_name = sorted_strategies[-1][0]

        # If all strategies have the same value, there is no "worst"
        if best_value == sorted_strategies[-1][1][metric_key]:
            return best_strategies, None

        return best_strategies, worst_name

    insights.append("### Automated Strategy Analysis")
    insights.append("This analysis compares your strategies based on key performance indicators from the simulation.")

    best_medians, worst_median = find_best_worst_with_ties('median_final_bankroll', higher_is_better=True)
    if best_medians:
        names = f"'{best_medians[0]}'" if len(best_medians) == 1 else f"'{', '.join(best_medians)}'"
        verb = "achieved" if len(best_medians) == 1 else "were tied for"
        insights.append(f"\n**🏆 Best Typical Outcome:** The **{names}** strategy {verb} the highest median final bankroll (€{all_results[best_medians[0]]['median_final_bankroll']:,.0f}). This suggests it provides the most consistent growth for the average simulation run.")
    if worst_median and worst_median not in best_medians:
         insights.append(f"\n**📉 Worst Typical Outcome:** The **'{worst_median}'** strategy had the lowest median result (€{all_results[worst_median]['median_final_bankroll']:,.0f}). Check its Risk of Ruin and Downswing metrics to understand why.")

    best_rors, worst_ror = find_best_worst_with_ties('risk_of_ruin', higher_is_better=False)
    if best_rors:
        names = f"'{best_rors[0]}'" if len(best_rors) == 1 else f"'{', '.join(best_rors)}'"
        verb = "was" if len(best_rors) == 1 else "were tied for"
        insights.append(f"\n**🛡️ Safest Strategy:** With a Risk of Ruin of only {all_results[best_rors[0]]['risk_of_ruin']:.2f}%, **{names}** {verb} the least likely to go broke. This is ideal for risk-averse players.")
    if worst_ror and worst_ror not in best_rors:
        insights.append(f"\n**🎲 Riskiest Strategy:** **'{worst_ror}'** had the highest Risk of Ruin at {all_results[worst_ror]['risk_of_ruin']:.2f}%. This strategy is significantly more volatile.")

    best_targets, _ = find_best_worst_with_ties('target_prob', higher_is_better=True)
    if best_targets:
        names = f"'{best_targets[0]}'" if len(best_targets) == 1 else f"'{', '.join(best_targets)}'"
        verb = "gave" if len(best_targets) == 1 else "were tied for giving"
        insights.append(f"\n**🚀 Highest Upside:** If your main goal is to reach the target bankroll, the **{names}** strategy {verb} the best chance at {all_results[best_targets[0]]['target_prob']:.2f}%. This often comes with higher risk, so check its RoR.")

    best_downswings, worst_downswing = find_best_worst_with_ties('median_max_downswing', higher_is_better=False)
    if best_downswings:
        names = f"'{best_downswings[0]}'" if len(best_downswings) == 1 else f"'{', '.join(best_downswings)}'"
        verb = "had" if len(best_downswings) == 1 else "were tied for having"
        best_strat_name = best_downswings[0]
        underwater_pct = all_results[best_strat_name]['median_time_underwater_pct']
        insights.append(f"\n**😌 Smoothest Ride:** The **{names}** strategy {verb} the smallest median downswing (€{all_results[best_strat_name]['median_max_downswing']:,.0f}). It also spent only {underwater_pct:.0f}% of the time 'underwater', making it the least stressful to play.")
    if worst_downswing and worst_downswing not in best_downswings:
        underwater_pct = all_results[worst_downswing]['median_time_underwater_pct']
        insights.append(f"\n**🎢 Rollercoaster Ride:** Be prepared for significant swings with the **'{worst_downswing}'** strategy, which had the largest median downswing of €{all_results[worst_downswing]['median_max_downswing']:,.0f} and spent {underwater_pct:.0f}% of the time 'underwater'.")

    best_pain, _ = find_best_worst_with_ties('median_integrated_drawdown', higher_is_better=False)
    if best_pain:
        names = f"'{best_pain[0]}'" if len(best_pain) == 1 else f"'{', '.join(best_pain)}'"
        verb = "offered" if len(best_pain) == 1 else "were tied for offering"
        insights.append(f"\n**🧠 Least Painful Journey:** The **{names}** strategy {verb} the most psychologically comfortable ride. It had the lowest 'Integrated Drawdown', meaning it minimized both the size and duration of downswings, reducing overall financial stress.")

    best_efficiency, _ = find_best_worst_with_ties('efficiency_score', higher_is_better=True)
    if best_efficiency:
        names = f"'{best_efficiency[0]}'" if len(best_efficiency) == 1 else f"'{', '.join(best_efficiency)}'"
        verb = "demonstrated" if len(best_efficiency) == 1 else "were tied for demonstrating"
        insights.append(f"\n**⚡ Most Efficient:** The **{names}** strategy {verb} the best risk-adjusted return. It generated the most 'total value' (growth + withdrawals) for the amount of 'pain' (downswing) it caused, making it a highly efficient choice.")

    # Add insight for withdrawals if enabled
    if config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
        best_income, worst_income = find_best_worst_with_ties('median_total_withdrawn', higher_is_better=True)
        if best_income:
            names = f"'{best_income[0]}'" if len(best_income) == 1 else f"'{', '.join(best_income)}'"
            verb = "generated" if len(best_income) == 1 else "were tied for generating"
            insights.append(f"\n**💸 Best Income Generator:** The **{names}** strategy {verb} the most income through withdrawals, with a median of €{all_results[best_income[0]]['median_total_withdrawn']:,.0f}. This is a key metric for players who need to live off their winnings.")
        if worst_income and worst_income not in best_income and all_results[worst_income]['median_total_withdrawn'] < all_results[best_income[0]]['median_total_withdrawn']:
            insights.append(f"\n**🏦 Lowest Income Generator:** **'{worst_income}'** generated the least income (€{all_results[worst_income]['median_total_withdrawn']:,.0f}). This might be because it was too conservative to generate profits to withdraw, or too risky, leading to frequent downswings that prevented withdrawals.")

    # Add insight for highest rakeback earner
    if config.get("RAKEBACK_PERCENTAGE", 0) > 0:
        best_rb_strats, _ = find_best_worst_with_ties('median_rakeback_eur', higher_is_better=True)
        if best_rb_strats:
            names = f"'{best_rb_strats[0]}'" if len(best_rb_strats) == 1 else f"'{', '.join(best_rb_strats)}'"
            verb = "generated" if len(best_rb_strats) == 1 else "were tied for generating"
            insights.append(f"\n**💰 Highest Rakeback Earner:** The **{names}** strategy {verb} the most rakeback (€{all_results[best_rb_strats[0]]['median_rakeback_eur']:,.0f}), often by playing more hands at higher stakes.")

    # Add insight for stop-loss triggers
    if config.get("STOP_LOSS_BB", 0) > 0:
        most_sl_strats, _ = find_best_worst_with_ties('median_stop_losses', higher_is_better=True)
        if most_sl_strats and all_results[most_sl_strats[0]]['median_stop_losses'] > 0:
            names = f"'{most_sl_strats[0]}'" if len(most_sl_strats) == 1 else f"'{', '.join(most_sl_strats)}'"
            verb = "triggered" if len(most_sl_strats) == 1 else "were tied for triggering"
            median_sl_val = all_results[most_sl_strats[0]]['median_stop_losses']
            insights.append(f"\n**⚠️ Session Volatility:** The **{names}** strategy {verb} the stop-loss most often (median of {median_sl_val:.1f} times). This indicates it was more prone to large, single-session losses.")

    insights.append("\n### Why Did They Perform This Way?")

    if worst_ror and best_targets and worst_ror in best_targets:
        insights.append(f"- The **'{worst_ror}'** strategy is a classic high-risk, high-reward approach. It achieved the highest probability of reaching the target, but also came with the highest Risk of Ruin. This is a trade-off between upside potential and safety.")   

    # Analyze the contribution of rakeback to the best strategy's success
    if config.get("RAKEBACK_PERCENTAGE", 0) > 0 and best_medians:
        best_median_strat_name = best_medians[0]
        best_median_res = all_results[best_median_strat_name]
        median_profit = best_median_res['median_final_bankroll'] - config['STARTING_BANKROLL_EUR']
        median_rakeback = best_median_res['median_rakeback_eur']
        if median_profit > 0 and median_rakeback > 0:
            rb_contribution = (median_rakeback / median_profit) * 100
            if rb_contribution > 30:
                insights.append(f"- Rakeback was a critical factor for the top-performing **'{best_median_strat_name}'** strategy, accounting for **{rb_contribution:.0f}%** of its median profit. This strategy's success may be highly dependent on the rakeback deal.")
    elif config.get("RAKEBACK_PERCENTAGE", 0) == 0:
        insights.append("- With **0% rakeback**, all strategies are handicapped. This significantly reduces profitability and increases risk, especially for aggressive strategies that rely on moving up to high-rake environments.")

    if worst_median:
        worst_median_res = all_results[worst_median]
        hands_dist = worst_median_res.get('hands_distribution_pct', {})
        if hands_dist:
            stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
            played_stakes = [s for s, p in hands_dist.items() if p > 5]
            if played_stakes:
                highest_played_stake = max(played_stakes, key=lambda s: stake_order_map.get(s, -1))
                demotion_risks = worst_median_res.get('risk_of_demotion', {})
                if highest_played_stake in demotion_risks and demotion_risks[highest_played_stake]['prob'] > 40:
                    insights.append(f"- The **'{worst_median}'** strategy likely underperformed due to instability. It had a high **Risk of Demotion of {demotion_risks[highest_played_stake]['prob']:.1f}%** from {highest_played_stake}. This 'yo-yo effect' of moving up and down frequently is inefficient and can hurt long-term growth.")

    hysteresis_strats = [name for name in all_results if 'Hysteresis' in name or 'Sticky' in name]
    if hysteresis_strats:
        h_strat_name = hysteresis_strats[0]
        demotion_risks = all_results[h_strat_name].get('risk_of_demotion', {})
        if demotion_risks:
            avg_demotion_risk = np.mean([v['prob'] for v in demotion_risks.values()]) if demotion_risks else 100
            if avg_demotion_risk < 20:
                insights.append(f"- The **'{h_strat_name}'** strategy likely performed well by minimizing demotions. Its 'sticky' nature prevented players from dropping stakes during minor downswings, leading to more stable growth at higher limits.")

    # Insight for overly conservative strategies
    stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
    conservative_strats = [name for name, res in all_results.items() if 'Conservative' in name]
    if conservative_strats:
        c_strat_name = conservative_strats[0]
        c_res = all_results[c_strat_name]
        if c_res['target_prob'] < np.mean([res['target_prob'] for res in all_results.values()]) - 10: # If target prob is >10% below average
            hands_dist = c_res.get('hands_distribution_pct', {})
            if hands_dist:
                lowest_stake_played = min(hands_dist.keys(), key=lambda s: stake_order_map.get(s, float('inf'))) if hands_dist else None
                if lowest_stake_played:
                    percent_at_lowest = hands_dist.get(lowest_stake_played, 0)
                    if percent_at_lowest > 80: # If it spends >80% of time at the lowest stake
                        display_percent = "over 99" if percent_at_lowest > 99 else f"over {percent_at_lowest:.0f}"
                        insights.append(f"- The **'{c_strat_name}'** strategy may be too conservative. It spent {display_percent}% of its time at {lowest_stake_played}, which significantly limited its growth and ability to reach the target.")

    # --- New Section: Actionable Recommendations ---
    insights.append("\n### Actionable Recommendations")
    recommendations_made = False

    # 1. Identify Dominated Strategies
    dominated_strategies = {}
    strategy_names = list(all_results.keys())
    for i in range(len(strategy_names)):
        for j in range(len(strategy_names)):
            if i == j: continue
            strat_a_name, strat_b_name = strategy_names[i], strategy_names[j]
            res_a, res_b = all_results[strat_a_name], all_results[strat_b_name]

            # Strategy B dominates A if it's better/equal on all key metrics, and strictly better on at least one.
            b_is_better = (res_b['median_final_bankroll'] >= res_a['median_final_bankroll'] and
                           res_b['risk_of_ruin'] <= res_a['risk_of_ruin'] and
                           res_b['p95_max_downswing'] <= res_a['p95_max_downswing'])
            b_is_strictly_better = (res_b['median_final_bankroll'] > res_a['median_final_bankroll'] or
                                    res_b['risk_of_ruin'] < res_a['risk_of_ruin'] or
                                    res_b['p95_max_downswing'] < res_a['p95_max_downswing'])
            if b_is_better and b_is_strictly_better:
                dominated_strategies.setdefault(strat_a_name, []).append(strat_b_name)

    if dominated_strategies:
        insights.append("\n**Consider Retiring Dominated Strategies:**")
        for dominated, dominators in dominated_strategies.items():
            dominator_str = f"'{dominators[0]}'" if len(dominators) == 1 else f"'{', '.join(dominators)}'"
            insights.append(f"- The **'{dominated}'** strategy appears to be inefficient. It is outperformed by **{dominator_str}**, which offer(s) a better combination of reward and safety. There is likely no reason to choose '{dominated}'.")
        recommendations_made = True

    # 2. Provide Prescriptive Advice
    advice_given = set() # To avoid giving the same advice twice
    if worst_ror and worst_ror not in best_rors and 'risky' not in advice_given:
        insights.append(f"\n**Tuning the Riskiest Strategy:**\n- To make the **'{worst_ror}'** strategy safer, consider increasing its bankroll thresholds. This will make it move up stakes more slowly, reducing its high Risk of Ruin ({all_results[worst_ror]['risk_of_ruin']:.2f}%).")
        advice_given.add('risky'); recommendations_made = True

    if conservative_strats and 'conservative' not in advice_given:
        c_strat_name = conservative_strats[0]
        if all_results[c_strat_name]['target_prob'] < np.mean([res['target_prob'] for res in all_results.values()]) - 10:
            insights.append(f"\n**Tuning the Most Conservative Strategy:**\n- The **'{c_strat_name}'** strategy may be too conservative and limiting your growth. Consider lowering its bankroll thresholds to allow it to move up to more profitable stakes sooner.")
            advice_given.add('conservative'); recommendations_made = True

    if not recommendations_made:
        insights.append("\nAll strategies appear to be well-balanced with clear trade-offs. Your choice depends on your personal risk tolerance.")

    return "\n".join(insights)

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

def run_full_analysis(config, progress_callback=None):
    """
    The main entry point for running the entire simulation analysis.
    This function is called by the Streamlit app.
    It now accepts an optional progress_callback function to report progress to the UI.
    """
    all_results = {}
    stake_level_map = {stake['name']: i for i, stake in enumerate(sorted(config['STAKES_DATA'], key=lambda s: s['bb_size']))}
    stake_name_map = {v: k for k, v in stake_level_map.items()}

    num_strategies = len(config['STRATEGIES_TO_RUN'])
    if num_strategies == 0:
        raise ValueError("No strategies are defined. Please add at least one strategy.")

    # Create a master RNG to generate unique, deterministic seeds for each strategy.
    # This ensures each strategy gets its own "deck of cards" while the overall
    # simulation remains reproducible from the main seed.
    master_rng = np.random.default_rng(config['SEED'])

    for i, (strategy_name, strategy_config) in enumerate(config['STRATEGIES_TO_RUN'].items()):
        if progress_callback:
            progress = i / num_strategies
            progress_callback(progress, f"Simulating strategy: {strategy_name} ({i+1}/{num_strategies})...")

        strategy_seed = master_rng.integers(1, 1_000_000_000)
        all_win_rates, rng = setup_simulation_parameters(config, strategy_seed)
        strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA'])

        # Determine which simulation function to use based on the class type
        if isinstance(strategy_obj, HysteresisStrategy):
            bankroll_histories, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns, stop_loss_triggers, underwater_hands_count, integrated_drawdown, total_withdrawn_histories, downswing_depth_exceeded, downswing_duration_exceeded = run_sticky_simulation_vectorized(strategy_obj, all_win_rates, rng, stake_level_map, config)
        else:
            bankroll_histories, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns, stop_loss_triggers, underwater_hands_count, integrated_drawdown, total_withdrawn_histories, downswing_depth_exceeded, downswing_duration_exceeded = run_multiple_simulations_vectorized(strategy_obj, all_win_rates, rng, stake_level_map, config)

        # Analyze the results and store them
        all_results[strategy_name] = analyze_strategy_results(
            strategy_name=strategy_name,
            strategy_obj=strategy_obj,
            bankroll_histories=bankroll_histories,
            hands_per_stake_histories=hands_per_stake_histories,
            rakeback_histories=rakeback_histories,
            all_win_rates=all_win_rates,
            rng=rng,
            peak_stake_levels=peak_stake_levels, demotion_flags=demotion_flags, stake_level_map=stake_level_map, stake_name_map=stake_name_map,
            max_drawdowns=max_drawdowns, stop_loss_triggers=stop_loss_triggers, underwater_hands_count=underwater_hands_count,
            integrated_drawdown=integrated_drawdown, total_withdrawn_histories=total_withdrawn_histories,
            downswing_depth_exceeded=downswing_depth_exceeded, downswing_duration_exceeded=downswing_duration_exceeded,
            config=config
        )

    # Generate the final qualitative analysis report
    if progress_callback:
        progress_callback(0.95, "Generating qualitative analysis...")
    analysis_report = generate_qualitative_analysis(all_results, config)

    if progress_callback:
        progress_callback(1.0, "Finalizing report...")

    return {
        "results": all_results,
        "analysis_report": analysis_report
    }