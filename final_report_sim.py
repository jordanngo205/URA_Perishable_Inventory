#!/usr/bin/env python3

from __future__ import annotations

"""Standalone simulation for the final perishable-inventory report.

This script is the implementation behind the report tables. It simulates a
two-node system with:

- perishable inventory aged as new -> medium -> old -> expired
- regular warehouse replenishment
- optional B -> A transshipment
- optional emergency warehouse orders for A
- periodic and continuous review policies
- two cost regimes for underage vs. overage incentives

Running the file prints the summary tables used in the report.
"""

import copy
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

# Simulation horizon and surge settings.
TOTAL_PERIODS = 15
PERIODS_PER_CYCLE = 3
RNG_SEED = 42
SURGE_START_PERIOD = 7
SURGE_DURATION_PERIODS = 1
SURGE_MULTIPLIER = {"A": 2.0, "B": 1.0}

# Continuous-review tuning used in the final report.
CONTINUOUS_ORDER_COVERAGE = {"A": 0.4, "B": 1.8}
CONTINUOUS_RESERVE_B = 5

# Shared response and ordering costs.
TRANSHIP_UNIT_COST = 1.2
NEW_TRANSHIP_SURCHARGE = 2.0
EMERGENCY_ORDER_UNIT_COST = 6.0
FIXED_ORDER_COST = 35.0
TRANSHIP_CAP_PER_PERIOD = 70
TRANSHIP_LEAD_TIME = 1
EMERGENCY_ORDER_LEAD_TIME = 1

# Base facility settings before cost-regime-specific tuning is applied.
BASE_FACILITIES = {
    "A": {
        "review_cycle_periods": 4,
        "lead_time": 2,
        "reorder_point": 95,
        "order_up_to": 140,
        "initial_on_hand": 100,
        "reserve_for_transship": 35,
    },
    "B": {
        "review_cycle_periods": 3,
        "lead_time": 3,
        "reorder_point": 80,
        "order_up_to": 210,
        "initial_on_hand": 210,
        "reserve_for_transship": 35,
    },
}

DEMAND_MEAN = {"A": 43, "B": 35}
DEMAND_STD = {"A": 4, "B": 3}

# The two cost structures compared in the report.
COST_REGIMES = {
    "underage_gt_overage": {
        "label": "Underage > Overage",
        "short_label": "Underage-dominant",
        "lost_profit": 12.0,
        "holding_costs": {"A": 0.35, "B": 0.30},
        "expiration_costs": {"A": 2.5, "B": 2.5},
    },
    "overage_gt_underage": {
        "label": "Overage > Underage",
        "short_label": "Overage-dominant",
        "lost_profit": 2.0,
        "holding_costs": {"A": 1.00, "B": 0.95},
        "expiration_costs": {"A": 4.0, "B": 4.0},
    },
}


def format_float(value: float) -> str:
    """Format report-facing numbers with two decimal places."""
    return f"{value:.2f}"


def build_demand_path(with_surge: bool) -> pd.DataFrame:
    """Create the period-by-period demand stream used by the simulator.

    A fixed random seed makes every policy face the same demand path. When
    `with_surge` is true, A's demand is multiplied during the surge window.
    """
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    surge_end_period = SURGE_START_PERIOD + SURGE_DURATION_PERIODS - 1

    for period in range(1, TOTAL_PERIODS + 1):
        surge_active = with_surge and SURGE_START_PERIOD <= period <= surge_end_period
        demand_a = max(0, int(round(rng.normal(DEMAND_MEAN["A"], DEMAND_STD["A"]))))
        demand_b = max(0, int(round(rng.normal(DEMAND_MEAN["B"], DEMAND_STD["B"]))))

        if surge_active:
            demand_a = int(round(demand_a * SURGE_MULTIPLIER["A"]))
            demand_b = int(round(demand_b * SURGE_MULTIPLIER["B"]))

        rows.append(
            {
                "period": period,
                "cycle": (period - 1) // PERIODS_PER_CYCLE + 1,
                "period_in_cycle": (period - 1) % PERIODS_PER_CYCLE + 1,
                "demand_A": demand_a,
                "demand_B": demand_b,
                "surge_active": surge_active,
            }
        )

    return pd.DataFrame(rows)


def prepare_facilities(
    base_facilities: dict,
    holding_costs: dict[str, float],
    expiration_costs: dict[str, float],
) -> dict:
    """Copy base facility settings and attach regime-specific cost inputs."""
    facilities = copy.deepcopy(base_facilities)
    for facility in facilities:
        facilities[facility]["holding_cost"] = float(holding_costs[facility])
        facilities[facility]["expiration_cost"] = float(expiration_costs[facility])
    return facilities


def derive_safety_stock_metrics(
    facilities: dict,
    demand_std: dict[str, float],
    lost_profit: float,
    review_mode: str,
) -> pd.DataFrame:
    """Compute the newsvendor-style service targets used in the report.

    The report uses these metrics to derive safety stock under each regime and
    each review mode. The values are returned as a DataFrame so they can be
    reused directly in tables and in policy configuration.
    """
    rows = []
    for facility, config in facilities.items():
        protection_period = (
            int(config["lead_time"])
            if review_mode == "continuous"
            else int(config["review_cycle_periods"]) + int(config["lead_time"])
        )
        # Underage is lost profit over the protection period; overage is a
        # proxy for holding inventory through its usable life and possibly
        # expiring it.
        cu = lost_profit * protection_period
        co = 3.0 * float(config["holding_cost"]) + float(config["expiration_cost"])
        raw_alpha = cu / (cu + co)
        capped_alpha = min(max(raw_alpha, 0.50), 0.999)
        z_value = NormalDist().inv_cdf(capped_alpha)
        safety_stock = int(math.ceil(max(0.0, z_value * demand_std[facility] * math.sqrt(protection_period))))
        rows.append(
            {
                "facility": facility,
                "review_mode": review_mode,
                "protection_period": protection_period,
                "Cu": cu,
                "Co": co,
                "alpha_raw": raw_alpha,
                "alpha_used": capped_alpha,
                "z_value": z_value,
                "safety_stock": safety_stock,
            }
        )
    return pd.DataFrame(rows)


def configure_periodic_benchmark(base_facilities: dict, periodic_metrics: pd.DataFrame) -> dict:
    """Attach periodic-review safety stock to the benchmark facility settings."""
    facilities = copy.deepcopy(base_facilities)
    ss_map = periodic_metrics.set_index("facility")["safety_stock"].to_dict()
    for facility in facilities:
        facilities[facility]["safety_stock"] = int(ss_map[facility])
    return facilities


def configure_continuous_policy(
    base_facilities: dict,
    continuous_metrics: pd.DataFrame,
    demand_mean: dict[str, float],
    order_coverage: dict[str, float],
    reserve_b: int,
) -> dict:
    """Build the continuous-review policy used in the final experiments.

    The continuous policy recalculates reorder points and order-up-to levels
    from demand means, lead times, safety stock, and the extra coverage
    parameters chosen for A and B.
    """
    facilities = copy.deepcopy(base_facilities)
    ss_map = continuous_metrics.set_index("facility")["safety_stock"].to_dict()

    for facility in facilities:
        lead = int(facilities[facility]["lead_time"])
        safety_stock = int(ss_map[facility])
        mean_demand = float(demand_mean[facility])
        reorder_point = int(math.ceil(mean_demand * lead + safety_stock))
        order_up_to = int(math.ceil(mean_demand * (lead + order_coverage[facility]) + safety_stock))

        facilities[facility]["safety_stock"] = safety_stock
        facilities[facility]["reorder_point"] = reorder_point
        facilities[facility]["order_up_to"] = order_up_to
        facilities[facility]["initial_on_hand"] = order_up_to

    facilities["B"]["reserve_for_transship"] = reserve_b
    return facilities


def run_simulation(
    facilities: dict,
    demand_df: pd.DataFrame,
    lost_profit: float,
    review_mode: str,
    response_policy: str,
) -> pd.DataFrame:
    """Run one full simulation scenario and return a period-level audit table.

    Parameters
    ----------
    facilities:
        Facility settings for the chosen review policy and cost regime.
    demand_df:
        Demand path produced by `build_demand_path`.
    lost_profit:
        Backlog penalty per unit per period for the active regime.
    review_mode:
        Either "periodic" or "continuous".
    response_policy:
        One of "warehouse", "transship", or "emergency".

    Returns
    -------
    pd.DataFrame
        One row per period with operating state, response decisions, and cost
        breakdowns. This table is the main audit trail behind the report.
    """
    # State carried from one period to the next. "carry_old" and "carry_med"
    # represent usable inventory surviving into the next period after aging.
    state = {
        "A": {
            "carry_old": 0,
            "carry_med": 0,
            "backlog": 0,
            "order_pipeline": [],
            "transship_pipeline": [],
            "emergency_pipeline": [],
        },
        "B": {
            "carry_old": 0,
            "carry_med": 0,
            "backlog": 0,
            "order_pipeline": [],
            "transship_pipeline": [],
            "emergency_pipeline": [],
        },
    }

    def receive_order_arrivals(facility: str, period: int) -> int:
        """Pull regular warehouse orders that arrive this period."""
        arrived = sum(q for (arrival_period, q) in state[facility]["order_pipeline"] if arrival_period == period)
        state[facility]["order_pipeline"] = [
            (arrival_period, q)
            for (arrival_period, q) in state[facility]["order_pipeline"]
            if arrival_period != period
        ]
        return arrived

    def receive_emergency_arrivals(facility: str, period: int) -> int:
        """Pull emergency warehouse orders that arrive this period."""
        arrived = sum(q for (arrival_period, q) in state[facility]["emergency_pipeline"] if arrival_period == period)
        state[facility]["emergency_pipeline"] = [
            (arrival_period, q)
            for (arrival_period, q) in state[facility]["emergency_pipeline"]
            if arrival_period != period
        ]
        return arrived

    def receive_transship_arrivals(facility: str, period: int) -> tuple[int, int]:
        """Pull inbound transshipments that arrive this period.

        Medium units sent from B arrive to A as old units one period later.
        New units sent from B arrive to A as medium units one period later.
        """
        arrived_old = sum(old_q for (arrival_period, old_q, med_q) in state[facility]["transship_pipeline"] if arrival_period == period)
        arrived_med = sum(med_q for (arrival_period, old_q, med_q) in state[facility]["transship_pipeline"] if arrival_period == period)
        state[facility]["transship_pipeline"] = [
            (arrival_period, old_q, med_q)
            for (arrival_period, old_q, med_q) in state[facility]["transship_pipeline"]
            if arrival_period != period
        ]
        return arrived_old, arrived_med

    def inventory_position(facility: str) -> int:
        """Compute the inventory position used by the reorder logic."""
        on_order = sum(q for (_, q) in state[facility]["order_pipeline"])
        inbound_transship = sum(old_q + med_q for (_, old_q, med_q) in state[facility]["transship_pipeline"])
        inbound_emergency = sum(q for (_, q) in state[facility]["emergency_pipeline"])
        on_hand_usable = state[facility]["carry_old"] + state[facility]["carry_med"]
        return on_hand_usable + on_order + inbound_transship + inbound_emergency - state[facility]["backlog"]

    def estimate_wait_if_no_response(receiver: str, period: int) -> int:
        """Estimate how long A would wait if no new emergency action is taken.

        This drives the economic trigger for transshipment and emergency
        ordering. The model first looks for useful arrivals already in the
        pipeline. If none exist, it falls back to the policy's implied wait.
        """
        future_arrivals = [arrival_period for (arrival_period, _) in state[receiver]["order_pipeline"] if arrival_period > period]
        future_arrivals.extend(
            arrival_period
            for (arrival_period, old_q, med_q) in state[receiver]["transship_pipeline"]
            if arrival_period > period and (old_q + med_q) > 0
        )
        future_arrivals.extend(
            arrival_period
            for (arrival_period, q) in state[receiver]["emergency_pipeline"]
            if arrival_period > period and q > 0
        )
        if future_arrivals:
            return max(1, min(future_arrivals) - period)

        if review_mode == "continuous":
            return int(facilities[receiver]["lead_time"])

        review_cycle = int(facilities[receiver]["review_cycle_periods"])
        periods_to_next_review = (review_cycle - (period % review_cycle)) % review_cycle
        return periods_to_next_review + int(facilities[receiver]["lead_time"])

    def donor_excess(med_left: int, new_left: int, donor_facility: str) -> int:
        """Amount of donor inventory that can be released without violating B's protection rules."""
        keep = int(facilities[donor_facility]["reserve_for_transship"]) + int(facilities[donor_facility]["safety_stock"])
        return max(0, (med_left + new_left) - keep)

    records = []

    for row in demand_df.itertuples(index=False):
        # Read this period's demand and the inventory/backlog carried into it.
        period = int(row.period)
        demand = {"A": int(row.demand_A), "B": int(row.demand_B)}
        surge_active = bool(row.surge_active)

        backlog_start = {facility: state[facility]["backlog"] for facility in ("A", "B")}
        start_old = {facility: state[facility]["carry_old"] for facility in ("A", "B")}
        start_med = {facility: state[facility]["carry_med"] for facility in ("A", "B")}

        order_arrivals = {facility: receive_order_arrivals(facility, period) for facility in ("A", "B")}
        emergency_arrivals = {facility: receive_emergency_arrivals(facility, period) for facility in ("A", "B")}
        transship_arrivals = {facility: receive_transship_arrivals(facility, period) for facility in ("A", "B")}
        initial_batch = {facility: int(facilities[facility]["initial_on_hand"]) if period == 1 else 0 for facility in ("A", "B")}

        # Inbound transshipment ages on arrival: sent-medium becomes old, and
        # sent-new becomes medium.
        for facility in ("A", "B"):
            start_old[facility] += transship_arrivals[facility][0]
            start_med[facility] += transship_arrivals[facility][1]

        start_new = {
            facility: order_arrivals[facility] + emergency_arrivals[facility] + initial_batch[facility]
            for facility in ("A", "B")
        }

        # FEFO service: use old first, then medium, then new.
        total_demand = {facility: backlog_start[facility] + demand[facility] for facility in ("A", "B")}
        use_old = {facility: min(start_old[facility], total_demand[facility]) for facility in ("A", "B")}
        rem_after_old = {facility: total_demand[facility] - use_old[facility] for facility in ("A", "B")}
        use_med = {facility: min(start_med[facility], rem_after_old[facility]) for facility in ("A", "B")}
        rem_after_med = {facility: rem_after_old[facility] - use_med[facility] for facility in ("A", "B")}
        use_new = {facility: min(start_new[facility], rem_after_med[facility]) for facility in ("A", "B")}

        served_total = {facility: use_old[facility] + use_med[facility] + use_new[facility] for facility in ("A", "B")}
        served_backlog = {facility: min(backlog_start[facility], served_total[facility]) for facility in ("A", "B")}
        served_current = {facility: served_total[facility] - served_backlog[facility] for facility in ("A", "B")}
        new_shortage = {facility: max(0, demand[facility] - served_current[facility]) for facility in ("A", "B")}
        backlog_end_pre_response = {facility: max(0, total_demand[facility] - served_total[facility]) for facility in ("A", "B")}

        old_left = {facility: start_old[facility] - use_old[facility] for facility in ("A", "B")}
        med_left = {facility: start_med[facility] - use_med[facility] for facility in ("A", "B")}
        new_left = {facility: start_new[facility] - use_new[facility] for facility in ("A", "B")}

        # Economic triggers compare response cost to the cost of making A wait.
        est_wait_a = estimate_wait_if_no_response("A", period)
        penalty_signal_a = lost_profit * est_wait_a
        base_transship_economic = TRANSHIP_UNIT_COST < penalty_signal_a
        new_transship_economic = (TRANSHIP_UNIT_COST + NEW_TRANSHIP_SURCHARGE) < penalty_signal_a
        emergency_economic = EMERGENCY_ORDER_UNIT_COST < penalty_signal_a

        ship_med_b_to_a = 0
        ship_new_b_to_a = 0
        emergency_order_a = 0

        # If transshipment is the chosen policy, send medium first and only use
        # fresh units if they are also economical.
        if response_policy == "transship" and backlog_end_pre_response["A"] > 0 and base_transship_economic:
            excess = donor_excess(med_left["B"], new_left["B"], "B")
            remaining_cap = TRANSHIP_CAP_PER_PERIOD

            ship_med_b_to_a = min(backlog_end_pre_response["A"], med_left["B"], excess, remaining_cap)
            med_left["B"] -= ship_med_b_to_a
            excess -= ship_med_b_to_a
            remaining_cap -= ship_med_b_to_a

            if new_transship_economic:
                ship_new_b_to_a = min(backlog_end_pre_response["A"] - ship_med_b_to_a, new_left["B"], excess, remaining_cap)
                new_left["B"] -= ship_new_b_to_a

            if ship_med_b_to_a > 0 or ship_new_b_to_a > 0:
                state["A"]["transship_pipeline"].append((period + TRANSHIP_LEAD_TIME, ship_med_b_to_a, ship_new_b_to_a))

        # Emergency ordering is available only to A and only in the emergency
        # scenario. It covers the full pre-response backlog when it is economical.
        if response_policy == "emergency" and backlog_end_pre_response["A"] > 0 and emergency_economic:
            emergency_order_a = backlog_end_pre_response["A"]
            state["A"]["emergency_pipeline"].append((period + EMERGENCY_ORDER_LEAD_TIME, emergency_order_a))

        post_response_old = {"A": old_left["A"], "B": old_left["B"]}
        post_response_med = {"A": med_left["A"], "B": med_left["B"]}
        post_response_new = {"A": new_left["A"], "B": new_left["B"]}

        # End-of-period aging: leftover old expires, medium becomes old, and
        # new becomes medium for the next period.
        expired = {"A": post_response_old["A"], "B": post_response_old["B"]}
        next_start_old = {"A": post_response_med["A"], "B": post_response_med["B"]}
        next_start_med = {"A": post_response_new["A"], "B": post_response_new["B"]}
        end_on_hand = {facility: next_start_old[facility] + next_start_med[facility] for facility in ("A", "B")}
        backlog_end = backlog_end_pre_response.copy()

        for facility in ("A", "B"):
            state[facility]["carry_old"] = next_start_old[facility]
            state[facility]["carry_med"] = next_start_med[facility]
            state[facility]["backlog"] = backlog_end[facility]

        orders = {"A": 0, "B": 0}
        order_arrival_period = {"A": math.nan, "B": math.nan}
        for facility in ("A", "B"):
            # Under continuous review, every period is a review opportunity.
            # Under periodic review, the node can only order on its review dates.
            review_hit = True if review_mode == "continuous" else period % int(facilities[facility]["review_cycle_periods"]) == 0
            if not review_hit:
                continue

            current_ip = inventory_position(facility)
            trigger_level = int(facilities[facility]["reorder_point"])
            target_level = int(facilities[facility]["order_up_to"])

            if review_mode == "periodic":
                trigger_level += int(facilities[facility]["safety_stock"])
                target_level += int(facilities[facility]["safety_stock"])

            if current_ip <= trigger_level:
                quantity = max(0, int(round(target_level - current_ip)))
                if quantity > 0:
                    arrival_period = period + int(facilities[facility]["lead_time"])
                    state[facility]["order_pipeline"].append((arrival_period, quantity))
                    orders[facility] = quantity
                    order_arrival_period[facility] = arrival_period

        # Cost accounting is done after all operating decisions for the period.
        total_shortage = new_shortage["A"] + new_shortage["B"]
        total_backlog_end = backlog_end["A"] + backlog_end["B"]
        total_transship = ship_med_b_to_a + ship_new_b_to_a
        wait_cost_A = lost_profit * backlog_end["A"]
        wait_cost_B = lost_profit * backlog_end["B"]
        wait_cost = wait_cost_A + wait_cost_B

        base_transship_cost_B = TRANSHIP_UNIT_COST * total_transship
        fresh_receipt_surcharge_A = NEW_TRANSHIP_SURCHARGE * ship_new_b_to_a
        emergency_cost_A = EMERGENCY_ORDER_UNIT_COST * emergency_order_a
        order_cost_A = FIXED_ORDER_COST if orders["A"] > 0 else 0.0
        order_cost_B = FIXED_ORDER_COST if orders["B"] > 0 else 0.0
        holding_cost_A = float(facilities["A"]["holding_cost"]) * end_on_hand["A"]
        holding_cost_B = float(facilities["B"]["holding_cost"]) * end_on_hand["B"]
        expire_cost_A = float(facilities["A"]["expiration_cost"]) * expired["A"]
        expire_cost_B = float(facilities["B"]["expiration_cost"]) * expired["B"]

        node_total_cost_A = wait_cost_A + fresh_receipt_surcharge_A + emergency_cost_A + order_cost_A + holding_cost_A + expire_cost_A
        node_total_cost_B = wait_cost_B + base_transship_cost_B + order_cost_B + holding_cost_B + expire_cost_B
        total_cost = node_total_cost_A + node_total_cost_B

        records.append(
            {
                "period": period,
                "cycle": int(row.cycle),
                "period_in_cycle": int(row.period_in_cycle),
                "surge_active": surge_active,
                "response_policy": response_policy,
                "demand_A": demand["A"],
                "demand_B": demand["B"],
                "backlog_start_A": backlog_start["A"],
                "backlog_start_B": backlog_start["B"],
                "start_old_A": start_old["A"],
                "start_med_A": start_med["A"],
                "start_new_A": start_new["A"],
                "start_old_B": start_old["B"],
                "start_med_B": start_med["B"],
                "start_new_B": start_new["B"],
                "served_old_A": use_old["A"],
                "served_med_A": use_med["A"],
                "served_new_A": use_new["A"],
                "served_old_B": use_old["B"],
                "served_med_B": use_med["B"],
                "served_new_B": use_new["B"],
                "shortage_A": new_shortage["A"],
                "shortage_B": new_shortage["B"],
                "total_shortage": total_shortage,
                "backlog_end_A": backlog_end["A"],
                "backlog_end_B": backlog_end["B"],
                "total_backlog_end": total_backlog_end,
                "transship_med_B_to_A": ship_med_b_to_a,
                "transship_new_B_to_A": ship_new_b_to_a,
                "total_transship": total_transship,
                "emergency_order_A": emergency_order_a,
                "emergency_arrival_A": emergency_arrivals["A"],
                "post_response_old_A": post_response_old["A"],
                "post_response_med_A": post_response_med["A"],
                "post_response_new_A": post_response_new["A"],
                "post_response_old_B": post_response_old["B"],
                "post_response_med_B": post_response_med["B"],
                "post_response_new_B": post_response_new["B"],
                "expired_A": expired["A"],
                "expired_B": expired["B"],
                "next_start_old_A": next_start_old["A"],
                "next_start_med_A": next_start_med["A"],
                "next_start_old_B": next_start_old["B"],
                "next_start_med_B": next_start_med["B"],
                "end_on_hand_A": end_on_hand["A"],
                "end_on_hand_B": end_on_hand["B"],
                "order_A": orders["A"],
                "order_B": orders["B"],
                "order_A_arrival_period": order_arrival_period["A"],
                "order_B_arrival_period": order_arrival_period["B"],
                "wait_cost_A": wait_cost_A,
                "wait_cost_B": wait_cost_B,
                "wait_cost": wait_cost,
                "base_transship_cost_B": base_transship_cost_B,
                "fresh_receipt_surcharge_A": fresh_receipt_surcharge_A,
                "emergency_cost_A": emergency_cost_A,
                "order_cost_A": order_cost_A,
                "order_cost_B": order_cost_B,
                "holding_cost_A": holding_cost_A,
                "holding_cost_B": holding_cost_B,
                "expire_cost_A": expire_cost_A,
                "expire_cost_B": expire_cost_B,
                "node_total_cost_A": node_total_cost_A,
                "node_total_cost_B": node_total_cost_B,
                "total_cost": total_cost,
                "penalty_signal_A": penalty_signal_a,
                "base_transship_economic": base_transship_economic,
                "new_transship_economic": new_transship_economic,
                "emergency_economic": emergency_economic,
                "estimated_wait_A": est_wait_a,
            }
        )

    return pd.DataFrame(records)


def summarize_run(df: pd.DataFrame, regime_label: str, scenario_label: str) -> dict:
    """Collapse a period-level audit table into one scenario summary row."""
    return {
        "Regime": regime_label,
        "Response": scenario_label,
        "Shortage": int(df["total_shortage"].sum()),
        "End backlog": int(df["total_backlog_end"].sum()),
        "Units transshipped": int(df["total_transship"].sum()),
        "Emergency units": int(df["emergency_order_A"].sum()),
        "Lost-profit cost": float(df["wait_cost"].sum()),
        "Transship cost": float(df["base_transship_cost_B"].sum() + df["fresh_receipt_surcharge_A"].sum()),
        "Emergency cost": float(df["emergency_cost_A"].sum()),
        "Order cost": float(df["order_cost_A"].sum() + df["order_cost_B"].sum()),
        "Holding cost": float(df["holding_cost_A"].sum() + df["holding_cost_B"].sum()),
        "Expiry cost": float(df["expire_cost_A"].sum() + df["expire_cost_B"].sum()),
        "Node A total cost": float(df["node_total_cost_A"].sum()),
        "Node B total cost": float(df["node_total_cost_B"].sum()),
        "Total cost": float(df["total_cost"].sum()),
    }


def build_report_context() -> dict:
    """Run every regime/scenario combination needed by the report.

    The returned dictionary contains:
    - raw period-level DataFrames
    - safety-stock metrics
    - baseline summaries
    - surge-response summaries
    """
    baseline_demand = build_demand_path(with_surge=False)
    surge_demand = build_demand_path(with_surge=True)

    context = {
        "settings": {
            "transship_unit_cost": TRANSHIP_UNIT_COST,
            "new_transship_surcharge": NEW_TRANSHIP_SURCHARGE,
            "emergency_order_unit_cost": EMERGENCY_ORDER_UNIT_COST,
            "fixed_order_cost": FIXED_ORDER_COST,
            "transship_cap": TRANSHIP_CAP_PER_PERIOD,
            "transship_lead_time": TRANSHIP_LEAD_TIME,
            "emergency_order_lead_time": EMERGENCY_ORDER_LEAD_TIME,
        },
        "regimes": {},
    }

    for regime_key, regime in COST_REGIMES.items():
        regime_facilities = prepare_facilities(BASE_FACILITIES, regime["holding_costs"], regime["expiration_costs"])
        periodic_metrics = derive_safety_stock_metrics(regime_facilities, DEMAND_STD, regime["lost_profit"], review_mode="periodic")
        continuous_metrics = derive_safety_stock_metrics(regime_facilities, DEMAND_STD, regime["lost_profit"], review_mode="continuous")
        periodic_facilities = configure_periodic_benchmark(regime_facilities, periodic_metrics)
        continuous_facilities = configure_continuous_policy(
            regime_facilities,
            continuous_metrics,
            DEMAND_MEAN,
            CONTINUOUS_ORDER_COVERAGE,
            CONTINUOUS_RESERVE_B,
        )

        periodic_baseline_df = run_simulation(periodic_facilities, baseline_demand, regime["lost_profit"], "periodic", "warehouse")
        continuous_baseline_df = run_simulation(continuous_facilities, baseline_demand, regime["lost_profit"], "continuous", "warehouse")
        surge_warehouse_df = run_simulation(continuous_facilities, surge_demand, regime["lost_profit"], "continuous", "warehouse")
        surge_transship_df = run_simulation(continuous_facilities, surge_demand, regime["lost_profit"], "continuous", "transship")
        surge_emergency_df = run_simulation(continuous_facilities, surge_demand, regime["lost_profit"], "continuous", "emergency")

        context["regimes"][regime_key] = {
            "label": regime["label"],
            "lost_profit": regime["lost_profit"],
            "periodic_metrics": periodic_metrics,
            "continuous_metrics": continuous_metrics,
            "periodic_baseline_df": periodic_baseline_df,
            "continuous_baseline_df": continuous_baseline_df,
            "surge_warehouse_df": surge_warehouse_df,
            "surge_transship_df": surge_transship_df,
            "surge_emergency_df": surge_emergency_df,
            "baseline_summary": [
                summarize_run(periodic_baseline_df, regime["label"], "Periodic baseline"),
                summarize_run(continuous_baseline_df, regime["label"], "Continuous baseline"),
            ],
            "surge_summary": [
                summarize_run(surge_warehouse_df, regime["label"], "Warehouse only"),
                summarize_run(surge_transship_df, regime["label"], "Use node B"),
                summarize_run(surge_emergency_df, regime["label"], "Emergency order"),
            ],
        }

    return context


def print_report_tables(context: dict) -> None:
    """Print the compact summary tables shown at the command line."""
    baseline_rows = []
    surge_rows = []

    for regime in context["regimes"].values():
        periodic_row, continuous_row = regime["baseline_summary"]
        baseline_rows.append(
            {
                "Regime": regime["label"],
                "Periodic shortage": periodic_row["Shortage"],
                "Periodic backlog": periodic_row["End backlog"],
                "Periodic total cost": format_float(periodic_row["Total cost"]),
                "Continuous shortage": continuous_row["Shortage"],
                "Continuous backlog": continuous_row["End backlog"],
                "Continuous total cost": format_float(continuous_row["Total cost"]),
            }
        )
        for row in regime["surge_summary"]:
            surge_rows.append(
                {
                    "Regime": row["Regime"],
                    "Response": row["Response"],
                    "Shortage": row["Shortage"],
                    "End backlog": row["End backlog"],
                    "Units transshipped": row["Units transshipped"],
                    "Emergency units": row["Emergency units"],
                    "Lost-profit cost": format_float(row["Lost-profit cost"]),
                    "Transship cost": format_float(row["Transship cost"]),
                    "Emergency cost": format_float(row["Emergency cost"]),
                    "Total cost": format_float(row["Total cost"]),
                }
            )

    print("\nBaseline comparison\n")
    print(pd.DataFrame(baseline_rows).to_string(index=False))
    print("\nSurge response comparison\n")
    print(pd.DataFrame(surge_rows).to_string(index=False))


def main() -> None:
    """Entry point for running the full simulation from the command line."""
    context = build_report_context()
    print_report_tables(context)


if __name__ == "__main__":
    main()
