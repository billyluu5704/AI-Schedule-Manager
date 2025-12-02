"""
Experiments 2 & 3: Team-level scheduling with your fuzzy availability system.

This script:
  - Generates synthetic staff and task sets for 3 regimes:
        "under" (60% load), "borderline" (100%), "overloaded" (140%)
  - Builds schedules using:
        (1) fuzzy availability (your system)
        (2) capacity-based heuristic
        (3) random assignment
  - Computes metrics:
        late_rate, avg_lateness, workload_std, avg_utilization,
        gini_load, overloaded_staff_count, schedule_cost
  - Outputs plots:
        * boxplot of schedule_cost for all regimes/methods
        * fairness (Gini) vs regime
        * late_rate vs regime

Run from repo root:
    python -m experiments.fuzzy_scheduling_experiments
"""

import os
import math
import random
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pipeline.fuzzy_logic_for_availability_check import (
    parse_payload,
    fuzzy_system_availability,
)
_AVAIL_SIM = fuzzy_system_availability()


# ---------------------------------------------------------------------
# 1. Data structures
# ---------------------------------------------------------------------


@dataclass
class Staff:
    id: int
    capacity_hours_per_day: float
    current_hours: float = 0.0
    # list of deadlines (in "days from now" units) for projects already assigned
    project_deadlines_days: List[float] = None

    def __post_init__(self):
        if self.project_deadlines_days is None:
            self.project_deadlines_days = []


@dataclass
class Task:
    id: int
    hours: float
    deadline_day: float  # relative to "now", in days
    priority: int        # 1=low, 2=med, 3=high


@dataclass
class ScheduleResult:
    finish_day: float
    staff_id: int


# ---------------------------------------------------------------------
# 2. Wrapper for your fuzzy availability
# ---------------------------------------------------------------------


def compute_availability_from_deadlines(
    project_deadlines_days: List[float],
    new_task_deadline_day: float,
) -> float:
    """
    Use your fuzzy availability system based on deadlines only,
    reusing a single ControlSystemSimulation for speed.
    """

    now = dt.datetime.now()

    def day_to_str(days: float) -> str:
        return (now + dt.timedelta(days=float(days))).strftime("%Y-%m-%d %H:%M")

    payload = {
        "projects": [{"deadline": day_to_str(d)} for d in project_deadlines_days],
        "new_task": {"deadline": day_to_str(new_task_deadline_day)},
    }

    # 1) Extract features using your original logic
    feats = parse_payload(payload)
    past_due = feats.pop("_past_due", False)

    # 2) Hard rule: past due -> availability = 0
    if past_due:
        return 0.0

    # 3) No current work at all -> clamp to 100 (same as fuzzy_availability)
    if (
        feats["context_switch_load"] == 0
        and feats["workload_pressure"] == 0
        and feats["overlap_severity"] == 0
    ):
        return 100.0

    # 4) Run fuzzy system (REUSE global sim)
    sim = _AVAIL_SIM
    # reset if available (scikit-fuzzy has reset())
    if hasattr(sim, "reset"):
        sim.reset()

    for k, v in feats.items():
        sim.input[k] = float(v)

    try:
        sim.compute()
        score = float(sim.output["availability"])
    except Exception:
        score = 0.0

    return float(np.clip(score, 0.0, 100.0))



# ---------------------------------------------------------------------
# 3. Synthetic staff & task generation
# ---------------------------------------------------------------------


def generate_staff(n_staff: int, seed: int = 0) -> List[Staff]:
    random.seed(seed)
    staff_list: List[Staff] = []
    for i in range(n_staff):
        cap = random.uniform(4.0, 8.0)  # hours/day
        staff_list.append(Staff(id=i, capacity_hours_per_day=cap))
    return staff_list


def generate_tasks_for_regime(
    regime: Literal["under", "borderline", "overloaded"],
    staff: List[Staff],
    n_tasks_max: int,
    seed: int = 0,
) -> List[Task]:
    random.seed(seed)

    if regime == "under":
        factor = 0.6
    elif regime == "borderline":
        factor = 1.0
    else:
        factor = 1.4

    total_capacity_hours = sum(s.capacity_hours_per_day for s in staff) * 5.0  # 5-day horizon
    target_total_hours = total_capacity_hours * factor

    tasks: List[Task] = []
    accumulated_hours = 0.0
    tid = 0

    while accumulated_hours < target_total_hours and tid < n_tasks_max:
        hours = random.uniform(2.0, 8.0)
        deadline_day = random.uniform(1.0, 30.0)
        priority = random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]

        tasks.append(Task(id=tid, hours=hours, deadline_day=deadline_day, priority=priority))
        accumulated_hours += hours
        tid += 1

    return tasks


# ---------------------------------------------------------------------
# 4. Scheduling methods
# ---------------------------------------------------------------------


def schedule_tasks(
    staff_list: List[Staff],
    tasks: List[Task],
    method: Literal["fuzzy", "capacity", "random"],
) -> Dict[int, ScheduleResult]:
    """
    Build a schedule and return:
        {task_id: ScheduleResult(finish_day, staff_id)}
    """

    # Clone states so we don't mutate the original
    staff_states: Dict[int, Staff] = {
        s.id: Staff(
            id=s.id,
            capacity_hours_per_day=s.capacity_hours_per_day,
            current_hours=0.0,
            project_deadlines_days=list(s.project_deadlines_days),
        )
        for s in staff_list
    }

    # For deterministic behavior, order tasks by deadline
    tasks_sorted = sorted(tasks, key=lambda t: t.deadline_day)

    schedule: Dict[int, ScheduleResult] = {}

    for task in tasks_sorted:
        if method == "random":
            chosen_staff = random.choice(list(staff_states.values()))

        elif method == "capacity":
            horizon_days = 5.0

            def remaining_capacity(st: Staff) -> float:
                return st.capacity_hours_per_day * horizon_days - st.current_hours

            chosen_staff = max(staff_states.values(), key=remaining_capacity)

        elif method == "fuzzy":
            best_staff = None
            best_score = -1.0
            for st in staff_states.values():
                score = compute_availability_from_deadlines(
                    project_deadlines_days=st.project_deadlines_days,
                    new_task_deadline_day=task.deadline_day,
                )
                if score > best_score:
                    best_score = score
                    best_staff = st
            chosen_staff = best_staff
        else:
            raise ValueError(f"Unknown method: {method}")

        st = chosen_staff

        # Sequential processing model:
        start_day = st.current_hours / st.capacity_hours_per_day
        finish_day = start_day + task.hours / st.capacity_hours_per_day

        st.current_hours += task.hours
        st.project_deadlines_days.append(task.deadline_day)

        schedule[task.id] = ScheduleResult(finish_day=finish_day, staff_id=st.id)

    return schedule


# ---------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------


def compute_metrics(
    staff_list: List[Staff],
    schedule: Dict[int, ScheduleResult],
    tasks: List[Task],
    overload_threshold: float = 1.2,
) -> Dict[str, float]:
    task_by_id = {t.id: t for t in tasks}

    # Late tasks and lateness
    late_count = 0
    lateness_list: List[float] = []

    for tid, res in schedule.items():
        task = task_by_id[tid]
        lateness = max(0.0, res.finish_day - task.deadline_day)
        if lateness > 1e-9:
            late_count += 1
        lateness_list.append(lateness)

    n_tasks = len(tasks)
    late_rate = late_count / n_tasks if n_tasks else 0.0
    avg_lateness = float(np.mean(lateness_list)) if lateness_list else 0.0

    # Workload & utilization
    hours_per_staff: Dict[int, float] = {s.id: 0.0 for s in staff_list}
    for tid, res in schedule.items():
        task = task_by_id[tid]
        hours_per_staff[res.staff_id] += task.hours

    loads = np.array(list(hours_per_staff.values()), dtype=float)
    capacities = np.array([s.capacity_hours_per_day * 5.0 for s in staff_list], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        utilizations = np.where(capacities > 0, loads / capacities, 0.0)

    mean_load = float(np.mean(loads)) if len(loads) > 0 else 0.0
    std_load = float(np.std(loads)) if len(loads) > 0 else 0.0
    avg_util = float(np.mean(utilizations)) if len(utilizations) > 0 else 0.0

    # Gini for fairness
    def gini(x: np.ndarray) -> float:
        x = x.astype(float)
        if np.amin(x) < 0:
            x = x - np.amin(x)
        mean_x = np.mean(x)
        if mean_x == 0:
            return 0.0
        diff_sum = np.sum(np.abs(x[:, None] - x[None, :]))
        return float(diff_sum / (2 * len(x) ** 2 * mean_x))

    gini_load = gini(loads) if len(loads) > 0 else 0.0

    # Overloaded staff count
    overloaded = 0
    for load, cap in zip(loads, capacities):
        if cap <= 0:
            continue
        if load / cap > overload_threshold:
            overloaded += 1

    # Composite cost (you can tune alpha, beta in the paper)
    alpha = 1.0
    beta = 0.5
    if mean_load > 0:
        schedule_cost = alpha * late_rate + beta * (std_load / mean_load)
    else:
        schedule_cost = alpha * late_rate

    return {
        "late_rate": late_rate,
        "avg_lateness": avg_lateness,
        "workload_std": std_load,
        "avg_utilization": avg_util,
        "gini_load": gini_load,
        "overloaded_staff_count": float(overloaded),
        "schedule_cost": float(schedule_cost),
    }


# ---------------------------------------------------------------------
# 6. Running the experiments
# ---------------------------------------------------------------------


def ensure_plots_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def run_experiments(
    n_staff: int = 10,
    n_tasks_max: int = 60,
    n_runs: int = 50,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    methods = ["fuzzy", "capacity", "random"]
    regimes = ["under", "borderline", "overloaded"]

    # results[regime][method][metric] = list[float]
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for regime in regimes:
        results[regime] = {}
        for method in methods:
            results[regime][method] = {
                "late_rate": [],
                "avg_lateness": [],
                "workload_std": [],
                "avg_utilization": [],
                "gini_load": [],
                "overloaded_staff_count": [],
                "schedule_cost": [],
            }

        for run_idx in range(n_runs):
            seed = 1000 + run_idx
            staff = generate_staff(n_staff=n_staff, seed=seed)
            tasks = generate_tasks_for_regime(
                regime=regime,  # type: ignore[arg-type]
                staff=staff,
                n_tasks_max=n_tasks_max,
                seed=seed,
            )

            for method in methods:
                schedule = schedule_tasks(staff, tasks, method=method)  # type: ignore[arg-type]
                metrics = compute_metrics(staff, schedule, tasks)
                for mname, mval in metrics.items():
                    results[regime][method][mname].append(mval)

    return results


# ---------------------------------------------------------------------
# 7. Plotting helpers
# ---------------------------------------------------------------------


def boxplot_schedule_cost(results: Dict[str, Dict[str, Dict[str, List[float]]]]) -> None:
    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, "exp2_schedule_cost_boxplot.png")

    methods = ["fuzzy", "capacity", "random"]
    regimes = ["under", "borderline", "overloaded"]

    data = []
    labels = []
    for regime in regimes:
        for method in methods:
            data.append(results[regime][method]["schedule_cost"])
            labels.append(f"{regime[:3]}-{method[:3]}")

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Schedule cost")
    plt.title("Schedule cost across regimes and methods")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved boxplot to {path}")


def barplot_metric_by_regime(
    results: Dict[str, Dict[str, Dict[str, List[float]]]],
    metric: str,
    filename: str,
    title: str,
    ylabel: str,
    methods: List[str] = None,
):
    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, filename)

    if methods is None:
        methods = ["fuzzy", "capacity"]

    regimes = ["under", "borderline", "overloaded"]

    means = np.zeros((len(regimes), len(methods)))
    for i, regime in enumerate(regimes):
        for j, method in enumerate(methods):
            vals = results[regime][method][metric]
            means[i, j] = float(np.mean(vals)) if vals else 0.0

    x = np.arange(len(regimes))
    width = 0.35

    plt.figure(figsize=(7, 4))
    for j, method in enumerate(methods):
        plt.bar(x + (j - 0.5) * width, means[:, j], width=width, label=method.capitalize())

    plt.xticks(x, regimes)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved barplot to {path}")


# ---------------------------------------------------------------------
# 8. Main entry
# ---------------------------------------------------------------------


def main():
    print("[INFO] Running fuzzy scheduling experiments...")
    results = run_experiments(
        n_staff=5,
        n_tasks_max=30,
        n_runs=5,
    )

    print("[INFO] Generating plots...")
    boxplot_schedule_cost(results)
    barplot_metric_by_regime(
        results,
        metric="gini_load",
        filename="exp3_gini_by_regime.png",
        title="Fairness: Gini load vs load regime",
        ylabel="Mean Gini load (lower is better)",
    )
    barplot_metric_by_regime(
        results,
        metric="late_rate",
        filename="exp3_late_rate_by_regime.png",
        title="Robustness: late rate vs load regime",
        ylabel="Mean late rate",
    )
    print("[INFO] All plots saved in plots/ directory.")


if __name__ == "__main__":
    main()

