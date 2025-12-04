"""
Suitability FIS sensitivity experiments (FIS2).

- Uses structured staff + task metadata (no LLM, no HF login).
- Recreates the suitability fuzzy system from pipeline.fuzzy_system
  (availability, skill_match, experience, workload, priority, complexity,
   nts_avg, importance_avg -> suitability).
- Sweeps availability_score and skill_match and plots the resulting
  suitability scores.

Run from repo root:
    python -m experiments.suitability_sensitivity
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ---------------------------------------------------------------------
# 0. Utilities
# ---------------------------------------------------------------------

def ensure_plots_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def compute_skill_match(staff_skills: List[str], required_skills: List[str]) -> float:
    staff_set = set(map(str.lower, staff_skills))
    req_set = set(map(str.lower, required_skills))
    if not req_set:
        return 0.0
    return round(100 * len(staff_set & req_set) / len(req_set), 2)


def compute_nts_importance_avgs(task: Dict) -> (float, float):
    dims = [
        "ambiguity_tolerance",
        "communication",
        "planning",
        "collaboration",
        "reasoning",
        "risk_awareness",
        "ownership",
        "stakeholder_mgmt",
    ]

    nts = task.get("NTS_skills", {}) or {}
    imp = task.get("importance", {}) or {}

    nts_vals = [float(nts.get(k, 0)) for k in dims]
    imp_vals = [float(imp.get(k, 0)) for k in dims]

    nts_avg = sum(nts_vals) / len(dims)
    imp_avg = sum(imp_vals) / len(dims)
    return nts_avg, imp_avg


# ---------------------------------------------------------------------
# 1. Build the suitability FIS (FIS2) without the LLM
# ---------------------------------------------------------------------

def build_suitability_engine() -> ctrl.ControlSystemSimulation:
    # Antecedents
    availability = ctrl.Antecedent(np.arange(0, 101, 1), "availability")
    skill_match = ctrl.Antecedent(np.arange(0, 101, 1), "skill_match")
    experience = ctrl.Antecedent(np.arange(0, 5, 1), "experience")
    workload = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "workload")
    task_priority = ctrl.Antecedent(np.arange(0, 3, 1), "task_priority")
    task_complexity = ctrl.Antecedent(np.arange(1, 11, 1), "task_complexity")
    nts_avg = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "nts_avg")
    importance_avg = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "importance_avg")

    # Consequent
    suitability = ctrl.Consequent(np.arange(0, 101, 1), "suitability")

    # --- Membership functions (copied from your fuzzy_system.py) ---

    # Priority
    task_priority["low"] = fuzz.trimf(task_priority.universe, [0, 0, 1])
    task_priority["medium"] = fuzz.trimf(task_priority.universe, [0.5, 1, 1.5])
    task_priority["high"] = fuzz.trimf(task_priority.universe, [1, 2, 2])

    # Complexity
    task_complexity["low"] = fuzz.trimf(task_complexity.universe, [1, 1, 4])
    task_complexity["medium"] = fuzz.trimf(task_complexity.universe, [3, 5, 8])
    task_complexity["high"] = fuzz.trimf(task_complexity.universe, [7, 10, 10])

    # Availability
    availability["low"] = fuzz.trapmf(availability.universe, [0, 0, 30, 45])
    availability["medium"] = fuzz.trimf(availability.universe, [40, 55, 70])
    availability["high"] = fuzz.trapmf(availability.universe, [65, 80, 92, 96])
    availability["very_high"] = fuzz.trapmf(availability.universe, [90, 94, 100, 100])

    # Skill match
    skill_match["low"] = fuzz.trimf(skill_match.universe, [0, 0, 50])
    skill_match["medium"] = fuzz.trimf(skill_match.universe, [30, 50, 70])
    skill_match["high"] = fuzz.trimf(skill_match.universe, [60, 100, 100])

    # Experience
    experience["entry_level"] = fuzz.trimf(experience.universe, [0, 0, 1])
    experience["junior"] = fuzz.trimf(experience.universe, [0, 1, 2])
    experience["mid_level"] = fuzz.trimf(experience.universe, [1, 2, 3])
    experience["senior"] = fuzz.trimf(experience.universe, [2, 3, 4])

    # Workload
    workload["light"] = fuzz.trimf(workload.universe, [0, 0, 4])
    workload["moderate"] = fuzz.trimf(workload.universe, [3, 5.5, 8])
    workload["heavy"] = fuzz.trimf(workload.universe, [7, 10, 10])

    # NTS & importance
    nts_avg["low"] = fuzz.trimf(nts_avg.universe, [1.0, 1.0, 3.0])
    nts_avg["medium"] = fuzz.trimf(nts_avg.universe, [2.5, 3.0, 3.5])
    nts_avg["high"] = fuzz.trimf(nts_avg.universe, [3.0, 5.0, 5.0])

    importance_avg["low"] = fuzz.trimf(importance_avg.universe, [1.0, 1.0, 3.0])
    importance_avg["medium"] = fuzz.trimf(importance_avg.universe, [2.5, 3.0, 3.5])
    importance_avg["high"] = fuzz.trimf(importance_avg.universe, [3.0, 5.0, 5.0])

    # Suitability
    suitability["low"] = fuzz.trapmf(suitability.universe, [0, 0, 30, 45])
    suitability["medium"] = fuzz.trimf(suitability.universe, [40, 55, 70])
    suitability["high"] = fuzz.trapmf(suitability.universe, [70, 85, 100, 100])

    # --- Rules (simplified but aligned with your design) ---

    rules = []

    # Critical: high priority & high complexity only for top performers
    rules.append(
        ctrl.Rule(
            task_priority["high"] & task_complexity["high"] &
            availability["high"] & skill_match["high"] &
            experience["senior"] & workload["light"],
            suitability["high"],
        )
    )

    # High priority, medium complexity
    rules.append(
        ctrl.Rule(
            task_priority["high"] & task_complexity["medium"] &
            skill_match["high"] &
            (experience["senior"] | experience["mid_level"]) &
            (availability["high"] | availability["medium"]) &
            (workload["light"] | workload["moderate"]),
            suitability["high"],
        )
    )

    # Medium priority, high complexity
    rules.append(
        ctrl.Rule(
            task_priority["medium"] & task_complexity["high"] &
            (skill_match["high"] | experience["senior"]) &
            (availability["medium"] | availability["high"]) &
            workload["moderate"],
            suitability["medium"],
        )
    )

    # Medium priority, medium complexity
    rules.append(
        ctrl.Rule(
            task_priority["medium"] & task_complexity["medium"] &
            skill_match["medium"] & experience["mid_level"] &
            availability["medium"] & workload["moderate"],
            suitability["medium"],
        )
    )

    # Low priority, low complexity → entry/junior/mid
    rules.append(
        ctrl.Rule(
            task_priority["low"] & task_complexity["low"] &
            (experience["entry_level"] | experience["junior"] | experience["mid_level"]) &
            (skill_match["medium"] | skill_match["high"]) &
            workload["light"],
            suitability["medium"],
        )
    )

    # Strong negative factors
    rules.append(
        ctrl.Rule(
            skill_match["low"] | workload["heavy"] | availability["low"],
            suitability["low"],
        )
    )

    # Entry-level with low availability & low skill
    rules.append(
        ctrl.Rule(
            availability["low"] & skill_match["low"] & experience["entry_level"],
            suitability["low"],
        )
    )

    # Broad good-case path
    rules.append(
        ctrl.Rule(
            (availability["high"] | availability["very_high"]) &
            skill_match["high"] &
            (workload["light"] | workload["moderate"]),
            suitability["high"],
        )
    )

    # Medium band
    rules.append(
        ctrl.Rule(
            availability["medium"] & skill_match["medium"] &
            (workload["light"] | workload["moderate"]),
            suitability["medium"],
        )
    )

    # Junior with good availability & skills
    rules.append(
        ctrl.Rule(
            experience["junior"] &
            availability["high"] &
            (skill_match["medium"] | skill_match["high"]) &
            workload["light"],
            suitability["medium"],
        )
    )

    # NTS high & important → boost
    rules.append(
        ctrl.Rule(
            nts_avg["high"] & importance_avg["high"] &
            (availability["high"] | availability["very_high"]) &
            (skill_match["medium"] | skill_match["high"]),
            suitability["high"],
        )
    )

    # NTS medium & importance medium
    rules.append(
        ctrl.Rule(
            nts_avg["medium"] & importance_avg["medium"] &
            (availability["medium"] | availability["high"]) &
            (skill_match["medium"] | skill_match["high"]),
            suitability["medium"],
        )
    )

    # NTS/importance low on hard tasks
    rules.append(
        ctrl.Rule(
            (nts_avg["low"] | importance_avg["low"]) &
            (task_priority["high"] | task_complexity["high"]),
            suitability["low"],
        )
    )

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim


_SUIT_SIM = build_suitability_engine()

priority_map = {"low": 0, "medium": 1, "high": 2}
experience_map = {"entry_level": 0, "junior": 1, "mid_level": 2, "senior": 3}


def compute_suitability_score(
    availability_score: float,
    skill_match_val: float,
    experience_label: str,
    workload_count: int,
    task_priority_label: str,
    task_complexity_val: float,
    nts_avg_val: float,
    importance_avg_val: float,
) -> float:
    sim = _SUIT_SIM
    if hasattr(sim, "reset"):
        sim.reset()

    sim.input["availability"] = float(np.clip(availability_score, 0, 100))
    sim.input["skill_match"] = float(np.clip(skill_match_val, 0, 100))
    sim.input["experience"] = float(np.clip(experience_map[experience_label], 0, 4))
    sim.input["workload"] = float(np.clip(workload_count, 0, 10))
    sim.input["task_priority"] = float(np.clip(priority_map[task_priority_label], 0, 2))
    sim.input["task_complexity"] = float(np.clip(task_complexity_val, 1, 10))
    sim.input["nts_avg"] = float(np.clip(nts_avg_val, 1, 5))
    sim.input["importance_avg"] = float(np.clip(importance_avg_val, 1, 5))

    sim.compute()

    suit_val = sim.output.get("suitability", np.nan)
    if suit_val is None or (isinstance(suit_val, float) and np.isnan(suit_val)):
        suit = 50.0   # fallback
    else:
        suit = float(suit_val)

    return round(suit, 2)



# ---------------------------------------------------------------------
# 2. Base scenario from your example staff + task
# ---------------------------------------------------------------------

# Example staff (clone of what you posted)
example_staff = {
    "availability": [
        {"deadline": "2025-12-15 17:00"},
        {"deadline": "2026-02-01 17:00"},
    ],
    "skills": [
        "Python",
        "scikit-learn",
        "XGBoost",
        "ROC-AUC",
        "F1",
        "Airflow",
        "Data Engineering",
    ],
    "experience": "junior",
}

example_task = {
    "Task Priority": "medium",
    "Task Complexity": 5,
    "Required Skills": [
        "scikit-learn",
        "XGBoost",
        "pipeline",
        "ROC-AUC",
        "F1-score",
        "Airflow DAG",
    ],
    "NTS_skills": {
        "ambiguity_tolerance": 3,
        "communication": 4,
        "planning": 4,
        "collaboration": 4,
        "reasoning": 4,
        "risk_awareness": 3,
        "ownership": 4,
        "stakeholder_mgmt": 3,
    },
    "importance": {
        "ambiguity_tolerance": 3,
        "communication": 4,
        "planning": 4,
        "collaboration": 4,
        "reasoning": 4,
        "risk_awareness": 3,
        "ownership": 4,
        "stakeholder_mgmt": 3,
    },
}

# Pre-compute skill_match and NTS averages for the base scenario
base_skill_match = compute_skill_match(
    example_staff["skills"], example_task["Required Skills"]
)
base_nts_avg, base_importance_avg = compute_nts_importance_avgs(example_task)
base_workload = len(example_staff["availability"])
base_priority = example_task["Task Priority"]
base_experience = example_staff["experience"]


# ---------------------------------------------------------------------
# 3. Sensitivity experiments
# ---------------------------------------------------------------------

def sweep_availability():
    xs = np.linspace(0, 100, 21)
    ys = []
    for a in xs:
        s = compute_suitability_score(
            availability_score=a,
            skill_match_val=base_skill_match,
            experience_label=base_experience,
            workload_count=base_workload,
            task_priority_label=base_priority,
            task_complexity_val=example_task["Task Complexity"],
            nts_avg_val=base_nts_avg,
            importance_avg_val=base_importance_avg,
        )
        ys.append(s)
    return xs, np.array(ys)


def sweep_skill_match():
    xs = np.linspace(0, 100, 21)
    ys = []
    for sm in xs:
        s = compute_suitability_score(
            availability_score=80.0,  # fix availability high
            skill_match_val=sm,
            experience_label=base_experience,
            workload_count=base_workload,
            task_priority_label=base_priority,
            task_complexity_val=example_task["Task Complexity"],
            nts_avg_val=base_nts_avg,
            importance_avg_val=base_importance_avg,
        )
        ys.append(s)
    return xs, np.array(ys)


def plot_curve(x, y, xlabel, title, filename):
    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, filename)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Suitability score")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved plot to {path}")

def heatmap_availability_skill():
    # grid
    avail_vals = np.linspace(0, 100, 41)
    skill_vals = np.linspace(0, 100, 41)

    Z = np.zeros((len(skill_vals), len(avail_vals)))
    for i, sm in enumerate(skill_vals):
        for j, a in enumerate(avail_vals):
            Z[i, j] = compute_suitability_score(
                availability_score=a,
                skill_match_val=sm,
                experience_label=base_experience,
                workload_count=base_workload,
                task_priority_label=base_priority,
                task_complexity_val=example_task["Task Complexity"],
                nts_avg_val=base_nts_avg,
                importance_avg_val=base_importance_avg,
            )

    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, "exp_suitability_heatmap_avail_skill.png")

    plt.figure(figsize=(6, 5))
    plt.imshow(
        Z,
        origin="lower",
        extent=[avail_vals[0], avail_vals[-1], skill_vals[0], skill_vals[-1]],
        aspect="auto",
    )
    plt.colorbar(label="Suitability score")
    plt.xlabel("Availability score (0–100)")
    plt.ylabel("Skill match (%)")
    plt.title("Suitability as a function of availability and skill match")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved heatmap to {path}")



def main():
    print("[INFO] Sweeping availability...")
    x_a, y_a = sweep_availability()
    plot_curve(
        x_a,
        y_a,
        xlabel="Availability score (0–100)",
        title="Suitability vs availability (FIS2)",
        filename="exp_suitability_vs_availability.png",
    )

    print("[INFO] Sweeping skill_match...")
    x_sm, y_sm = sweep_skill_match()
    plot_curve(
        x_sm,
        y_sm,
        xlabel="Skill match (%)",
        title="Suitability vs skill match (FIS2)",
        filename="exp_suitability_vs_skillmatch.png",
    )

    print("[INFO] Generating heatmap...")
    heatmap_availability_skill()

    print("[INFO] Done. Check the plots/ folder.")


if __name__ == "__main__":
    main()
