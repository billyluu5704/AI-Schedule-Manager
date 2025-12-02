"""
Plot membership functions for the Suitability FIS (task–staff matching).

This mirrors the definitions in fuzzy_system.py:
  Inputs:
    - availability, skill_match, experience, workload,
      task_priority, task_complexity, nts_avg, importance_avg
  Output:
    - suitability

Run from repo root:
    python -m experiments.plot_suitability_mfs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def ensure_plots_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def build_suitability_variables():
    # Antecedents
    availability = ctrl.Antecedent(np.arange(0, 101, 1), "availability")
    skill_match = ctrl.Antecedent(np.arange(0, 101, 1), "skill_match")
    experience = ctrl.Antecedent(np.arange(0, 5, 1), "experience")
    workload = ctrl.Antecedent(np.arange(0, 11, 1), "workload")
    task_priority = ctrl.Antecedent(np.arange(0, 3, 1), "task_priority")
    task_complexity = ctrl.Antecedent(np.arange(1, 11, 1), "task_complexity")
    nts_avg = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "nts_avg")
    importance_avg = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "importance_avg")

    # Consequent
    suitability = ctrl.Consequent(np.arange(0, 101, 1), "suitability")

    # ---- Membership functions (copied from your fuzzy_system.py) ----

    # Task priority
    task_priority["low"] = fuzz.trimf(task_priority.universe, [0, 0, 1])
    task_priority["medium"] = fuzz.trimf(task_priority.universe, [0.5, 1, 1.5])
    task_priority["high"] = fuzz.trimf(task_priority.universe, [1, 2, 2])

    # Task complexity
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

    # NTS average
    nts_avg["low"] = fuzz.trimf(nts_avg.universe, [1.0, 1.0, 3.0])
    nts_avg["medium"] = fuzz.trimf(nts_avg.universe, [2.5, 3.0, 3.5])
    nts_avg["high"] = fuzz.trimf(nts_avg.universe, [3.0, 5.0, 5.0])

    # Importance average
    importance_avg["low"] = fuzz.trimf(importance_avg.universe, [1.0, 1.0, 3.0])
    importance_avg["medium"] = fuzz.trimf(importance_avg.universe, [2.5, 3.0, 3.5])
    importance_avg["high"] = fuzz.trimf(importance_avg.universe, [3.0, 5.0, 5.0])

    # Suitability
    suitability["low"] = fuzz.trapmf(suitability.universe, [0, 0, 30, 45])
    suitability["medium"] = fuzz.trimf(suitability.universe, [40, 55, 70])
    suitability["high"] = fuzz.trapmf(suitability.universe, [70, 85, 100, 100])

    return (
        availability,
        skill_match,
        experience,
        workload,
        task_priority,
        task_complexity,
        nts_avg,
        importance_avg,
        suitability,
    )


def plot_input_mfs(vars_list):
    """
    vars_list: list of (Antecedent, title, xlabel)
    """
    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, "exp_suitability_input_mfs.png")

    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    fig.suptitle("Input Membership Functions of Suitability FIS")

    for ax, (var, title, xlabel) in zip(axes.flat, vars_list):
        for term_name, term in var.terms.items():
            ax.plot(var.universe, term.mf, label=term_name)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Degree of membership")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

    # Hide any unused axes (in case)
    for ax in axes.flat[len(vars_list):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved input MF figure to {path}")


def plot_output_mfs(suitability):
    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, "exp_suitability_output_mfs.png")

    plt.figure(figsize=(6, 4))
    for term_name, term in suitability.terms.items():
        plt.plot(suitability.universe, term.mf, label=term_name)

    plt.title("Output Membership Functions: suitability")
    plt.xlabel("Suitability score")
    plt.ylabel("Degree of membership")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved output MF figure to {path}")


def main():
    (
        availability,
        skill_match,
        experience,
        workload,
        task_priority,
        task_complexity,
        nts_avg,
        importance_avg,
        suitability,
    ) = build_suitability_variables()

    vars_list = [
        (availability,    "availability",      "Availability score (0–100)"),
        (skill_match,     "skill_match",       "Skill match (%)"),
        (experience,      "experience",        "Experience level index"),
        (workload,        "workload",          "Number of active tasks"),
        (task_priority,   "task_priority",     "Priority level index"),
        (task_complexity, "task_complexity",   "Complexity (1–10)"),
        (nts_avg,         "nts_avg",           "Avg NTS rating (1–5)"),
        (importance_avg,  "importance_avg",    "Avg importance rating (1–5)"),
    ]

    plot_input_mfs(vars_list)
    plot_output_mfs(suitability)


if __name__ == "__main__":
    main()
