"""
Plot membership functions for the availability fuzzy inference system.

This script recreates the same Antecedents/Consequent as
pipeline.fuzzy_logic_for_availability_check.fuzzy_system_availability()
and plots:

  - A 2x2 grid for the 4 input variables:
        workload_pressure, deadline_buffer_days,
        overlap_severity, context_switch_load
  - A single plot for the output:
        availability

Run from repo root:
    python -m experiments.plot_availability_mfs
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


def build_availability_variables():
    """
    Rebuild the fuzzy variables exactly as in fuzzy_logic_for_availability_check.py
    (for plotting only).
    """
    work_pressure = ctrl.Antecedent(np.linspace(0, 2.5, 251),
                                    "workload_pressure")
    buffer = ctrl.Antecedent(np.linspace(0, 30, 301),
                             "deadline_buffer_days")
    overlap = ctrl.Antecedent(np.linspace(0, 1, 101),
                              "overlap_severity")
    context = ctrl.Antecedent(np.linspace(0, 12, 121),
                              "context_switch_load")

    availability = ctrl.Consequent(np.linspace(0, 100, 101),
                                   "availability")

    # --- Input membership functions (copied from your file) ---

    # Work pressure
    work_pressure["low"] = fuzz.trapmf(work_pressure.universe,
                                       [0.0, 0.0, 0.45, 0.7])
    work_pressure["med"] = fuzz.trimf(work_pressure.universe,
                                      [0.4, 0.9, 1.2])
    work_pressure["high"] = fuzz.trapmf(work_pressure.universe,
                                        [1.0, 1.2, 2.5, 2.5])

    # Buffer (0..30)
    buffer["small"] = fuzz.trapmf(buffer.universe,
                                  [0, 0, 2, 6])
    buffer["med"] = fuzz.trimf(buffer.universe,
                               [4, 10, 16])
    buffer["large"] = fuzz.trapmf(buffer.universe,
                                  [12, 18, 30, 30])

    # Overlap
    overlap["low"] = fuzz.trapmf(overlap.universe,
                                 [0, 0, 0.2, 0.35])
    overlap["med"] = fuzz.trimf(overlap.universe,
                                [0.25, 0.5, 0.75])
    overlap["high"] = fuzz.trapmf(overlap.universe,
                                  [0.6, 0.75, 1, 1])

    # Context switches
    context["few"] = fuzz.trapmf(context.universe,
                                 [0, 0, 1, 3])
    context["moderate"] = fuzz.trimf(context.universe,
                                     [2, 4, 6])
    context["many"] = fuzz.trapmf(context.universe,
                                  [5, 7, 12, 12])

    # --- Output membership functions ---

    availability["none"] = fuzz.trapmf(availability.universe,
                                       [0, 0, 10, 20])
    availability["low"] = fuzz.trimf(availability.universe,
                                     [15, 30, 45])
    availability["moderate"] = fuzz.trimf(availability.universe,
                                          [40, 55, 70])
    availability["high"] = fuzz.trapmf(availability.universe,
                                       [65, 78, 90, 95])
    availability["very_high"] = fuzz.trapmf(availability.universe,
                                            [88, 93, 100, 100])

    return work_pressure, buffer, overlap, context, availability


def plot_input_mfs(work_pressure, buffer, overlap, context):
    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, "exp0_input_mfs.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("Input Membership Functions of Availability FIS")

    # Helper to plot all terms of one variable on an axis
    def plot_var(ax, var, title, xlabel):
        for term_name, term in var.terms.items():
            ax.plot(var.universe, term.mf, label=term_name)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Degree of membership")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

    plot_var(axes[0, 0], work_pressure,
             "workload_pressure", "Tasks per business day")
    plot_var(axes[0, 1], buffer,
             "deadline_buffer_days", "Business days until nearest deadline")
    plot_var(axes[1, 0], overlap,
             "overlap_severity", "Overlap ratio (0–1)")
    plot_var(axes[1, 1], context,
             "context_switch_load", "Number of active projects")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved input MF figure to {path}")


def plot_output_mfs(availability):
    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, "exp0_output_mfs.png")

    plt.figure(figsize=(6, 4))
    for term_name, term in availability.terms.items():
        plt.plot(availability.universe, term.mf, label=term_name)

    plt.title("Output Membership Functions: availability")
    plt.xlabel("Availability score")
    plt.ylabel("Degree of membership")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved output MF figure to {path}")


def main():
    work_pressure, buffer, overlap, context, availability = build_availability_variables()
    plot_input_mfs(work_pressure, buffer, overlap, context)
    plot_output_mfs(availability)


if __name__ == "__main__":
    main()
