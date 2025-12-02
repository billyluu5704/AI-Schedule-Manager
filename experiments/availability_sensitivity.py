"""
Experiment 1: Sensitivity analysis for the availability fuzzy system.

Uses your existing:
  - pipeline.fuzzy_logic_for_availability_check.fuzzy_system_availability

We sweep:
  - workload_pressure (0 → 2.5)
  - deadline_buffer_days (0 → 30)
  - overlap_severity (0 → 1)
  - context_switch_load (0 → 12)

and plot how the fuzzified availability responds.

Run from repo root:
    python -m experiments.availability_sensitivity
"""

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from pipeline.fuzzy_logic_for_availability_check import fuzzy_system_availability


def ensure_plots_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def eval_availability(
    workload_pressure: float,
    deadline_buffer_days: float,
    overlap_severity: float,
    context_switch_load: float,
) -> float:
    """
    Call your fuzzy availability engine with raw feature values.
    This uses fuzzy_system_availability() directly (no JSON / parse_payload),
    so it isolates the fuzzy logic behavior.
    """
    sim = fuzzy_system_availability()

    sim.input["workload_pressure"] = float(workload_pressure)
    sim.input["deadline_buffer_days"] = float(deadline_buffer_days)
    sim.input["overlap_severity"] = float(overlap_severity)
    sim.input["context_switch_load"] = float(context_switch_load)

    sim.compute()
    score = float(sim.output["availability"])
    return score


def sweep_single_feature(
    feature_name: str,
    x_values: np.ndarray,
    base_vals: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep one feature at a time, keeping others fixed to base_vals.
    """
    scores = []

    for x in x_values:
        vals = base_vals.copy()
        vals[feature_name] = float(x)
        score = eval_availability(
            workload_pressure=vals["workload_pressure"],
            deadline_buffer_days=vals["deadline_buffer_days"],
            overlap_severity=vals["overlap_severity"],
            context_switch_load=vals["context_switch_load"],
        )
        scores.append(score)

    return x_values, np.array(scores)


def plot_sweep(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    title: str,
    filename: str,
):
    plots_dir = ensure_plots_dir()
    path = os.path.join(plots_dir, filename)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Availability score")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[INFO] Saved plot to {path}")


def main():
    # Baseline "typical" values
    base_vals = {
        "workload_pressure": 0.8,      # moderate tasks per business day
        "deadline_buffer_days": 10.0,  # decent buffer
        "overlap_severity": 0.3,       # low/med overlap
        "context_switch_load": 4.0,    # a few concurrent projects
    }

    # 1) Workload sweep
    wr_vals = np.linspace(0.0, 2.5, num=26)
    x_wr, y_wr = sweep_single_feature("workload_pressure", wr_vals, base_vals)
    plot_sweep(
        x_wr,
        y_wr,
        xlabel="workload_pressure (tasks / business day)",
        title="Availability vs workload_pressure",
        filename="exp1_availability_vs_workload_pressure.png",
    )

    # 2) Deadline buffer sweep
    buf_vals = np.linspace(0.0, 30.0, num=31)
    x_buf, y_buf = sweep_single_feature("deadline_buffer_days", buf_vals, base_vals)
    plot_sweep(
        x_buf,
        y_buf,
        xlabel="deadline_buffer_days",
        title="Availability vs deadline_buffer_days",
        filename="exp1_availability_vs_deadline_buffer.png",
    )

    # 3) Overlap severity sweep
    ov_vals = np.linspace(0.0, 1.0, num=21)
    x_ov, y_ov = sweep_single_feature("overlap_severity", ov_vals, base_vals)
    plot_sweep(
        x_ov,
        y_ov,
        xlabel="overlap_severity (0–1)",
        title="Availability vs overlap_severity",
        filename="exp1_availability_vs_overlap_severity.png",
    )

    # 4) Context switch load sweep
    ctx_vals = np.linspace(0.0, 12.0, num=25)
    x_ctx, y_ctx = sweep_single_feature("context_switch_load", ctx_vals, base_vals)
    plot_sweep(
        x_ctx,
        y_ctx,
        xlabel="context_switch_load (number of projects)",
        title="Availability vs context_switch_load",
        filename="exp1_availability_vs_context_switch.png",
    )


if __name__ == "__main__":
    main()
