"""
evaluate.py
============
Loads the trained PPO/DQN agents plus Round Robin and FCFS baselines,
runs each over the SAME held-out task stream, and produces:
  - results/comparison_table.csv        (Table 1 equivalent)
  - results/fig2_multi_objective.png    (latency/energy bar chart)
  - results/fig3_forecast_accuracy.png  (BiLSTM predicted vs actual)
  - results/fig4_swarm_distribution.png (task routing per node, per policy)
  - results/fig5_qos_routing.png        (Priority-1 Fog-routing success rate)
  - results/fig6_cost_comparison.png    (operational cost, reference only)
  - results/fig7_relative_improvement.png (% improvement of PPO vs each baseline)
  - results/fig8_training_curves.png    (PPO vs DQN episode reward during training --
                                          requires re-running train_agents.py with
                                          Monitor logging; skipped with a message
                                          if the log files aren't found)
  - results/fig9_latency_energy_distribution.png (per-task boxplots, not just the mean)
  - results/fig10_forecast_residuals.png (BiLSTM forecast error over time, reusing fig3's data)

This script prints real numbers computed from real simulation runs. It
does NOT hardcode or assume any particular ranking between RR / FCFS /
DQN / PPO -- whatever the trained policies actually do is what gets
reported. If PPO does not win on every axis, that is useful, honest
information for your paper (you can report partial wins, or iterate on
the reward weights / training budget, rather than the script quietly
making PPO look better than it is).
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from fog_cloud_env import FogCloudEnv, NODES
from baselines import RoundRobinPolicy, FCFSPolicy, run_episode
from forecasting import LoadForecastingPipeline, WINDOW

from stable_baselines3 import PPO, DQN

# Single source of truth for method ordering/colors -- every table, print,
# and figure below iterates this instead of repeating a literal list, so
# adding/removing a baseline (like FCFS here) only requires editing this
# one place.
METHOD_NAMES = ["RoundRobin", "FCFS", "DQN", "PPO (proposed)"]
METHOD_COLORS = ["#8c8c8c", "#dd8452", "#4c72b0", "#55a868"]


def summarize(info_log: list[dict]) -> dict:
    df = pd.DataFrame(info_log)
    return dict(
        mean_latency_s=df["latency_s"].mean(),
        total_energy_w=df["energy_w"].sum(),
        total_cost_usd=df["cost_usd"].sum(),
        qos_violation_rate=df["qos_violation"].mean(),
        crash_violation_rate=df["crash_violation"].mean(),
        p1_fog_routing_success=1.0 - df.loc[df["priority"] == 1, "qos_violation"].mean()
        if (df["priority"] == 1).any() else float("nan"),
        node_distribution=df["node"].value_counts(normalize=True).to_dict(),
        raw=df,
    )


def pct_improvement(baseline, proposed):
    if baseline == 0:
        return float("nan")
    return 100.0 * (baseline - proposed) / baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="../models")
    ap.add_argument("--results-dir", type=str, default="../results")
    ap.add_argument("--eval-fraction", type=float, default=0.25,
                     help="Fraction of tasks (from the end) held out purely for evaluation")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    tasks = pd.read_csv(os.path.join(args.model_dir, "eval_tasks.csv"))
    lpred = np.load(os.path.join(args.model_dir, "eval_lpred.npy"))
    cpu_trace = np.load(os.path.join(args.model_dir, "eval_cpu_trace.npy"))

    n_eval = int(len(tasks) * args.eval_fraction)
    eval_tasks = tasks.iloc[-n_eval:].reset_index(drop=True)
    eval_lpred = lpred[-n_eval:]

    ppo = PPO.load(os.path.join(args.model_dir, "ppo_agent"))
    dqn = DQN.load(os.path.join(args.model_dir, "dqn_agent"))
    rr = RoundRobinPolicy(n_fog_nodes=3, include_cloud=False)
    fcfs = FCFSPolicy(n_fog_nodes=3)

    results = {}
    for name, policy, is_sb3 in [
        ("RoundRobin", rr, False),
        ("FCFS", fcfs, False),
        ("DQN", dqn, True),
        ("PPO (proposed)", ppo, True),
    ]:
        env = FogCloudEnv(eval_tasks, eval_lpred, seed=0)
        log = run_episode(env, policy, is_sb3=is_sb3)
        results[name] = summarize(log)

    # ---------------- Table 1 equivalent (latency + energy only) ----------------
    # Cost was dropped from the optimisation objective and from this primary
    # comparison per your request. It's still logged in results['raw'] /
    # comparison_table_with_cost_reference.csv below purely as an
    # informational reference (e.g. for an appendix table), never as
    # something either agent was trained to minimise.
    table_rows = []
    for name in METHOD_NAMES:
        r = results[name]
        table_rows.append(dict(
            Method=name,
            Mean_Latency_s=round(r["mean_latency_s"], 4),
            Total_Energy_W=round(r["total_energy_w"], 2),
            QoS_Violation_Rate=round(r["qos_violation_rate"], 4),
            Crash_Violation_Rate=round(r["crash_violation_rate"], 4),
            P1_Fog_Routing_Success=round(r["p1_fog_routing_success"], 4)
            if not np.isnan(r["p1_fog_routing_success"]) else "n/a",
        ))
    table = pd.DataFrame(table_rows)
    table.to_csv(os.path.join(args.results_dir, "comparison_table.csv"), index=False)
    print(table.to_string(index=False))

    # informational-only reference table including cost, not used for any claim
    ref_rows = [dict(Method=n, Total_Cost_USD=round(results[n]["total_cost_usd"], 2))
                for n in METHOD_NAMES]
    pd.DataFrame(ref_rows).to_csv(
        os.path.join(args.results_dir, "cost_reference_not_optimized.csv"), index=False)

    rr_row = table[table.Method == "RoundRobin"].iloc[0]
    fcfs_row = table[table.Method == "FCFS"].iloc[0]
    dqn_row = table[table.Method == "DQN"].iloc[0]
    ppo_row = table[table.Method == "PPO (proposed)"].iloc[0]
    improvements = {}  # {baseline_name: {"Mean_Latency_s": pct, "Total_Energy_W": pct}}
    for baseline_name, baseline_row in [("RoundRobin", rr_row), ("FCFS", fcfs_row), ("DQN", dqn_row)]:
        improvements[baseline_name] = {
            col: pct_improvement(baseline_row[col], ppo_row[col])
            for col in ["Mean_Latency_s", "Total_Energy_W"]
        }
        print(f"\n--- % improvement of PPO vs {baseline_name} ---")
        for col, val in improvements[baseline_name].items():
            print(f"{col}: {val:.2f}%")

    # ---------------- Figure 2: multi-objective bar chart (latency + energy) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics = ["Mean_Latency_s", "Total_Energy_W"]
    titles = ["Mean Latency (s)", "Total Energy (W)"]
    for ax, m, t in zip(axes, metrics, titles):
        ax.bar(table["Method"], table[m], color=METHOD_COLORS)
        ax.set_title(t)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Latency / energy comparison (held-out evaluation tasks)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig2_multi_objective.png"), dpi=150)
    plt.close(fig)

    # ---------------- Figure 3: forecast accuracy ----------------
    forecaster = LoadForecastingPipeline().fit(cpu_trace, epochs=10, verbose=False)
    preds = forecaster.rolling_forecast(cpu_trace)
    actual = cpu_trace[WINDOW:]
    # Export Figure 3 data for Gnuplot
    forecast_df = pd.DataFrame({
        "Timestep": np.arange(len(actual)),
        "Actual": actual,
        "Prediction": preds
    })
    forecast_df.to_csv(
        os.path.join(args.results_dir, "fig3_forecast_data.csv"),
        index=False
    )

    print("\n===== FIGURE 3 DATA =====")
    print(forecast_df.head())
    print("Saved: fig3_forecast_data.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    span = slice(0, min(400, len(actual)))
    ax.plot(actual[span], label="Actual CPU utilisation")
    ax.plot(preds[span], label="BiLSTM prediction", linestyle="--")
    ax.set_title("BiLSTM load forecast vs actual (Yeo-Johnson transformed)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("CPU utilisation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig3_forecast_accuracy.png"), dpi=150)
    plt.close(fig)
    mae = np.mean(np.abs(preds - actual))
    print(f"\nBiLSTM rolling forecast MAE (full trace): {mae:.4f}")

    # ---------------- Figure 4: swarm distribution ----------------
    # Labelled with exact percentages -- important here specifically
    # because a bar that LOOKS like zero and a bar that IS exactly zero
    # are visually indistinguishable otherwise, and whether the learned
    # agents actually use Fog-2/Fog-3 at all is a real finding worth
    # reading precisely rather than eyeballing.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, name in zip(axes, METHOD_NAMES):
        dist = results[name]["node_distribution"]
        names = [n["name"] for n in NODES]
        vals = [dist.get(n, 0.0) for n in names]
        bars = ax.bar(names, vals, color="#4c72b0")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v*100:.1f}%",
                     ha="center", fontsize=8)
        ax.set_title(name)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Task distribution across the Fog-Cloud swarm")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig4_swarm_distribution.png"), dpi=150)
    plt.close(fig)

    # ---------------- Figure 5: QoS routing success ----------------
    fig, ax = plt.subplots(figsize=(8, 4))
    succ = [results[n]["p1_fog_routing_success"] for n in METHOD_NAMES]
    ax.bar(METHOD_NAMES, succ, color="#c44e52")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Priority-1 Fog-routing success rate")
    ax.set_title("QoS priority routing success")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig5_qos_routing.png"), dpi=150)
    plt.close(fig)

    # ---------------- Figure 6: cost comparison (reference only, not optimized) ----------------
    fig, ax = plt.subplots(figsize=(8, 4))
    costs = [results[n]["total_cost_usd"] for n in METHOD_NAMES]
    ax.bar(METHOD_NAMES, costs, color=METHOD_COLORS)
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Operational cost -- logged for reference only, not part of the reward")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig6_cost_comparison.png"), dpi=150)
    plt.close(fig)

    # ---------------- Figure 7: relative improvement summary ----------------
    # One consolidated view of "% improvement of PPO vs each baseline" on
    # both latency and energy, computed from the SAME improvements dict
    # printed to the console above -- nothing here is recomputed
    # differently or rounded differently from what's in the console log.
    baseline_names = ["RoundRobin", "FCFS", "DQN"]
    lat_vals = [improvements[b]["Mean_Latency_s"] for b in baseline_names]
    eng_vals = [improvements[b]["Total_Energy_W"] for b in baseline_names]
    x = np.arange(len(baseline_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, lat_vals, width, label="Latency improvement (%)", color="#4c72b0")
    ax.bar(x + width / 2, eng_vals, width, label="Energy improvement (%)", color="#55a868")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_names)
    ax.set_ylabel("% improvement of PPO vs baseline")
    ax.set_title("PPO's relative improvement over each baseline")
    ax.legend()
    for xi, (lv, ev) in enumerate(zip(lat_vals, eng_vals)):
        ax.text(xi - width / 2, lv + (1 if lv >= 0 else -3), f"{lv:.1f}%", ha="center", fontsize=8)
        ax.text(xi + width / 2, ev + (1 if ev >= 0 else -3), f"{ev:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig7_relative_improvement.png"), dpi=150)
    plt.close(fig)

    # ---------------- Figure 8: training reward curves (PPO vs DQN) ----------------
    # Reads the Monitor CSV logs written by train_agents.py (one row per
    # completed training episode: episode reward 'r', episode length 'l',
    # wall-clock time 't'). Requires having re-run train_agents.py with
    # the monitor-logging version -- if the log files aren't found, this
    # figure is skipped with a clear message rather than failing silently
    # or fabricating a curve.
    def load_monitor_csv(path):
        return pd.read_csv(path, skiprows=1)  # skip the '#{...}' JSON comment line

    dqn_monitor_path = os.path.join(args.model_dir, "dqn_monitor.monitor.csv")
    ppo_monitor_dir = os.path.join(args.model_dir, "ppo_monitor_logs")

    if os.path.exists(dqn_monitor_path) and os.path.isdir(ppo_monitor_dir):
        dqn_hist = load_monitor_csv(dqn_monitor_path)
        dqn_hist["cum_timesteps"] = dqn_hist["l"].cumsum()

        ppo_files = sorted(f for f in os.listdir(ppo_monitor_dir) if f.endswith(".monitor.csv"))
        ppo_hist = pd.concat(
            [load_monitor_csv(os.path.join(ppo_monitor_dir, f)) for f in ppo_files],
            ignore_index=True,
        )
        # With the default --ppo-n-envs 1 there's a single, unambiguous
        # per-episode ordering. With >1 parallel envs this sorts by
        # wall-clock completion time as an approximation of arrival order
        # across envs, since Monitor doesn't expose a shared global step
        # counter -- noted here rather than silently treated as exact.
        ppo_hist = ppo_hist.sort_values("t").reset_index(drop=True)
        ppo_hist["cum_timesteps"] = ppo_hist["l"].cumsum()
        # Export Figure 8 data
        dqn_hist[["cum_timesteps", "r"]].rename(columns={"cum_timesteps": "Timesteps", "r": "Reward"}).to_csv(os.path.join(args.results_dir, "fig8_dqn_training.csv"),index=False)
        ppo_hist[["cum_timesteps", "r"]].rename(columns={"cum_timesteps": "Timesteps", "r": "Reward"}).to_csv(os.path.join(args.results_dir, "fig8_ppo_training.csv"),index=False)

        print("\n===== FIGURE 8 DATA =====")
        print("Saved: fig8_dqn_training.csv")
        print("Saved: fig8_ppo_training.csv")
        # Export Figure 9 raw distributions
        latency_export = pd.DataFrame({
            name: pd.Series(results[name]["raw"]["latency_s"].values)
            for name in METHOD_NAMES
        })

        energy_export = pd.DataFrame({
            name: pd.Series(results[name]["raw"]["energy_w"].values)
            for name in METHOD_NAMES
        })

        latency_export.to_csv(
            os.path.join(args.results_dir, "fig9_latency_distribution.csv"),
            index=False
        )

        energy_export.to_csv(
            os.path.join(args.results_dir, "fig9_energy_distribution.csv"),
            index=False
        )

        print("\n===== FIGURE 9 DATA =====")
        print("Saved: fig9_latency_distribution.csv")
        print("Saved: fig9_energy_distribution.csv")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(dqn_hist["cum_timesteps"], dqn_hist["r"], marker="o", markersize=3,
                label="DQN", color="#4c72b0", alpha=0.8)
        ax.plot(ppo_hist["cum_timesteps"], ppo_hist["r"], marker="o", markersize=3,
                label="PPO (proposed)", color="#55a868", alpha=0.8)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel("Episode reward")
        ax.set_title("Training reward curves: PPO vs DQN")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.results_dir, "fig8_training_curves.png"), dpi=150)
        plt.close(fig)
        print(f"\nTraining curves: PPO {len(ppo_hist)} episodes, DQN {len(dqn_hist)} episodes "
              f"-> fig8_training_curves.png")
    else:
        print(f"\n[fig8 skipped] Monitor logs not found in {args.model_dir} "
              f"(expected dqn_monitor.monitor.csv and ppo_monitor_logs/). "
              f"Re-run train_agents.py with the current version to generate them, "
              f"then re-run evaluate.py.")

    # ---------------- Figure 9: per-task latency/energy distributions ----------------
    # Every other figure reports a MEAN or a SUM. This is the one figure
    # that shows spread: two methods can have near-identical means (as
    # DQN and PPO do here) while differing in variance or tail behaviour,
    # and a reader can't tell that from Table II alone. Built from the
    # same per-task `raw` DataFrame summarize() already keeps -- no new
    # logging needed, just not discarding it into an aggregate.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lat_data = [results[n]["raw"]["latency_s"].values for n in METHOD_NAMES]
    eng_data = [results[n]["raw"]["energy_w"].values for n in METHOD_NAMES]
    for ax, data, ylabel, title in [
        (axes[0], lat_data, "Per-task latency (s)", "Latency distribution"),
        (axes[1], eng_data, "Per-task energy (W)", "Energy distribution"),
    ]:
        try:
            bp = ax.boxplot(data, tick_labels=METHOD_NAMES, patch_artist=True, showfliers=True)
        except TypeError:  # matplotlib < 3.9 doesn't have tick_labels yet
            bp = ax.boxplot(data, labels=METHOD_NAMES, patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], METHOD_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Per-task latency and energy distributions (not just the mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig9_latency_energy_distribution.png"), dpi=150)
    plt.close(fig)
    for name in METHOD_NAMES:
        lat = results[name]["raw"]["latency_s"]
        print(f"[fig9] {name}: latency median={lat.median():.4f}s "
              f"IQR=[{lat.quantile(0.25):.4f}, {lat.quantile(0.75):.4f}] "
              f"p95={lat.quantile(0.95):.4f}")

    # ---------------- Figure 10: BiLSTM forecast residuals over time ----------------
    # Reuses the SAME preds/actual arrays computed for Figure 3 above --
    # no re-fitting, no new numbers, just a different view of them. The
    # single MAE number in Figure 3's caption can't show whether error is
    # uniform or concentrated at the load spikes the Related Work section
    # specifically flags as characteristic of these traces; this can.
    residuals = preds - actual
    # Export Figure 10 residuals
    residual_df = pd.DataFrame({
        "Timestep": np.arange(len(residuals)),
        "Residual": residuals
    })

    residual_df.to_csv(
        os.path.join(args.results_dir, "fig10_residuals.csv"),
        index=False
    )

    print("\n===== FIGURE 10 DATA =====")
    print(residual_df.head())
    print("Saved: fig10_residuals.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    span = slice(0, min(400, len(residuals)))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(residuals[span], color="#c44e52", linewidth=1, label="Forecast error (pred - actual)")
    ax.fill_between(range(len(residuals[span])), residuals[span], 0, color="#c44e52", alpha=0.2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Residual (CPU utilisation)")
    ax.set_title(f"BiLSTM forecast residuals over time (MAE={np.mean(np.abs(residuals)):.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig10_forecast_residuals.png"), dpi=150)
    plt.close(fig)
    print(f"[fig10] Residual std={np.std(residuals):.4f}, max abs residual={np.max(np.abs(residuals)):.4f}")

    print(f"\nAll figures + comparison_table.csv written to {args.results_dir}")


if __name__ == "__main__":
    main()
