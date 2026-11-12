"""
sweep_reward_weights.py
=========================
Journal-only addition (Figure idea #4): trains the proposed PPO agent at
several (w_latency, w_energy) reward-weight settings and plots the
resulting (mean latency, total energy) points as a Pareto frontier on
the held-out evaluation split.

This is the one figure among the "add more graphs" batch that is
genuinely NEW evidence rather than a new view of evidence you already
have -- everything else (distributions, node routing, forecast
residuals) is computed from data your existing pipeline already
produces. This one requires actually retraining PPO multiple times, so
budget real wall-clock time for it: with N weight settings and T
timesteps each, this is N x T total PPO training steps, run
sequentially.

The paper's main results (Table II, w_latency=0.6/w_energy=0.4) are
UNCHANGED by running this -- this script only adds new PPO runs at
additional weight settings for the frontier plot; it does not retrain
or overwrite the model your other results depend on (pass a separate
--out-dir, e.g. sweep_models, so nothing in your existing models/
directory is touched).

Usage:
    python sweep_reward_weights.py --trace-path your_export.csv \\
        --timesteps 50000 --n-tasks 2000 --out-dir ../sweep_models \\
        --results-dir ../sweep_results

Default sweep grid is 5 points spanning latency-heavy to energy-heavy;
override with --weight-pairs if you want a different or finer grid,
e.g. --weight-pairs "0.9,0.1 0.7,0.3 0.5,0.5 0.3,0.7 0.1,0.9"
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
from train_agents import build_dataset
from fog_cloud_env import FogCloudEnv
from baselines import run_episode
from evaluate import summarize

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
try:
    from stable_baselines3.common.utils import LinearSchedule
except ImportError:
    from stable_baselines3.common.utils import get_linear_fn as LinearSchedule


DEFAULT_WEIGHT_PAIRS = [(0.9, 0.1), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]


def parse_weight_pairs(s: str) -> list[tuple[float, float]]:
    pairs = []
    for token in s.split():
        w_lat, w_eng = token.split(",")
        pairs.append((float(w_lat), float(w_eng)))
    return pairs


def train_one_weight_setting(tasks, lpred, w_lat, w_eng, timesteps, seed, out_dir):
    """Trains a single PPO agent at the given reward weights and returns
    its held-out evaluation metrics. Mirrors train_agents.py's PPO
    hyperparameters exactly (same schedule, same network size) so weight
    is the only thing that varies across the sweep -- otherwise a
    difference in the frontier could be a hyperparameter artefact
    instead of a genuine latency/energy trade-off."""
    n_eval = int(len(tasks) * 0.25)
    train_tasks, train_lpred = tasks.iloc[:-n_eval], lpred[:-n_eval]
    eval_tasks, eval_lpred = tasks.iloc[-n_eval:].reset_index(drop=True), lpred[-n_eval:]

    def make_env():
        return Monitor(FogCloudEnv(train_tasks, train_lpred, seed=seed,
                                    w_latency=w_lat, w_energy=w_eng))

    ppo = PPO(
        "MlpPolicy", make_env(),
        learning_rate=LinearSchedule(start=3e-4, end=1e-5, end_fraction=1.0),
        gamma=0.99, gae_lambda=0.95,
        clip_range=LinearSchedule(start=0.2, end=0.1, end_fraction=1.0),
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
        n_steps=2048, batch_size=256, n_epochs=10,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=0, seed=seed,
    )
    print(f"\n=== Training PPO at w_latency={w_lat}, w_energy={w_eng} ===")
    ppo.learn(total_timesteps=timesteps)

    tag = f"w{w_lat:.1f}_{w_eng:.1f}".replace(".", "")
    model_path = os.path.join(out_dir, f"ppo_{tag}")
    ppo.save(model_path)

    # Evaluate on the SAME held-out split every other method in the paper
    # uses, so frontier points are comparable to Table II's PPO row.
    eval_env = FogCloudEnv(eval_tasks, eval_lpred, seed=0, w_latency=w_lat, w_energy=w_eng)
    log = run_episode(eval_env, ppo, is_sb3=True)
    metrics = summarize(log)
    return dict(w_latency=w_lat, w_energy=w_eng,
                mean_latency_s=metrics["mean_latency_s"],
                total_energy_w=metrics["total_energy_w"],
                qos_violation_rate=metrics["qos_violation_rate"],
                crash_violation_rate=metrics["crash_violation_rate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50000,
                     help="Timesteps PER weight setting. Total training time scales "
                          "with (timesteps x number of weight pairs) -- keep this lower "
                          "than your main run's budget unless you have time for "
                          "len(weight-pairs)x the wall-clock cost.")
    ap.add_argument("--n-tasks", type=int, default=2000)
    ap.add_argument("--trace-path", type=str, default=None)
    ap.add_argument("--bilstm-epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="../sweep_models",
                     help="Separate from --out-dir used by train_agents.py, so this "
                          "sweep never overwrites the models your main results use.")
    ap.add_argument("--results-dir", type=str, default="../sweep_results")
    ap.add_argument("--weight-pairs", type=str, default=None,
                     help='Space-separated w_latency,w_energy pairs, e.g. '
                          '"0.9,0.1 0.5,0.5 0.1,0.9". Defaults to a 5-point sweep.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    weight_pairs = (parse_weight_pairs(args.weight_pairs) if args.weight_pairs
                     else DEFAULT_WEIGHT_PAIRS)
    for w_lat, w_eng in weight_pairs:
        if abs(w_lat + w_eng - 1.0) > 1e-6:
            print(f"[warning] w_latency={w_lat} + w_energy={w_eng} != 1.0 -- "
                  f"the paper's convention normalizes to 1.0, but the env doesn't "
                  f"require it. Proceeding as given.")

    tasks, lpred, forecaster, qos_clf, cpu_trace = build_dataset(
        args.trace_path, args.n_tasks, args.seed, args.bilstm_epochs,
        use_real_tasks=False,
    )

    rows = []
    for w_lat, w_eng in weight_pairs:
        rows.append(train_one_weight_setting(
            tasks, lpred, w_lat, w_eng, args.timesteps, args.seed, args.out_dir
        ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, "sweep_results.csv"), index=False)
    print("\n" + df.to_string(index=False))

    # ---------------- Pareto frontier plot ----------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(df["mean_latency_s"], df["total_energy_w"], "o-", color="#4c72b0", markersize=8)
    for _, row in df.iterrows():
        ax.annotate(f"({row['w_latency']:.1f}, {row['w_energy']:.1f})",
                     (row["mean_latency_s"], row["total_energy_w"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Mean latency (s)")
    ax.set_ylabel("Total energy (W)")
    ax.set_title("PPO latency-energy Pareto frontier across reward-weight settings\n"
                  "(labels: (w_latency, w_energy))")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig_pareto_frontier.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved sweep_results.csv and fig_pareto_frontier.png -> {args.results_dir}")
    print("Reminder: Table II in the paper reports the w_latency=0.6/w_energy=0.4 "
          "setting from your main train_agents.py run -- this sweep is additive, "
          "not a replacement for it.")


if __name__ == "__main__":
    main()
