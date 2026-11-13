"""
train_agents.py
================
Trains:
  * the proposed PPO agent (Section 4.4), and
  * a reactive DQN agent, as the second published-style baseline
    (Choppara & Mangalampalli 2025 use DQN; we reproduce the *idea*,
    not their exact code, since that wasn't released).

Both agents see the identical FogCloudEnv (same reward, same state) so
the comparison is apples-to-apples -- the ONLY thing that differs is the
learning algorithm. That is the correct way to claim "PPO beats DQN":
if the environments differed, the comparison would be meaningless.

Usage:
    python train_agents.py --timesteps 50000 --trace-path /path/to/real_borg.csv
    python train_agents.py --timesteps 50000        # falls back to synthetic trace
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_gen import (
    synthetic_borg_like_trace, synthetic_task_stream,
    load_borg_trace, load_borg_tasks_and_trace,
)
from qos_classifier import train_qos_classifier, label_priority
from forecasting import LoadForecastingPipeline, WINDOW
from fog_cloud_env import FogCloudEnv

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
try:
    from stable_baselines3.common.utils import LinearSchedule
except ImportError:  # older SB3 versions
    from stable_baselines3.common.utils import get_linear_fn as LinearSchedule


def build_dataset(trace_path: str | None, n_tasks: int, seed: int, bilstm_epochs: int,
                   use_real_tasks: bool = False, eval_fraction: float = 0.25):
    if trace_path and use_real_tasks:
        # Derive BOTH the CPU trace and the task stream (deadline/payload/
        # RAM) from the real file -- stronger than synthetic tasks, but
        # sensor-type labels are still a synthetic proxy (see docstring
        # in data_gen.load_borg_tasks_and_trace).
        tasks, cpu_trace_series = load_borg_tasks_and_trace(trace_path)
        cpu_trace = cpu_trace_series.values
        tasks = tasks.iloc[:n_tasks].reset_index(drop=True)
        n_tasks = len(tasks)
        print(f"[data] Loaded REAL Borg trace + REAL task metadata: "
              f"{len(cpu_trace)} trace points, {n_tasks} tasks from {trace_path}")
    elif trace_path:
        cpu_trace = load_borg_trace(trace_path).values
        tasks = synthetic_task_stream(n_tasks, seed=seed)
        print(f"[data] Loaded REAL Borg trace ({len(cpu_trace)} points) but SYNTHETIC "
              f"task stream from {trace_path}. Pass --use-real-tasks for real task metadata too.")
    else:
        cpu_trace = synthetic_borg_like_trace(max(n_tasks + WINDOW + 50, 3000), seed=seed).values
        tasks = synthetic_task_stream(n_tasks, seed=seed)
        print("[data] WARNING: using SYNTHETIC Borg-like trace (no --trace-path given). "
              "Do not report these numbers as being from the real Google trace.")

    if len(cpu_trace) < WINDOW + 20:
        raise ValueError(
            f"Trace only has {len(cpu_trace)} points after cleaning/resampling -- "
            f"need at least {WINDOW + 20}. Your CSV sample may be too small or too "
            f"heavily filtered by failed/event; export a larger time range."
        )

    # ---- Leak-free protocol -------------------------------------------------
    # Tasks:  indices [0, n_train_tasks)            -> PPO/DQN training
    #         indices [n_train_tasks, n_tasks)      -> held-out evaluation ONLY
    # Trace:  chronological 70/15/15 train/val/test split. The forecaster
    #         (and its Yeo-Johnson transform) only ever sees train/val.
    #         Training tasks get forecasts from windows in train+val;
    #         evaluation tasks get forecasts from windows in the TEST
    #         segment, so the forecast the agents see at evaluation time
    #         is itself genuinely held-out.
    n_eval = int(n_tasks * eval_fraction)
    n_train_tasks = n_tasks - n_eval
    forecaster = LoadForecastingPipeline().fit_split(cpu_trace, epochs=bilstm_epochs)
    split = forecaster.split
    te0, te1 = split["test"]
    forecaster.test_metrics = forecaster.evaluate(cpu_trace[te0:te1])
    print(f"[forecast] split (half-open): {split}")
    print(f"[forecast] held-out TEST metrics: {forecaster.test_metrics}")

    qos_clf, acc, f1 = train_qos_classifier(seed=seed)
    tasks["priority"] = qos_clf.predict(tasks)

    # Alignment: task i is given a forecast from the 50-step trace window
    # starting at offset 3*i (wrapping inside its region). The trace is an
    # exogenous, time-indexed load context; it is NOT coupled to the
    # simulated nodes' queues.
    def region_forecast(lo: int, hi: int, j: int) -> float:
        span = hi - lo - WINDOW - 1
        if span <= 0:
            raise ValueError(f"Trace region [{lo},{hi}) too short for window {WINDOW}.")
        start = lo + (j * 3) % span
        return forecaster.predict_next(cpu_trace[start:start + WINDOW])

    train_hi = split["val"][1]
    lpred = [region_forecast(0, train_hi, i) for i in range(n_train_tasks)]
    lpred += [region_forecast(te0, te1, j) for j in range(n_eval)]
    lpred = np.array(lpred, dtype=np.float32)

    if not np.isfinite(lpred).all():
        n_bad = (~np.isfinite(lpred)).sum()
        print(f"[data] WARNING: {n_bad}/{len(lpred)} forecast values were non-finite "
              f"and have been replaced with the trace median. This should be rare after "
              f"the Yeo-Johnson domain-clipping fix in forecasting.py -- if you see this "
              f"a lot, your trace likely has extreme outliers worth inspecting.")
        lpred = np.nan_to_num(lpred, nan=float(np.median(cpu_trace)),
                               posinf=1.0, neginf=0.0)

    return tasks, lpred, forecaster, qos_clf, cpu_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--n-tasks", type=int, default=2000)
    ap.add_argument("--trace-path", type=str, default=None,
                     help="Path to a real Google Borg instance_usage CSV (see README Step 1). "
                          "If omitted, a synthetic fallback trace is used.")
    ap.add_argument("--use-real-tasks", action="store_true",
                     help="Also derive task deadline/payload/RAM from --trace-path instead of "
                          "synthetic_task_stream (requires --trace-path). Sensor-type labels "
                          "remain a synthetic proxy since Borg has no IoT sensor semantics.")
    ap.add_argument("--bilstm-epochs", type=int, default=20)
    ap.add_argument("--eval-fraction", type=float, default=0.25,
                     help="Fraction of tasks (from the END of the stream) held out for evaluation and "
                          "NEVER used to train PPO/DQN. Must match evaluate.py.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="../models")
    ap.add_argument("--ppo-timesteps", type=int, default=None,
                     help="Override timesteps for PPO only (defaults to --timesteps). PPO is "
                          "on-policy and typically needs more environment interactions than "
                          "off-policy DQN to reach a comparable policy on the same task -- if "
                          "you use this, report the two budgets separately rather than as an "
                          "equal-budget comparison.")
    ap.add_argument("--ent-coef", type=float, default=0.01,
                     help="PPO entropy bonus. Higher = more exploration, which helps escape "
                          "a degenerate 'always pick node X' policy early in training.")
    ap.add_argument("--ppo-n-envs", type=int, default=1,
                     help="Number of parallel copies of the env PPO collects rollouts from "
                          "before each update. More copies = more diverse trajectories per "
                          "gradient update (lower variance) but FEWER updates for a given "
                          "--timesteps budget, since PPO needs n_envs*n_steps samples before "
                          "its first update. In testing, n_envs=1 converged faster at moderate "
                          "budgets (~40k steps); n_envs=4 was still catching up at the same "
                          "budget and may need a larger --timesteps to pay off. Defaulting to "
                          "1; try 4 if you're training for 100k+ steps. This does NOT change "
                          "the total --timesteps budget (SB3 counts steps across all copies "
                          "combined), so either choice stays a fair equal-budget comparison "
                          "with DQN.")
    args = ap.parse_args()

    if args.use_real_tasks and not args.trace_path:
        ap.error("--use-real-tasks requires --trace-path")

    os.makedirs(args.out_dir, exist_ok=True)

    tasks, lpred, forecaster, qos_clf, cpu_trace = build_dataset(
        args.trace_path, args.n_tasks, args.seed, args.bilstm_epochs,
        use_real_tasks=args.use_real_tasks, eval_fraction=args.eval_fraction,
    )

    # Agents are trained ONLY on the first (1 - eval_fraction) of the stream.
    n_eval = int(len(tasks) * args.eval_fraction)
    n_train = len(tasks) - n_eval
    train_tasks = tasks.iloc[:n_train].reset_index(drop=True)
    train_lpred = lpred[:n_train]
    assert len(train_tasks) == n_train and len(train_lpred) == n_train
    print(f"[split] PPO/DQN train on tasks 0..{n_train-1} ({n_train}); "
          f"held-out eval = tasks {n_train}..{len(tasks)-1} ({n_eval})")

    def make_env(monitor_path: str | None = None):
        return Monitor(FogCloudEnv(train_tasks, train_lpred, seed=args.seed), filename=monitor_path)

    # ---------------- PPO (proposed) ----------------
    # Tuning notes (why these differ from the first-pass defaults):
    #  - learning_rate 3e-4 with linear decay to 1e-5: SB3's own benchmark
    #    default for discrete-action PPO; 1e-4 flat (the earlier setting)
    #    is conservative and was likely under-converged at 50k steps.
    #  - ent_coef=0.01: without an entropy bonus PPO can collapse onto a
    #    single action too early (e.g. "always Cloud" or "always Fog-2")
    #    and never discover the better mixed policy DQN's epsilon-greedy
    #    exploration stumbles into more easily.
    #  - n_epochs=10, gae_lambda=0.95: SB3 discrete-control defaults,
    #    more stable advantage estimates than the previous implicit
    #    n_epochs=10/gae_lambda=0.95 (unchanged, listed for clarity).
    #  - net_arch 128x128 (vs SB3's 64x64 default): a little more capacity
    #    to represent the queue-state / forecast interaction; cheap to try
    #    smaller again if you see overfitting on a small task stream.
    #  - clip_range schedule 0.2 -> 0.1: smaller trust-region late in
    #    training once the policy is already decent, to avoid destructive
    #    large updates that can undo earlier progress.
    #  - target_kl=0.03: SB3 aborts an epoch early if the policy update
    #    would move too far from the old policy in one step -- a standard
    #    PPO stabiliser that prevents the occasional catastrophic update.
    #  - n_envs=4 parallel rollout collectors (DummyVecEnv): more diverse
    #    trajectories per gradient update at the SAME total --timesteps
    #    budget (SB3 counts steps across all copies combined, so this
    #    doesn't change the fairness of the PPO-vs-DQN comparison).
    ppo_lr_schedule = LinearSchedule(start=3e-4, end=1e-5, end_fraction=1.0)
    ppo_clip_schedule = LinearSchedule(start=0.2, end=0.1, end_fraction=1.0)
    # Monitor writes one row per completed episode (episode reward, length,
    # wall-clock time) to a CSV -- this is what evaluate.py's training-curve
    # figure reads. One log file per parallel env, so nothing gets
    # clobbered if --ppo-n-envs > 1.
    ppo_monitor_dir = os.path.join(args.out_dir, "ppo_monitor_logs")
    os.makedirs(ppo_monitor_dir, exist_ok=True)
    ppo_env = DummyVecEnv([
        (lambda i=i: make_env(os.path.join(ppo_monitor_dir, f"env{i}")))
        for i in range(args.ppo_n_envs)
    ])
    ppo = PPO(
        "MlpPolicy", ppo_env,
        learning_rate=ppo_lr_schedule, gamma=0.99, gae_lambda=0.95,
        clip_range=ppo_clip_schedule, ent_coef=args.ent_coef, vf_coef=0.5,
        max_grad_norm=0.5, target_kl=0.03,
        n_steps=2048, batch_size=256, n_epochs=10,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=1, seed=args.seed,
    )
    print(f"\n=== Training PPO (proposed Hybrid BiLSTM-PPO agent), n_envs={args.ppo_n_envs} ===")
    ppo.learn(total_timesteps=args.ppo_timesteps or args.timesteps)
    ppo.save(os.path.join(args.out_dir, "ppo_agent"))

    # ---------------- DQN (reactive baseline) ----------------
    # Given matching net_arch to PPO (128x128) so neither agent has a
    # capacity advantage -- the comparison should isolate the learning
    # algorithm, not accidentally isolate network size instead.
    dqn_env = make_env(os.path.join(args.out_dir, "dqn_monitor"))
    dqn = DQN(
        "MlpPolicy", dqn_env, learning_rate=1e-4, gamma=0.99,
        buffer_size=20000, learning_starts=500, batch_size=64,
        target_update_interval=500, policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1, seed=args.seed,
    )
    print("\n=== Training DQN (reactive baseline) ===")
    dqn.learn(total_timesteps=args.timesteps)
    dqn.save(os.path.join(args.out_dir, "dqn_agent"))

    tasks.to_parquet(os.path.join(args.out_dir, "eval_tasks.parquet")) \
        if False else tasks.to_csv(os.path.join(args.out_dir, "eval_tasks.csv"), index=False)
    np.save(os.path.join(args.out_dir, "eval_lpred.npy"), lpred)
    np.save(os.path.join(args.out_dir, "eval_cpu_trace.npy"), cpu_trace)
    with open(os.path.join(args.out_dir, "forecaster.pkl"), "wb") as fh:
        pickle.dump(forecaster, fh)
    split_info = dict(
        seed=args.seed, n_tasks=len(tasks),
        train_task_indices=[0, n_train - 1], eval_task_indices=[n_train, len(tasks) - 1],
        trace_len=int(len(cpu_trace)),
        trace_split_half_open={k: list(v) for k, v in forecaster.split.items()},
        forecaster_best_val_mse=forecaster.best_val_mse,
        forecaster_test_metrics=forecaster.test_metrics,
    )
    with open(os.path.join(args.out_dir, "split_info.json"), "w") as fh:
        json.dump(split_info, fh, indent=2)
    print("Split info:", json.dumps(split_info, indent=2))

    print(f"\nSaved: ppo_agent.zip, dqn_agent.zip, eval_tasks.csv, eval_lpred.npy, "
          f"ppo_monitor_logs/, dqn_monitor.monitor.csv -> {args.out_dir}")


if __name__ == "__main__":
    main()