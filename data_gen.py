"""
data_gen.py
============
Two responsibilities:

1. `load_borg_trace(path)` / `load_borg_tasks_and_trace(path)` — load a REAL
   Google Borg Cluster Trace 2019 `instance_usage` extract (see README,
   Step 1). This sandbox has no access to Google Cloud Storage/BigQuery,
   so it could not be downloaded here — you fetch it yourself and point
   these loaders at a local CSV.

2. `synthetic_borg_like_trace(...)` / `synthetic_task_stream(...)` — a
   fallback generator that is NOT the real trace. It reproduces the
   statistical shape reported for Borg 2019 in the literature (heavy
   right skew, bursty spikes) so the pipeline can be developed and
   unit-tested end-to-end without the real file.

   IMPORTANT (read this): any numbers you publish MUST be produced from
   the real trace, not this synthetic generator. Reporting synthetic
   results as if they came from the Google trace would be a factual
   error in your paper.
"""
from __future__ import annotations
import ast
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

SENSOR_TYPES = [
    "thermostat",       # delay tolerant
    "smart_meter",      # delay tolerant
    "industrial_sensor",# mixed
    "camera_feed",      # mixed / large payload
    "av_lidar",         # delay sensitive (autonomous vehicle)
    "health_monitor",   # delay sensitive
]

# ----------------------------------------------------------------------
# 1. Real trace loaders -- schema matches Google's `instance_usage` table
#    (clusterdata-2019, joined with a few `instance_events` columns), which
#    is what your CSV export actually contains. Columns used:
#      start_time            -- window start, microseconds since epoch
#      average_usage         -- dict-like string "{'cpus': .., 'memory': ..}"
#                                normalized-CPU-usage rate for the window
#      resource_request      -- dict-like string, the task's requested
#                                cpus/memory (used as a payload/RAM proxy
#                                if you use load_borg_tasks_and_trace)
#      scheduling_class      -- 0-3, higher = more latency-sensitive
#                                (per Google's trace documentation) --
#                                used as a deadline proxy
#      failed / event        -- rows with failed==1 or event=="FAIL" are
#                                dropped; they don't reflect real steady
#                                -state usage
# ----------------------------------------------------------------------
def _parse_dict_col(series: pd.Series, key: str) -> pd.Series:
    """Borg usage/request columns are stored as Python-dict-repr strings,
    e.g. "{'cpus': 0.0046, 'memory': 0.0059}". Pull one key out safely."""
    def _get(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, dict):
            d = v
        else:
            try:
                d = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return np.nan
        val = d.get(key)
        return np.nan if val is None else float(val)
    return series.apply(_get)


def load_borg_trace(path: str, resample_rule: str = "1min") -> pd.Series:
    """Returns a CPU-utilisation time series (fraction 0-1, already
    normalized-CPU-units per Google's docs) resampled to `resample_rule`,
    built straight from your `instance_usage` CSV export -- no renaming
    needed, point this directly at the file."""
    df = pd.read_csv(path)
    required = {"start_time", "average_usage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trace file is missing expected columns: {missing}")

    if "failed" in df.columns:
        df = df[df["failed"] != 1]
    if "event" in df.columns:
        df = df[df["event"] != "FAIL"]

    df = df.dropna(subset=["start_time"])
    df["cpu_usage"] = _parse_dict_col(df["average_usage"], "cpus")
    df = df.dropna(subset=["cpu_usage"])

    df["ts"] = pd.to_datetime(df["start_time"].astype(np.int64), unit="us")
    s = df.set_index("ts")["cpu_usage"].sort_index()
    s = s.resample(resample_rule).mean().interpolate().clip(lower=0)
    # normalize to [0,1] by the trace's own observed max so the BiLSTM/env
    # (which expect a 0-1 fraction) get a sensible scale regardless of the
    # absolute NCU values in your particular cell/export
    max_val = s.max() if s.max() > 0 else 1.0
    return (s / max_val).clip(0, 1)


def load_borg_tasks_and_trace(path: str, resample_rule: str = "1min",
                               ram_scale_mb: float = 65536.0,
                               payload_scale_kb: float = 200000.0):
    """Optional stronger alternative to synthetic_task_stream(): derives
    the task stream (deadline/payload/ram) from the SAME real CSV instead
    of from synthetic_task_stream, using:
      - scheduling_class (0-3) as a deadline proxy (documented by Google
        as "how latency-sensitive" a task is; we map higher class -> a
        tighter synthetic deadline band, since Borg has no literal
        millisecond deadline field)
      - resource_request.cpus / .memory as payload/RAM proxies, scaled
        to KB/MB by the constants above (tune these to your cell's real
        machine sizes if you know them)
    Sensor-type labels are still synthetic (Borg tasks are anonymised
    production jobs, not IoT sensors) -- say so explicitly in your setup
    section, exactly as EcoTaskSched and the DQN paper both had to.
    Returns (tasks_df, cpu_trace_series).
    """
    df = pd.read_csv(path)
    if "failed" in df.columns:
        df = df[df["failed"] != 1]
    if "event" in df.columns:
        df = df[df["event"] != "FAIL"]
    df = df.dropna(subset=["start_time", "resource_request"]).reset_index(drop=True)

    df["cpus_req"] = _parse_dict_col(df["resource_request"], "cpus")
    df["mem_req"] = _parse_dict_col(df["resource_request"], "memory")
    df = df.dropna(subset=["cpus_req", "mem_req"])

    deadline_bands = {0: (400, 5000), 1: (60, 800), 2: (20, 200), 3: (5, 60)}
    rng = np.random.default_rng(7)
    sched_class = df.get("scheduling_class", pd.Series(0, index=df.index)).clip(0, 3).astype(int)
    lo = sched_class.map(lambda c: deadline_bands[c][0]).to_numpy()
    hi = sched_class.map(lambda c: deadline_bands[c][1]).to_numpy()
    deadline_ms = rng.uniform(lo, hi)

    sensor = np.where(sched_class >= 3, "av_lidar",
              np.where(sched_class == 2, "health_monitor",
              np.where(sched_class == 1, "industrial_sensor", "thermostat")))

    tasks = pd.DataFrame({
        "task_id": np.arange(len(df)),
        "sensor_type": sensor,
        "deadline_ms": deadline_ms,
        "payload_kb": (df["cpus_req"] * payload_scale_kb).to_numpy(),
        "ram_mb": (df["mem_req"] * ram_scale_mb).to_numpy(),
    })

    cpu_trace = load_borg_trace(path, resample_rule=resample_rule)
    return tasks, cpu_trace


# ----------------------------------------------------------------------
# 2. Synthetic fallback (clearly labelled — do not report as real results)
# ----------------------------------------------------------------------
def synthetic_borg_like_trace(n_steps: int = 5000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    # diurnal seasonality + slow drift
    base = 0.30 + 0.12 * np.sin(2 * np.pi * t / 288) + 0.03 * np.sin(2 * np.pi * t / 2016)
    # heavy-tailed bursty noise (lognormal) to mimic Borg's right skew
    noise = rng.lognormal(mean=-2.2, sigma=0.9, size=n_steps)
    spikes = (rng.random(n_steps) < 0.02) * rng.uniform(0.3, 0.6, n_steps)
    trace = np.clip(base + noise + spikes, 0.01, 1.0)
    idx = pd.date_range("2019-05-01", periods=n_steps, freq="1min")
    return pd.Series(trace, index=idx, name="cpu_util")


def synthetic_task_stream(n_tasks: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Generates an IoT task arrival stream with the fields the ID3
    classifier and the environment both need."""
    rng = np.random.default_rng(seed)
    sensor = rng.choice(SENSOR_TYPES, size=n_tasks, p=[0.22, 0.18, 0.18, 0.17, 0.13, 0.12])

    deadline_ms = np.empty(n_tasks)
    payload_kb = np.empty(n_tasks)
    ram_mb = np.empty(n_tasks)

    tight = np.isin(sensor, ["av_lidar", "health_monitor"])
    medium = sensor == "industrial_sensor"
    loose = np.isin(sensor, ["thermostat", "smart_meter", "camera_feed"])

    deadline_ms[tight] = rng.uniform(5, 60, tight.sum())
    deadline_ms[medium] = rng.uniform(60, 400, medium.sum())
    deadline_ms[loose] = rng.uniform(400, 5000, loose.sum())

    payload_kb[sensor == "camera_feed"] = rng.uniform(500, 4000, (sensor == "camera_feed").sum())
    other = sensor != "camera_feed"
    payload_kb[other] = rng.uniform(2, 300, other.sum())

    ram_mb = payload_kb * rng.uniform(0.5, 1.5, n_tasks) + rng.uniform(16, 128, n_tasks)

    df = pd.DataFrame({
        "task_id": np.arange(n_tasks),
        "sensor_type": sensor,
        "deadline_ms": deadline_ms,
        "payload_kb": payload_kb,
        "ram_mb": ram_mb,
    })
    return df


if __name__ == "__main__":
    trace = synthetic_borg_like_trace(200)
    tasks = synthetic_task_stream(10)
    print(trace.head())
    print(tasks.head())
