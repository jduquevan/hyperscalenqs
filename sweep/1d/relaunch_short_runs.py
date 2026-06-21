"""
Identify W&B runs whose runtime is below a threshold, optionally delete them,
and emit (or submit) targeted sbatch commands to relaunch only those seeds.

Dry-run by default — no deletions or job submissions without explicit flags.

Usage
-----
# Inspect what would be relaunched:
python sweep/relaunch_short_runs.py

# Delete short runs from W&B AND submit sbatch jobs:
python sweep/relaunch_short_runs.py --delete --submit

# Custom project / threshold:
python sweep/relaunch_short_runs.py --project my-project --threshold 7200
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import NamedTuple

import wandb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 3 h 55 min in seconds
DEFAULT_THRESHOLD_SECS = 3 * 3600 + 55 * 60  # 14 100

DEFAULT_PROJECT = "scaling-system"

# Maps run-name method token → (slurm file, extra positional args)
_SLURM = {
    "adam":   ("sweep/1d/run_adam.slurm",  ["j1j2"]),
    "minsr":  ("sweep/1d/run_minsr.slurm", ["j1j2", "minsr"]),
    "spring": ("sweep/1d/run_minsr.slurm", ["j1j2", "spring"]),
    "qps":    ("sweep/1d/run_qps.slurm",   ["j1j2"]),
}

# Model-size presets (mirrors run_scaling_model.sh)
_MODEL_SIZES: dict[str, dict[str, str]] = {
    "tiny":   {"N_GRU_LAYERS": "1", "RNN_HIDDEN": "64",  "HEAD_HIDDEN": "64"},
    "small":  {"N_GRU_LAYERS": "2", "RNN_HIDDEN": "128", "HEAD_HIDDEN": "128"},
    "medium": {"N_GRU_LAYERS": "3", "RNN_HIDDEN": "256", "HEAD_HIDDEN": "256"},
}

# Base env shared by model- and system-scaling experiments
_BASE_ENV = {
    "CHAIN_N": "12",
    "N_GRU_LAYERS": "3",
    "RNN_HIDDEN": "256",
    "HEAD_HIDDEN": "256",
    "TRANSITION_STEPS": "40000",
}


def tag_to_env(tag: str) -> dict[str, str]:
    """Map a RUN_TAG to the env-var overrides needed to relaunch that run.

    Supported tag formats (one per scaling experiment):
      n<digits>  — scaling-samples  (e.g. n512, n1024, n2048)
      tiny/small/medium — scaling-model
      N<digits>  — scaling-system   (e.g. N8, N16, N32)
    """
    # scaling-samples: n512, n1024, n2048
    m = re.fullmatch(r"n(\d+)", tag)
    if m:
        return {**_BASE_ENV, "N_SAMPLES": m.group(1)}
    # scaling-model: tiny, small, medium
    if tag in _MODEL_SIZES:
        return {"N_SAMPLES": "2048", "CHAIN_N": "12", "TRANSITION_STEPS": "40000", **_MODEL_SIZES[tag]}
    # scaling-system: N8, N16, N32
    m = re.fullmatch(r"N(\d+)", tag)
    if m:
        return {"N_SAMPLES": "2048", "N_GRU_LAYERS": "3", "RNN_HIDDEN": "256",
                "HEAD_HIDDEN": "256", "TRANSITION_STEPS": "40000", "CHAIN_N": m.group(1)}
    raise ValueError(f"Unrecognised RUN_TAG: {tag!r}")


# Run-name regex — works for all three scaling experiments:
#   scaling-samples:  adam_j1j2_n512_seed_3
#   scaling-model:    adam_j1j2_tiny_seed_3
#   scaling-system:   adam_j1j2_N8_seed_3
_RUN_RE = re.compile(
    r"^(?P<method>adam|minsr|spring|qps)_j1j2_(?P<tag>[a-zA-Z0-9]+)_seed_(?P<seed>\d+)$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RunKey(NamedTuple):
    method: str
    tag: str


def get_runtime(run) -> float:
    """Return the runtime in seconds for a W&B run, or 0 if unknown."""
    # run.summary['_runtime'] is the canonical field W&B populates on crash/finish
    val = run.summary.get("_runtime", None)
    if val is not None:
        return float(val)
    # Fall back to wall-clock from timestamps
    if run.created_at and run.heartbeat_at:
        from datetime import datetime, timezone
        fmt = "%Y-%m-%dT%H:%M:%S"
        try:
            created = datetime.strptime(run.created_at[:19], fmt).replace(tzinfo=timezone.utc)
            heartbeat = datetime.strptime(run.heartbeat_at[:19], fmt).replace(tzinfo=timezone.utc)
            return (heartbeat - created).total_seconds()
        except ValueError:
            pass
    return 0.0


def fmt_duration(secs: float) -> str:
    h = int(secs) // 3600
    m = (int(secs) % 3600) // 60
    s = int(secs) % 60
    return f"{h}h {m:02d}m {s:02d}s"


def build_sbatch_cmd(key: RunKey, seeds: list[int], project: str) -> str:
    slurm_file, positional = _SLURM[key.method]
    array_spec = ",".join(str(s) for s in sorted(seeds))
    env_pairs = {
        "RUN_TAG": key.tag,
        "WANDB_PROJECT": project,
        **tag_to_env(key.tag),
    }
    env_str = " ".join(f"{k}={v}" for k, v in env_pairs.items())
    positional_str = " ".join(positional)
    return f"{env_str} sbatch --array={array_spec} {slurm_file} {positional_str}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project",   default=DEFAULT_PROJECT, help="W&B project name")
    parser.add_argument("--entity",    default=None,            help="W&B entity (user/org); uses default if omitted")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_SECS,
                        help=f"Runtime threshold in seconds (default: {DEFAULT_THRESHOLD_SECS} = 3h 55m)")
    parser.add_argument("--delete",    action="store_true", help="Delete short runs from W&B (asks for confirmation)")
    parser.add_argument("--submit",    action="store_true", help="Actually call sbatch (default: dry-run only)")
    args = parser.parse_args()

    project_path = f"{args.entity}/{args.project}" if args.entity else args.project

    print(f"Querying W&B project: {project_path}")
    print(f"Runtime threshold   : {fmt_duration(args.threshold)}\n")

    api = wandb.Api()
    all_runs = list(api.runs(project_path))
    print(f"Total runs found    : {len(all_runs)}")

    # Filter: short runtime AND name matches expected pattern
    short_runs = []
    unmatched_short = []
    for run in all_runs:
        rt = get_runtime(run)
        if rt < args.threshold:
            m = _RUN_RE.match(run.name)
            if m:
                short_runs.append((run, m, rt))
            else:
                unmatched_short.append((run, rt))

    if not short_runs and not unmatched_short:
        print("No short runs found. Nothing to do.")
        return

    # -----------------------------------------------------------------------
    # Report unmatched short runs (won't be relaunched)
    # -----------------------------------------------------------------------
    if unmatched_short:
        print(f"\nShort runs NOT matching expected name pattern ({len(unmatched_short)}) — skipped:")
        for run, rt in unmatched_short:
            print(f"  {run.name:<50}  {fmt_duration(rt)}")

    # -----------------------------------------------------------------------
    # Group matched short runs
    # -----------------------------------------------------------------------
    groups: dict[RunKey, list[tuple]] = defaultdict(list)
    for run, m, rt in short_runs:
        key = RunKey(method=m.group("method"), tag=m.group("tag"))
        groups[key].append((run, int(m.group("seed")), rt))

    print(f"\nShort runs to relaunch: {len(short_runs)}")
    print(f"{'Run name':<50}  {'Runtime':>12}  {'State'}")
    print("-" * 75)
    for run, m, rt in sorted(short_runs, key=lambda x: x[0].name):
        print(f"  {run.name:<48}  {fmt_duration(rt):>12}  {run.state}")

    # -----------------------------------------------------------------------
    # Delete from W&B
    # -----------------------------------------------------------------------
    if args.delete:
        print(f"\nAbout to DELETE {len(short_runs)} runs from W&B project '{project_path}'.")
        answer = input("Type 'yes' to confirm: ").strip().lower()
        if answer != "yes":
            print("Deletion cancelled.")
            sys.exit(0)
        for run, _m, _rt in short_runs:
            run.delete()
            print(f"  Deleted: {run.name}")
        print("Deletion complete.\n")
    else:
        print("\n(Pass --delete to remove these runs from W&B.)")

    # -----------------------------------------------------------------------
    # Build and emit/submit sbatch commands
    # -----------------------------------------------------------------------
    print("\nsbatch commands to relaunch:")
    print("-" * 75)
    cmds = []
    for key in sorted(groups.keys()):
        seeds = [seed for _run, seed, _rt in groups[key]]
        cmd = build_sbatch_cmd(key, seeds, args.project)
        print(f"  {cmd}")
        cmds.append(cmd)

    if args.submit:
        print("\nSubmitting jobs...")
        # Ensure we run from the repo root so relative paths in slurm files resolve correctly
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for cmd in cmds:
            result = subprocess.run(cmd, shell=True, cwd=repo_root, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  OK  : {result.stdout.strip()}")
            else:
                print(f"  ERR : {result.stderr.strip()}", file=sys.stderr)
    else:
        print("\n(Pass --submit to submit the sbatch commands above.)")


if __name__ == "__main__":
    main()
