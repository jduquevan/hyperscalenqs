# ── Install dependencies (safe to re-run) ────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "wandb", "matplotlib", "python-dotenv", "-q"])

# ── Imports ───────────────────────────────────────────────────────────────────
import os, re, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np
import wandb
from dotenv import load_dotenv

# Load WANDB_API_KEY from a .env file if present (upload it to Colab, or skip
# and set WANDB_API_KEY manually below).
load_dotenv(Path(".env"))

# ── Settings — edit these ─────────────────────────────────────────────────────
WANDB_ENTITY  = None            # your wandb username or org, e.g. "myname"
WANDB_PROJECT = "hyperscalenqs"
OUT_PATH      = "scaling_plot.pdf"

# ─────────────────────────────────────────────────────────────────────────────

# Architecture constants (same for both methods)
EMBED_DIM = 32   # E
N_SITES   = 12   # N  (overridden by run config when available)

# Size label → (rnn_hidden, head_hidden, n_gru_layers)
SIZE_CONFIGS = {
    "normal": (256, 256, 3),
    "small":  (128, 128, 2),
    "tiny":   (64,  64,  1),
}
SIZE_ORDER = ["tiny", "small", "normal"]

# Thresholds for time-to-accuracy plot (right panel)
THRESHOLDS        = [1e-3, 1e-4, 1e-5]
THRESHOLD_MARKERS = {1e-3: "o", 1e-4: "s", 1e-5: "^"}
THRESHOLD_LABELS  = {1e-3: r"$10^{-3}$", 1e-4: r"$10^{-4}$", 1e-5: r"$10^{-5}$"}

# Time budgets for accuracy-vs-params plot (left panel), in seconds
TIME_BUDGETS        = [30 * 60, 60 * 60, 2 * 60 * 60]   # 30 min, 1 h, 2 h
TIME_BUDGET_MARKERS = {30 * 60: "o", 60 * 60: "s", 2 * 60 * 60: "^"}
TIME_BUDGET_LABELS  = {30 * 60: "30 min", 60 * 60: "1 h", 2 * 60 * 60: "2 h"}

METHOD_COLORS = {"minsr": "#e05a2b", "qps": "#3d7ec8"}
METHOD_LABELS = {"minsr": "minSR (SPRING)", "qps": "QPS"}

SIZE_LINESTYLES = {"tiny": ":", "small": "--", "normal": "-"}
SIZE_LABELS     = {"tiny": "tiny", "small": "small", "normal": "normal"}


# ── Parameter count formula ───────────────────────────────────────────────────

def compute_n_params(H: int, D: int, L: int, E: int = 32, N: int = 12) -> int:
    """
    Analytical parameter count for ComplexRecurrentAR (split amp/phase towers + LayerNorm).

      tok_emb    : 3*E
      pos_emb    : N*E
      input_proj : E*H + H
      input_ln   : 2*H
      GRU x L    : L * (6*H^2 + 6*H)
      amp tower  : (H*D + D) + (D^2 + D) + (2*D + 2)
      phase tower: identical to amp tower
    """
    emb       = (3 + N) * E
    proj      = E * H + H
    ln        = 2 * H
    gru       = L * (6 * H * H + 6 * H)
    one_tower = (H * D + D) + (D * D + D) + (2 * D + 2)
    return emb + proj + ln + gru + 2 * one_tower


# ── Run name parser ───────────────────────────────────────────────────────────

def parse_run_name(name: str):
    """Return (method, size_label) for matching runs, else None."""
    m = re.fullmatch(r"(minsr|qps)_ising_seed_\d+(?:_(tiny|small))?", name or "")
    if m is None:
        return None
    return m.group(1), (m.group(2) or "normal")


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_runs(entity, project):
    api  = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    print(f"Fetching runs from '{path}' ...")

    collected = []
    for run in api.runs(path):
        parsed = parse_run_name(run.name)
        if parsed is None:
            print(f"  [skip] {run.name!r}")
            continue

        method, size_label = parsed
        H0, D0, L0 = SIZE_CONFIGS[size_label]
        cfg = run.config
        E   = int(cfg.get("embed_dim",    EMBED_DIM))
        N   = int(cfg.get("N",            N_SITES))
        H   = int(cfg.get("rnn_hidden",   H0))
        D   = int(cfg.get("head_hidden",  D0))
        L   = int(cfg.get("n_gru_layers", L0))

        n_params = compute_n_params(H, D, L, E, N)

        try:
            history = run.history(
                keys=["eval_rel_error_exact", "_runtime"],
                x_axis="_step",
                pandas=True,
            )
        except Exception as exc:
            warnings.warn(f"  [warn] {run.name}: {exc}")
            history = None

        n_steps = len(history) if history is not None else "N/A"
        print(f"  [ok]   {run.name:<42} n_params={n_params:>9,}  steps={n_steps}")
        collected.append(dict(
            method=method, size_label=size_label,
            seed=int(cfg.get("seed", -1)),
            n_params=n_params, history=history,
        ))

    return collected


# ── Metric extraction ─────────────────────────────────────────────────────────

def error_at_time(history, t_limit: float) -> float:
    """Last eval_rel_error_exact logged at or before t_limit seconds of runtime."""
    if history is None:
        return float("nan")
    if "eval_rel_error_exact" not in history.columns or "_runtime" not in history.columns:
        return float("nan")
    subset = history[history["_runtime"] <= t_limit]["eval_rel_error_exact"].dropna()
    return float(subset.iloc[-1]) if len(subset) else float("nan")


def time_to_threshold(history, threshold: float) -> float:
    if history is None:
        return float("nan")
    if "eval_rel_error_exact" not in history.columns or "_runtime" not in history.columns:
        return float("nan")
    mask = history["eval_rel_error_exact"] < threshold
    if not mask.any():
        return float("nan")
    t = history.loc[history.index[mask][0], "_runtime"]
    return float(t) if t is not None else float("nan")


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(collected):
    data = {}
    for rec in collected:
        method, size_label = rec["method"], rec["size_label"]
        data.setdefault(method, {})
        data[method].setdefault(size_label, {
            "n_params":       rec["n_params"],
            "errors_at_time": {t: [] for t in TIME_BUDGETS},
            "times":          {t: [] for t in THRESHOLDS},
        })
        b = data[method][size_label]
        for tb in TIME_BUDGETS:
            b["errors_at_time"][tb].append(error_at_time(rec["history"], tb))
        for thr in THRESHOLDS:
            b["times"][thr].append(time_to_threshold(rec["history"], thr))

    for method, sizes in data.items():
        for b in sizes.values():
            b["errors_at_time"] = {tb: np.array(v, dtype=float)
                                   for tb, v in b["errors_at_time"].items()}
            b["times"]         = {thr: np.array(v, dtype=float)
                                   for thr, v in b["times"].items()}
    return data


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(data):
    tb_2h = 2 * 60 * 60
    print("\n-- Run summary (error at 2 h) " + "-" * 43)
    print(f"{'Method':<20} {'Size':<8} {'n_params':>10}  {'N seeds':>7}  "
          f"{'mean err @ 2h':>16}  {'std':>12}")
    print("-" * 80)
    for method in sorted(data):
        for size in SIZE_ORDER:
            if size not in data[method]:
                continue
            b     = data[method][size]
            valid = b["errors_at_time"][tb_2h]
            valid = valid[~np.isnan(valid)]
            mean  = np.mean(valid) if len(valid) else float("nan")
            std   = np.std(valid)  if len(valid) > 1 else float("nan")
            print(f"{METHOD_LABELS[method]:<20} {size:<8} {b['n_params']:>10,}  "
                  f"{len(valid):>7}  {mean:>16.3e}  {std:>12.3e}")
    print("-" * 80)


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_figure(data, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: error at time budget vs n_params
    method_handles_left = []
    for method in ("minsr", "qps"):
        if method not in data:
            continue
        c = METHOD_COLORS[method]
        method_handles_left.append(
            matplotlib.lines.Line2D([], [], color=c, lw=2, label=METHOD_LABELS[method])
        )
        for tb in TIME_BUDGETS:
            xs, means, stds = [], [], []
            for size in SIZE_ORDER:
                if size not in data[method]:
                    continue
                valid = data[method][size]["errors_at_time"][tb]
                valid = valid[~np.isnan(valid)]
                if not len(valid):
                    continue
                xs.append(data[method][size]["n_params"])
                means.append(np.mean(valid))
                stds.append(np.std(valid) if len(valid) > 1 else 0.0)
            if not xs:
                continue
            xs, means, stds = np.array(xs), np.array(means), np.array(stds)
            ax1.plot(xs, means, marker=TIME_BUDGET_MARKERS[tb], ls="-",
                     color=c, markersize=8, zorder=3)
            ax1.errorbar(xs, means, yerr=stds, fmt="none", color=c, capsize=4, zorder=2)

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Number of parameters")
    ax1.set_ylabel(r"$\epsilon_{\rm rel}$ at time budget")
    ax1.set_title("Accuracy vs model size at fixed training time")
    ax1.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)

    tb_handles = [
        matplotlib.lines.Line2D([], [], color="gray", marker=TIME_BUDGET_MARKERS[t],
                                ls="none", markersize=8, label=TIME_BUDGET_LABELS[t])
        for t in TIME_BUDGETS
    ]
    ax1.legend(handles=method_handles_left + tb_handles,
               title="Method / time budget", fontsize=9)

    # Right: time to threshold vs n_params
    method_handles = []
    for method in ("minsr", "qps"):
        if method not in data:
            continue
        c = METHOD_COLORS[method]
        method_handles.append(
            matplotlib.lines.Line2D([], [], color=c, lw=2, label=METHOD_LABELS[method])
        )
        for thr in THRESHOLDS:
            xs, means, stds = [], [], []
            for size in SIZE_ORDER:
                if size not in data[method]:
                    continue
                valid = data[method][size]["times"][thr]
                valid = valid[~np.isnan(valid)]
                if not len(valid):
                    continue
                xs.append(data[method][size]["n_params"])
                means.append(np.mean(valid))
                stds.append(np.std(valid) if len(valid) > 1 else 0.0)
            if not xs:
                continue
            xs, means, stds = np.array(xs), np.array(means), np.array(stds)
            ax2.plot(xs, means, marker=THRESHOLD_MARKERS[thr], ls="-",
                     color=c, markersize=8, zorder=3)
            ax2.errorbar(xs, means, yerr=stds, fmt="none", color=c, capsize=4, zorder=2)

    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Number of parameters")
    ax2.set_ylabel("Wall-clock time to threshold (s)")
    ax2.set_title("Time to accuracy threshold")
    ax2.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)

    thr_handles = [
        matplotlib.lines.Line2D([], [], color="gray", marker=THRESHOLD_MARKERS[t],
                                ls="none", markersize=8, label=THRESHOLD_LABELS[t])
        for t in THRESHOLDS
    ]
    ax2.legend(handles=method_handles + thr_handles,
               title="Method / threshold", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {out_path}")


# ── Training curves figure ───────────────────────────────────────────────────

def make_training_curves_figure(collected, out_path):
    """
    Single panel. Both methods on the same plot.
    x = wall-clock time (s, linear), y = eval_rel_error_exact (log).
    Color encodes method, line style encodes model size.
    Individual seeds: thin semi-transparent lines.
    Mean across seeds: thick line on a common linear time grid.
    """
    N_GRID = 300  # points in the common time grid for mean curves

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ("minsr", "qps"):
        c = METHOD_COLORS[method]

        for size in SIZE_ORDER:
            ls = SIZE_LINESTYLES[size]
            runs = [r for r in collected if r["method"] == method and r["size_label"] == size]
            if not runs:
                continue

            # Collect valid (time, error) series
            series = []
            for rec in runs:
                h = rec["history"]
                if h is None:
                    continue
                if "_runtime" not in h.columns or "eval_rel_error_exact" not in h.columns:
                    continue
                sub = h[["_runtime", "eval_rel_error_exact"]].dropna()
                if len(sub) < 2:
                    continue
                series.append((sub["_runtime"].values, sub["eval_rel_error_exact"].values))

            if not series:
                continue

            # Mean curve on a linear common grid
            t_max = max(s[0][-1] for s in series)
            t_grid = np.linspace(0, t_max, N_GRID)
            interp_errors = []
            for t_vals, e_vals in series:
                interp_errors.append(np.interp(t_grid, t_vals, e_vals,
                                               left=np.nan, right=e_vals[-1]))
            mean_curve = np.nanmean(interp_errors, axis=0)
            ax.plot(t_grid, mean_curve, color=c, ls=ls, lw=2.5, zorder=4)

    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel(r"$\epsilon_{\rm rel}$")
    ax.set_title("Training curves: relative error vs time")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)

    # Mark time budget positions
    for tb, label in TIME_BUDGET_LABELS.items():
        ax.axvline(tb, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.text(tb + ax.get_xlim()[1] * 0.005, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-6,
                label, fontsize=7, color="gray", va="bottom", rotation=90)

    # Legend: method (color) + size (line style)
    method_handles = [
        matplotlib.lines.Line2D([], [], color=METHOD_COLORS[m], lw=2,
                                label=METHOD_LABELS[m])
        for m in ("minsr", "qps")
    ]
    size_handles = [
        matplotlib.lines.Line2D([], [], color="gray", ls=SIZE_LINESTYLES[s],
                                lw=2, label=SIZE_LABELS[s])
        for s in SIZE_ORDER
    ]
    ax.legend(handles=method_handles + size_handles,
              title="Method / size", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved -> {out_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

collected = fetch_runs(WANDB_ENTITY, WANDB_PROJECT)

if not collected:
    print("No matching runs found. Check WANDB_ENTITY and WANDB_PROJECT above.")
else:
    data = aggregate(collected)
    print_summary(data)
    make_figure(data, OUT_PATH)

    curves_out = OUT_PATH.replace(".pdf", "_curves.pdf").replace(".png", "_curves.png")
    if curves_out == OUT_PATH:
        curves_out = OUT_PATH + "_curves.pdf"
    make_training_curves_figure(collected, curves_out)
