"""Plot training-loss curves and pass@1-vs-size from W&B history and results/.

    uv run --extra train --group dev python scripts/plot_curves.py

Writes results/figures/{training_loss,scaling}_{light,dark}.png for the README.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.report import summarize  # noqa: E402
from training.config import list_runs, load_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
WANDB_PROJECT = "horrizon/verilog-coder-ft"
SIZES = ("1.5B", "3B", "7B")
FAMILIES = {"L": "LoRA", "Q": "QLoRA (NF4)", "F": "Full fine-tune"}

# Categorical slots 1-4 of the reference palette, light and dark steps (validated).
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#33332f",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}


def run_label(run: str) -> str:
    cfg = load_run(run)
    if cfg["method"] == "full":
        return f"lr {cfg['learning_rate']:.0e}".replace("e-0", "e-")
    scope = "attn" if len(cfg["target_modules"]) == 4 else "all"
    return f"r={cfg['lora_r']} {scope}"


def fetch_histories() -> dict[str, list[tuple[float, float]]]:
    import wandb

    api = wandb.Api(timeout=60)
    wanted = {r for size in SIZES for r in list_runs(grid_size=size)}
    latest = {}
    for run in api.runs(WANDB_PROJECT, filters={"state": "finished"}):
        if run.name in wanted and run.summary.get("steps", 0) >= 600:
            if run.name not in latest or run.created_at > latest[run.name].created_at:
                latest[run.name] = run
    histories = {}
    for name, run in latest.items():
        rows = run.scan_history(keys=["train/loss", "train/epoch"])
        histories[name] = [(r["train/epoch"], r["train/loss"]) for r in rows
                           if r.get("train/loss") is not None and r["train/loss"] < 5]
    return histories


def ema(values: list[float], alpha: float = 0.1) -> list[float]:
    out, acc = [], values[0]
    for v in values:
        acc = alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def place_labels(ax, ends: list[tuple[float, float, str]], theme: dict, min_gap: float = 0.07) -> None:
    """Direct-label line ends, nudging labels apart so close curves don't collide."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap
    ends = sorted(ends, key=lambda e: e[1])
    placed = []
    for x, y, text in ends:
        target = max(y, placed[-1] + gap) if placed else y
        placed.append(target)
        ax.annotate(text, (x, y), xytext=(x + 0.03 * (ax.get_xlim()[1] - ax.get_xlim()[0]), target),
                    textcoords="data", fontsize=7.5, color=theme["text"], va="center",
                    arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.6})


def style(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=8, length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def plot_loss(histories: dict, theme_name: str) -> Path:
    theme = THEMES[theme_name]
    fig, axes = plt.subplots(len(SIZES), 3, figsize=(11, 8.2), sharex=True, facecolor=theme["surface"])
    for i, size in enumerate(SIZES):
        for j, (fam, fam_name) in enumerate(FAMILIES.items()):
            ax = axes[i][j]
            style(ax, theme)
            runs = [r for r in list_runs(grid_size=size) if r.rsplit("-", 1)[1][0] == fam and r in histories]
            ends = []
            for run in runs:
                slot = int(run[-1]) - 1  # config 1/2/3 keeps the same color in every panel
                xs, ys = zip(*histories[run])
                smooth = ema(list(ys))
                ax.plot(xs, smooth, color=theme["series"][slot], linewidth=2)
                ends.append((xs[-1], smooth[-1], run_label(run)))
            ax.set_xlim(0, 1.32)
            place_labels(ax, ends, theme)
            if not runs:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color=theme["muted"], fontsize=9)
            if i == 0:
                ax.set_title(fam_name, color=theme["text"], fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel(f"{size}\ntrain loss", color=theme["muted"], fontsize=9)
            if i == len(SIZES) - 1:
                ax.set_xlabel("epoch", color=theme["muted"], fontsize=9)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.suptitle("Training loss (EMA-smoothed, 40k examples, 1 epoch)", color=theme["text"],
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    path = OUT / f"training_loss_{theme_name}.png"
    fig.savefig(path, dpi=160, facecolor=theme["surface"])
    plt.close(fig)
    return path


def best_scores() -> dict:
    """OpenRTLSet test pass@1 for each size's base model and validation-selected run per family."""
    selection = json.loads((ROOT / "results" / "selection.json").read_text())
    load = lambda tag: [json.loads(line) for line in open(ROOT / "results" / f"{tag}.jsonl")]
    scores = {"Base model": {}}
    for size in SIZES:
        scores["Base model"][size] = summarize(load(f"base-Qwen2.5-Coder-{size}-Instruct"), None)["openrtl"]["pass@1"]
        for fam, run in selection.get(size, {}).get("best_by_family", {}).items():
            if (ROOT / "results" / f"{run}.jsonl").exists():
                scores.setdefault(FAMILIES[fam], {})[size] = summarize(load(run), None)["openrtl"]["pass@1"]
    return scores


def plot_scaling(scores: dict, theme_name: str) -> Path:
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=theme["surface"])
    style(ax, theme)
    order = ["Base model", "LoRA", "QLoRA (NF4)", "Full fine-tune"]
    ends = []
    for slot, name in enumerate(order):
        pts = [(SIZES.index(s), 100 * v) for s, v in scores.get(name, {}).items()]
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, color=theme["series"][slot], linewidth=2, marker="o", markersize=8,
                markeredgecolor=theme["surface"], markeredgewidth=2, label=name)
        ends.append((xs[-1], ys[-1], name))
    ax.set_xticks(range(len(SIZES)), SIZES)
    ax.set_xlim(-0.2, len(SIZES) - 0.1)
    place_labels(ax, ends, theme, min_gap=0.06)
    ax.set_ylabel("pass@1 (%)", color=theme["muted"], fontsize=9)
    ax.set_title("OpenRTLSet held-out pass@1 by model size (best run per method, chosen on validation)",
                 color=theme["text"], fontsize=10, loc="left")
    legend = ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    fig.tight_layout()
    path = OUT / f"scaling_{theme_name}.png"
    fig.savefig(path, dpi=160, facecolor=theme["surface"])
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    histories = fetch_histories()
    print(f"loss histories for {len(histories)} runs")
    scores = best_scores()
    for theme in THEMES:
        print(plot_loss(histories, theme))
        print(plot_scaling(scores, theme))


if __name__ == "__main__":
    main()
