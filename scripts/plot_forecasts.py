"""Report figures: baseline vs rule-based benchmarks on the 2025 validation year.

Uses the events family's saved forecasts because every usable row has event
features, so its baseline trains on the most history (other families drop
rows where their predictor is missing). All forecasts are day-ahead (origin
D 00:00 America/New_York) and scored on identical rows.

Inputs:  data/processed/predictor_eval/predictions_events.parquet, metrics.csv
Outputs: data/processed/predictor_eval/figures/*.png

    ./.venv/bin/python scripts/plot_forecasts.py
    ./.venv/bin/python scripts/plot_forecasts.py --week-start 2025-10-06
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from predictor_common import ROOT, TZ, load_stations

EVAL = ROOT / "data" / "processed" / "predictor_eval"
FIGURES = EVAL / "figures"
FAMILY = "events"

# Reference palette, light mode (validated: CVD dE 24.7, normal-vision dE 33.6).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACTUAL_COLOR = "#3d3c39"  # neutral: the observed series is not a model
SERIES = {
    "baseline": ("Poisson baseline", "#2a78d6", "-"),
    "seasonal_naive_168h": ("Same hour last week", "#eb6834", (0, (4, 2))),
}
BENCHMARK_LABELS = {
    "seasonal_naive_168h": "Same hour last week",
    "seasonal_naive_24h": "Same hour yesterday",
    "same_hour_mean_28d": "Same-hour mean, past 28 days",
    "baseline": "Poisson baseline",
}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def new_figure(width, height, rows=1, cols=1, **kw):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), facecolor=SURFACE, **kw)
    return fig, axes


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def plot_wape(metrics):
    rows = metrics[(metrics["family"] == FAMILY) & (metrics["regime"] == "strict")
                   & metrics["model"].isin(BENCHMARK_LABELS)]
    fig, axes = new_figure(9, 3.2, 1, 2, sharex=True)
    for ax, target in zip(axes, ("pickups", "dropoffs")):
        part = rows[rows["target"] == target].set_index("model").loc[list(BENCHMARK_LABELS)][::-1]
        labels = [BENCHMARK_LABELS[m] for m in part.index]
        # One series: a single hue; the baseline is distinguished by label text.
        ax.barh(labels, part["WAPE_percent"], color=SERIES["baseline"][1], height=0.55)
        for i, value in enumerate(part["WAPE_percent"]):
            ax.text(value + 0.8, i, f"{value:.1f}%", va="center", fontsize=8, color=INK)
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_title(target.capitalize(), loc="left", fontsize=10, color=INK)
        ax.set_xlabel("WAPE (% of actual trips)", fontsize=8, color=INK_2)
    axes[0].set_xlim(0, rows["WAPE_percent"].max() * 1.18)
    fig.suptitle("Forecast error by method, 2025 validation year (day-ahead)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "wape_by_method.png")


def plot_week(pred, target, week_start, names):
    start = pd.Timestamp(week_start, tz=TZ)
    week = pred[(pred["target"] == target) & (pred["local"] >= start) & (pred["local"] < start + pd.Timedelta(days=7))]
    if week.empty:
        raise ValueError(f"no scored rows in the week starting {week_start}")
    fig, axes = new_figure(12, 10, 4, 2, sharex=True)
    for ax, (station_id, g) in zip(axes.flat, week.groupby("station_id")):
        g = g.sort_values("local")
        ax.plot(g["local"], g["actual"], color=ACTUAL_COLOR, linewidth=1.6, label="Actual")
        for model, (label, color, dash) in SERIES.items():
            ax.plot(g["local"], g[model], color=color, linewidth=1.2, linestyle=dash, label=label)
        style(ax)
        ax.set_title(f"{names[station_id]} ({station_id})", loc="left", fontsize=9, color=INK)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d", tz=TZ))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"Hourly {target}, week of {start:%b %d, %Y}: actual vs day-ahead forecasts",
                 x=0.01, y=0.995, ha="left", fontsize=11, color=INK)
    fig.supylabel(f"{target.capitalize()} per hour", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"week_{target}_{start:%Y%m%d}.png")


def plot_error_by_hour(pred, target):
    # Mean forecasts match the mean daily profile almost exactly, so averages
    # hide the misses; absolute error by hour shows where they concentrate.
    part = pred[pred["target"] == target].assign(
        hour=lambda d: d["local"].dt.hour,
        day_type=lambda d: d["local"].dt.weekday.map(lambda w: "Weekend" if w >= 5 else "Weekday"),
        **{f"err_{m}": lambda d, m=m: (d["actual"] - d[m]).abs() for m in SERIES},
    )
    mae = part.groupby(["day_type", "hour"])[[f"err_{m}" for m in SERIES]].mean()
    fig, axes = new_figure(10, 3.6, 1, 2, sharey=True)
    for ax, day_type in zip(axes, ("Weekday", "Weekend")):
        m = mae.loc[day_type]
        for model, (label, color, dash) in SERIES.items():
            ax.plot(m.index, m[f"err_{model}"], color=color, linewidth=2, linestyle=dash, label=label)
        style(ax)
        ax.set_title(day_type, loc="left", fontsize=10, color=INK)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Local hour of day", fontsize=8, color=INK_2)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(f"Mean absolute error ({target} per hour)", fontsize=8, color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.suptitle(f"Where the day-ahead forecasts miss: {target} error by hour, 2025, all eight stations",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, f"error_by_hour_{target}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week-start", default="2025-06-02", help="Monday of the example week (2025)")
    args = parser.parse_args()

    pred = pd.read_parquet(EVAL / f"predictions_{FAMILY}.parquet")
    pred["local"] = pd.to_datetime(pred["timestamp"], utc=True).dt.tz_convert(TZ)
    names = load_stations().set_index("station_id")["station_name"].to_dict()

    plot_wape(pd.read_csv(EVAL / "metrics.csv"))
    for target in ("pickups", "dropoffs"):
        plot_week(pred, target, args.week_start, names)
        plot_error_by_hour(pred, target)


if __name__ == "__main__":
    main()
