import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from functools import lru_cache
from matplotlib.lines import Line2D
from hockey_rink import NHLRink, CircularImage
from wsba_hockey.tools.globals import EVENT_MARKERS, IMG_PATH, INFO_PATH, METRIC_EVENTS, STRENGTHS

## PLOTTING ##
# Functions to assist event plotting 

def wsba_rink(display_range: str = "offense", rotation: int | None = 0, ax=None, figsize=(10, 12)):
    logo_image = None
    try:
        if isinstance(IMG_PATH, str) and Path(IMG_PATH).exists():
            from PIL import Image

            logo_image = np.array(Image.open(IMG_PATH))
    except Exception:
        logo_image = None

    features = {}
    features["ice"] = {
        "image": np.zeros((1, 1, 4), dtype=np.uint8),
        "visible": False,
    }
    if logo_image is not None:
        features["center_logo"] = {
            "feature_class": CircularImage,
            "image": logo_image,
            "length": 25,
            "width": 25,
            "x": 0,
            "y": 0,
            "radius": 14,
            "zorder": 11,
        }

    rink = NHLRink(**features)

    ax = rink.draw(
        ax=ax,
        figsize=figsize if ax is None else None,
        display_range=display_range,
        rotation=rotation,
        despine=True,
    )
    fig = ax.figure
    return fig, ax, rink


class WSBAPlot:
    def __init__(
        self,
        display_range: str = "full",
        rotation: int | None = 0,
        figsize=(10, 12),
        facecolor="w",
        edgecolor="k",
    ):
        self.fig, self.rink_ax = plt.subplots(1, 1, figsize=figsize, facecolor=facecolor, edgecolor=edgecolor)
        _, self.rink_ax, self.rink = wsba_rink(display_range=display_range, rotation=rotation, ax=self.rink_ax)

        self.ax = self.fig.add_axes(self.rink_ax.get_position(), sharex=self.rink_ax, sharey=self.rink_ax, frameon=False)
        self.ax.patch.set_alpha(0)
        self.ax.set_axis_off()
        self.ax.set_zorder(self.rink_ax.get_zorder() + 1)

    def __getattr__(self, name):
        return getattr(self.ax, name)

    def flush(self):
        self.ax.cla()
        self.ax.patch.set_alpha(0)
        self.ax.set_axis_off()
        self.ax.set_xlim(self.rink_ax.get_xlim())
        self.ax.set_ylim(self.rink_ax.get_ylim())


@lru_cache(maxsize=4)
def load_teaminfo(info_path: str = INFO_PATH) -> pl.DataFrame:
    return pl.read_csv(info_path)


def team_primary_color_map(teaminfo: pl.DataFrame | None = None, *, info_path: str = INFO_PATH) -> dict[str, str]:
    teaminfo = load_teaminfo(info_path) if teaminfo is None else teaminfo
    if teaminfo is None or teaminfo.is_empty():
        return {}
    return dict(zip(teaminfo["wsba_id"].cast(pl.String).to_list(), teaminfo["primary_color"].cast(pl.String).to_list()))


def _legend_handles(events: list[str], marker_dict: dict) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker_dict.get(event, "o"),
            linestyle="None",
            label=event,
            markersize=8,
            markerfacecolor="#1f77b4",
            markeredgecolor="black" if event == "goal" else "white",
            markeredgewidth=0.75,
        )
        for event in events
    ]


def _horizontal_legend_columns(handle_count: int) -> int:
    return max(1, min(handle_count, 5))


def _vertical_legend_columns(handle_count: int) -> int:
    return max(1, int(np.ceil(handle_count / 5)))


def _add_below_rink_legend(plotter: WSBAPlot, handles: list[Line2D]) -> None:
    fig = plotter.fig
    w_in, h_in = fig.get_size_inches()
    pos = plotter.rink_ax.get_position()

    ncol = _horizontal_legend_columns(len(handles))
    rows = int(np.ceil(len(handles) / ncol))
    legend_h_in = 0.28 * rows + 0.18
    gap_in = 0.08
    extra_in = legend_h_in + gap_in

    # Increase canvas height to make room for legend without shrinking the rink.
    new_h_in = h_in + extra_in
    fig.set_size_inches(w_in, new_h_in, forward=True)

    # Keep rink size (in inches) and top margin constant by shifting axes up.
    axes_h_in = pos.height * h_in
    axes_y0_in = pos.y0 * h_in
    new_y0 = (axes_y0_in + extra_in) / new_h_in
    new_h = axes_h_in / new_h_in
    new_pos = [pos.x0, new_y0, pos.width, new_h]
    plotter.rink_ax.set_position(new_pos)
    plotter.ax.set_position(new_pos)

    center_x = new_pos[0] + (new_pos[2] / 2.0)
    anchor_y = new_pos[1] - (gap_in / new_h_in)

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(center_x, anchor_y),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="black",
        handletextpad=0.6,
        columnspacing=1.0,
        fontsize=9,
    )
    if leg is not None:
        leg.set_zorder(1000)


def _add_on_rink_legend(plotter: WSBAPlot, handles: list[Line2D], display_range: str) -> None:
    ax = plotter.ax
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    display = str(display_range).lower()

    if display == "full":
        anchor = (0, 0)
    else:
        # For vertical half-rinks, put the legend in the neutral-zone lane between
        # the red line and blue line, opposite the net.
        y_mid = (ylim[0] + ylim[1]) / 2.0
        y_anchor = 12.5 if y_mid > 0 else -12.5
        anchor = ((xlim[0] + xlim[1]) / 2.0, y_anchor)

    leg = ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=anchor,
        bbox_transform=ax.transData,
        ncol=_vertical_legend_columns(len(handles)),
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="black",
        borderpad=0.45,
        handletextpad=0.55,
        labelspacing=0.35,
        fontsize=8,
    )
    if leg is not None:
        leg.set_zorder(1000)


def _add_event_legend(
    plotter: WSBAPlot,
    events: list[str],
    marker_dict: dict,
    display_range: str,
    rotation: int | None,
) -> None:
    handles = _legend_handles(events, marker_dict)
    rot = 0 if rotation is None else int(rotation)
    is_vertical = abs(rot) % 180 == 90

    if is_vertical:
        _add_on_rink_legend(plotter, handles, display_range)
    else:
        _add_below_rink_legend(plotter, handles)


def apply_primary_colors(
    df: pl.DataFrame,
    color_map: dict[str, str],
    *,
    team_abbr_col: str = "event_team_abbr",
    season_col: str = "season",
    out_col: str = "color",
    fallback: str = "#1f77b4",
) -> pl.DataFrame:
    if team_abbr_col not in df.columns or season_col not in df.columns:
        return df.with_columns(pl.lit(fallback).alias(out_col))

    wsba_ids = (
        pl.col(team_abbr_col).cast(pl.String).str.to_uppercase()
        + pl.col(season_col).cast(pl.Int64, strict=False).cast(pl.String)
    )
    return df.with_columns(wsba_ids.replace(color_map, default=fallback).alias(out_col))


def plot_events(
    pbp: pl.DataFrame,
    events: list[str],
    title: str | None = None,
    marker_dict: dict | None = None,
    legend: bool = False,
    display_range: str = "full",
    rotation: int | None = 0,
    figsize=(6.4, 4.8),
):
    marker_dict = EVENT_MARKERS if marker_dict is None else marker_dict

    plotter = WSBAPlot(display_range=display_range, rotation=rotation, figsize=figsize)
    ax = plotter.ax

    if pbp.is_empty():
        if title:
            ax.set_title(title)
        return plotter.fig

    if "size" in pbp.columns:
        size_all = pbp["size"].to_numpy()
    else:
        if "xG" in pbp.columns:
            xg = pbp["xG"].cast(pl.Float64, strict=False).fill_null(0).to_numpy()
            size_all = np.where(xg < 0.05, 20.0, xg * 400.0).astype(np.float32, copy=False)
        else:
            size_all = np.full(len(pbp), 20.0, dtype=np.float32)

    event_type_arr = pbp["event_type"].to_numpy()
    if rotation is None:
        rotation = 0
    rot = float(rotation)
    if rot:
        theta = np.deg2rad(rot)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

    plotted_events: list[str] = []

    for event in events:
        event_mask = event_type_arr == event
        if not np.any(event_mask):
            continue

        x = pbp.filter(pl.Series(event_mask))["x_adj"].to_numpy()
        y = pbp.filter(pl.Series(event_mask))["y_adj"].to_numpy()

        ok = np.isfinite(x) & np.isfinite(y)
        if not np.any(ok):
            continue

        x = x[ok]
        y = y[ok]
        if rot:
            x0 = x.astype(np.float32, copy=False)
            y0 = y.astype(np.float32, copy=False)
            x = (x0 * cos_t) - (y0 * sin_t)
            y = (x0 * sin_t) + (y0 * cos_t)
        sizes = size_all[event_mask][ok]

        if "color" in pbp.columns:
            colors = pbp.filter(pl.Series(event_mask))["color"].to_numpy()[ok]
        elif "event_team_venue" in pbp.columns:
            venues = pbp.filter(pl.Series(event_mask))["event_team_venue"].to_numpy()[ok]
            colors = np.where(venues == "away", "#1f77b4", "#d62728")
        else:
            colors = "#1f77b4"

        ax.scatter(
            x,
            y,
            sizes,
            colors,
            marker=marker_dict.get(event, "o"),
            edgecolors="black" if event == "goal" else "white",
            linewidths=0.75,
            label=event,
            zorder=5,
        )
        plotted_events.append(event)

    if title:
        ax.set_title(title)

    if legend and plotted_events:
        _add_event_legend(plotter, plotted_events, marker_dict, display_range, rotation)

    return plotter.fig

def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]

def _normalize_strengths(strengths):
    if strengths == "all":
        return STRENGTHS
    if isinstance(strengths, str):
        return [strengths]
    return strengths

def prep_plot_data(
    pbp,
    strengths,
    season_types=2,
    marker_dict=EVENT_MARKERS
):
    try:
        pbp["xG"]
    except Exception:
        from wsba_hockey.wsba_main import nhl_apply_xG

        pbp = nhl_apply_xG(pbp)
        pbp = pbp.with_columns(pl.col("xG").fill_null(0).alias("xG"))

    pbp = pbp.with_columns([
        pl.when(pl.col("x_adj") < 0).then(-pl.col("y_adj")).otherwise(pl.col("y_adj")).alias("x_plot"),
        pl.col("x_adj").abs().alias("y_plot"),
        pl.col("strength_state").alias("strength_state_for"),
        pl.col("strength_state").cast(pl.String).str.reverse().alias("strength_state_against"),
        pl.when(pl.col("xG") < 0.05).then(20).otherwise(pl.col("xG") * 400).alias("size"),
        pl.col("event_type").replace(marker_dict, default=None).alias("marker"),
        pl.when(pl.col("home_team_abbr") == pl.col("event_team_abbr")).then(pl.col("home_on_ice_id")).otherwise(pl.col("away_on_ice_id")).alias("onice_for_id"),
        pl.when(pl.col("away_team_abbr") == pl.col("event_team_abbr")).then(pl.col("home_on_ice_id")).otherwise(pl.col("away_on_ice_id")).alias("onice_against_id"),
    ]).with_columns([
        (pl.col("onice_for_id").cast(pl.String) + ";" + pl.col("onice_against_id").cast(pl.String)).alias("onice_id"),
        (pl.col("event_team_abbr_for") + ";" + pl.col("event_team_abbr_against")).alias("event_team_abbrs"),
    ])

    season_types = _as_list(season_types)
    if season_types:
        pbp = pbp.filter(pl.col("season_type").is_in(season_types))

    strengths_norm = _normalize_strengths(strengths)
    if strengths != "all":
        pbp = pbp.filter(
            pl.col("strength_state_for").is_in(strengths_norm) | pl.col("strength_state_against").is_in(strengths_norm)
        )

    pbp = pbp.with_columns([
        (pl.col("event_type") == "goal").cast(pl.Int64).alias("is_goal"),
        pl.col("event_type").is_in(METRIC_EVENTS["Shots"]).cast(pl.Int64).alias("is_shot"),
        pl.col("event_type").is_in(METRIC_EVENTS["Fenwick"]).cast(pl.Int64).alias("is_fenwick"),
        pl.col("event_type").is_in(METRIC_EVENTS["Corsi"]).cast(pl.Int64).alias("is_corsi"),
        (pl.col("event_type") == "giveaway").cast(pl.Int64).alias("is_give"),
        (pl.col("event_type") == "takeaway").cast(pl.Int64).alias("is_take"),
        (pl.col("event_type") == "penalty").cast(pl.Int64).alias("is_penalty"),
        (pl.col("event_type") == "blocked-shot").cast(pl.Int64).alias("is_block"),
        (pl.col("event_type") == "hit").cast(pl.Int64).alias("is_hit"),
    ])

    return pbp
