"""Shared figure theme — one clinical look applied to every Plotly chart.

Centralising the figure style here (instead of styling each chart inline)
guarantees a consistent look across both tabs: same fonts, grid, margins,
hover, and colour sequence everywhere. Call apply_theme(fig) as the last step
when building any figure.

Palette mirrors the dashboard CSS (clinical: ink + teal, red reserved for
mortality / adverse outcomes only).

Run from src/ with:  python -m dashboard.theme   (sanity check, prints tokens)
"""

import plotly.graph_objects as go

# --- Clinical palette (single source of truth for figures) ----------------
INK = "#0B1F33"        # primary text
TEAL = "#1A7A8C"       # primary accent
AMBER = "#E8A33D"      # secondary / forecast / warning
RED = "#C2453D"        # reserved: mortality / deceased / adverse
GREEN = "#3C7A5A"      # positive / alive
MUTED = "#6B7A88"      # secondary text, labels
GRID = "#EDF1F4"       # subtle grid lines
PAPER = "#FFFFFF"      # chart background

# Ordered sequence for categorical series (teal-led, red kept out of the
# rotation so it stays meaningful for adverse outcomes only).
SEQUENCE = [TEAL, AMBER, GREEN, "#5B8DB8", "#9B6FB0", MUTED]

FONT_FAMILY = ('-apple-system, "Segoe UI", Roboto, Helvetica, Arial, '
               "sans-serif")


def apply_theme(fig: go.Figure, *, title: str | None = None,
                height: int | None = None) -> go.Figure:
    """Stamp the shared clinical style onto a figure (call last).

    Keeps any title text already set unless `title` is passed. Leaves the
    figure's data untouched — only layout styling is normalised.
    """
    layout = dict(
        font=dict(family=FONT_FAMILY, color=INK, size=13),
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        margin=dict(t=60, b=44, l=24, r=24),
        colorway=SEQUENCE,
        hoverlabel=dict(
            bgcolor="white", bordercolor=GRID,
            font=dict(family=FONT_FAMILY, color=INK, size=12)),
        title=dict(x=0.02, y=0.97, font=dict(size=16, color=INK)),
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center",
                    font=dict(size=12, color=MUTED)),
    )
    if title is not None:
        layout["title"]["text"] = title
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)

    # Quiet, consistent axes — hairline grid, no hard borders.
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor=GRID, tickfont=dict(color=MUTED, size=11),
                     title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor=GRID, tickfont=dict(color=MUTED, size=11),
                     title_font=dict(color=MUTED, size=12))
    return fig


if __name__ == "__main__":
    print("HAIP figure theme tokens")
    print("  INK", INK, "| TEAL", TEAL, "| AMBER", AMBER, "| RED", RED)
    print("  sequence:", SEQUENCE)