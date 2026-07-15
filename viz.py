"""Shared helpers for the Explore and Model info pages: cached JSON loading, the
chart colours, and a small Altair horizontal-bar builder.

Each chart shows a single series, so I use one colour per chart - blue for prices,
orange for model error. Streamlit themes the axes and background; I only set the
bar colour.
"""
from pathlib import Path
import json

import altair as alt
import pandas as pd
import streamlit as st

MODELS = Path(__file__).parent / "models"

BLUE = "#2a78d6"    # prices
ORANGE = "#eb6834"  # model error


@st.cache_data
def load_json(name):
    with open(MODELS / name) as f:
        return json.load(f)


def segment_df(segment: dict, keys) -> pd.DataFrame:
    """Flatten a {name: {metric: value, ...}} aggregate into a tidy DataFrame."""
    return pd.DataFrame(
        [{"name": name, **{k: row[k] for k in keys}} for name, row in segment.items()]
    )


def hbar(df, *, value, color, x_title, tooltip, x_format="$,.0f", sort="-x", height_per=26):
    """Horizontal bars: category on Y, one measure on X, one colour, hover tooltips."""
    height = max(150, height_per * len(df))
    return (
        alt.Chart(df)
        .mark_bar(color=color, cornerRadiusEnd=4, size=height_per - 9)
        .encode(
            x=alt.X(f"{value}:Q", title=x_title, axis=alt.Axis(format=x_format)),
            y=alt.Y("name:N", sort=sort, title=None, axis=alt.Axis(labelLimit=180)),
            tooltip=tooltip,
        )
        .properties(height=height)
    )
