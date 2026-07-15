"""Explore page — market overview from precomputed aggregates (no raw data loaded)."""
import altair as alt
import pandas as pd
import streamlit as st

from viz import load_json, segment_df, hbar, BLUE

st.set_page_config(page_title="Explore · Used Car Prices", page_icon="📊", layout="wide")

agg = load_json("dashboard_aggregates.json")

st.title("📊 Explore the market")
st.caption("Precomputed from ~358,000 Canadian listings — no raw dataset is loaded at runtime.")

ov = agg["overall"]
c1, c2, c3 = st.columns(3)
c1.metric("Listings", f"{ov['count']:,}")
c2.metric("Median price", f"${ov['median_price']:,.0f}")
c3.metric("Mean price", f"${ov['mean_price']:,.0f}")

# --- Price distribution ---
st.subheader("Price distribution")
st.caption("Listing prices, binned. Capped at the 99th percentile for readability.")
dist = agg["price_distribution"]
hist = pd.DataFrame(
    {"low": dist["bin_edges"][:-1], "high": dist["bin_edges"][1:], "count": dist["counts"]}
)
hist_chart = (
    alt.Chart(hist)
    .mark_bar(color=BLUE)
    .encode(
        x=alt.X("low:Q", title="Price ($)", axis=alt.Axis(format="$,.0f")),
        x2="high:Q",
        y=alt.Y("count:Q", title="Listings"),
        tooltip=[
            alt.Tooltip("low:Q", title="From", format="$,.0f"),
            alt.Tooltip("high:Q", title="To", format="$,.0f"),
            alt.Tooltip("count:Q", title="Listings", format=","),
        ],
    )
    .properties(height=260)
)
st.altair_chart(hist_chart, width="stretch")

# --- Price by segment ---
PRICE_TT = [
    alt.Tooltip("name:N", title=""),
    alt.Tooltip("median:Q", title="Median", format="$,.0f"),
    alt.Tooltip("mean:Q", title="Mean", format="$,.0f"),
    alt.Tooltip("count:Q", title="Listings", format=","),
]


def price_df(key):
    df = segment_df(agg[key], ["median_price", "mean_price", "count"])
    return df.rename(columns={"median_price": "median", "mean_price": "mean"})


st.subheader("Median price by make")
st.caption("Top 20 makes by listing volume.")
st.altair_chart(
    hbar(price_df("price_by_make"), value="median", color=BLUE,
         x_title="Median price ($)", tooltip=PRICE_TT),
    width="stretch",
)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("By body type")
    st.altair_chart(
        hbar(price_df("price_by_body_type"), value="median", color=BLUE,
             x_title="Median price ($)", tooltip=PRICE_TT),
        width="stretch",
    )
with col_b:
    st.subheader("By province")
    # Taller bands (height_per) so all 12 provinces get an axis label and the chart
    # fills the space next to the taller body-type chart instead of clipping the last one.
    st.altair_chart(
        hbar(price_df("price_by_province"), value="median", color=BLUE,
             x_title="Median price ($)", tooltip=PRICE_TT, height_per=40),
        width="stretch",
    )

with st.expander("Show the numbers"):
    st.dataframe(price_df("price_by_make"), width="stretch", hide_index=True)
