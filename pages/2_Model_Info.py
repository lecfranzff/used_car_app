"""Model-info page — headline metrics, accuracy bands, and where the model is weak."""
import altair as alt
import pandas as pd
import streamlit as st

from viz import load_json, segment_df, hbar, BLUE, ORANGE

st.set_page_config(page_title="Model info · Used Car Prices", page_icon="ℹ️", layout="wide")

meta = load_json("xgb_deployment_metadata.json")
agg = load_json("dashboard_aggregates.json")

st.title("ℹ️ Model information")

tm = meta["test_metrics"]
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"${tm['MAE']:,.0f}", help="Average absolute dollar error on the held-out test set.")
c2.metric("RMSE", f"${tm['RMSE']:,.0f}", help="Root mean squared error — penalises large misses.")
c3.metric("R²", f"{tm['R2']:.3f}", help="Share of price variance explained.")

# --- Accuracy bands ---
st.subheader("Accuracy bands")
st.caption("Share of test predictions within a given error of the real price.")
bands = meta["accuracy_bands"]
band_rows = [
    ("Within $1,000", bands["within_1000_pct"]),
    ("Within $2,500", bands["within_2500_pct"]),
    ("Within $5,000", bands["within_5000_pct"]),
    ("Within 10%", bands["within_10_percent_pct"]),
    ("Within 20%", bands["within_20_percent_pct"]),
]
bdf = pd.DataFrame(band_rows, columns=["name", "pct"])
band_chart = (
    alt.Chart(bdf)
    .mark_bar(color=BLUE, cornerRadiusEnd=4, size=24)
    .encode(
        x=alt.X("pct:Q", title="Share of predictions (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("name:N", sort=[r[0] for r in band_rows], title=None),
        tooltip=[alt.Tooltip("name:N", title=""), alt.Tooltip("pct:Q", title="Share", format=".1f")],
    )
    .properties(height=210)
)
st.altair_chart(band_chart, width="stretch")

# --- Where the model is weak ---
st.subheader("Where the model is weak")
st.caption("Larger bars = larger errors. These are the segments the model is least reliable on.")

ERR_TT = [
    alt.Tooltip("name:N", title=""),
    alt.Tooltip("mae:Q", title="MAE", format="$,.0f"),
    alt.Tooltip("pct:Q", title="Median % error", format=".1f"),
    alt.Tooltip("count:Q", title="Test listings", format=","),
]


def err_df(key):
    df = segment_df(agg[key], ["mae", "median_abs_pct_error", "count"])
    return df.rename(columns={"median_abs_pct_error": "pct"})


st.markdown("**By make** — worst 20 by dollar error")
st.altair_chart(
    hbar(err_df("error_by_make"), value="mae", color=ORANGE, x_title="MAE ($)", tooltip=ERR_TT),
    width="stretch",
)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**By province**")
    st.altair_chart(
        hbar(err_df("error_by_province"), value="mae", color=ORANGE, x_title="MAE ($)", tooltip=ERR_TT),
        width="stretch",
    )
with col_b:
    st.markdown("**By vehicle age** — median % error")
    st.altair_chart(
        hbar(err_df("error_by_car_age_bucket"), value="pct", color=ORANGE,
             x_title="Median % error", x_format=".0f", tooltip=ERR_TT,
             sort=["4-6 yrs", "7-10 yrs", "11-15 yrs", "15+ yrs"]),
        width="stretch",
    )

st.info(
    "**Reading the errors honestly.** Absolute error grows with price, so luxury/exotic makes "
    "(Ferrari, Porsche, Maserati) show the largest dollar MAE. In *percentage* terms the model is "
    "proportionally **worst on the cheapest cars** and on vehicles **15+ years old**. It is most "
    "reliable on mainstream 4–10 year-old vehicles."
)

with st.expander("Model details"):
    fp = meta["final_params"]
    lv = meta["library_versions"]
    st.markdown(
        f"""
- **Algorithm:** XGBoost regression — {fp['n_estimators']} trees, max_depth {fp['max_depth']}, learning_rate {fp['learning_rate']}
- **Deployment artifact:** {meta['artifact_size_mb']} MB (CPU refit of the GPU-tuned model)
- **Training:** ~358k listings · staged GridSearch tuning · leakage-safe scikit-learn pipeline
- **Key libraries:** Python {lv['python']} · scikit-learn {lv['scikit-learn']} · XGBoost {lv['xgboost']} · pandas {lv['pandas']}
"""
    )
