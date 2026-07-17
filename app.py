"""
Used car price estimator - Streamlit app.

Loads the trained pipeline once and predicts a price from the details a user
actually knows about their car. The engineered features (car_age, miles_per_year,
miles_was_zero) are built here from those inputs so they line up with how the
model was trained.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# app.py sits at the repo root; the artifacts live under models/
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "xgb_deployment_pipeline.joblib"
META_PATH = ROOT / "models" / "xgb_deployment_metadata.json"
OPTIONS_PATH = ROOT / "models" / "form_options.json"

# car_age was built from the current year at training time (2026). I fix it here so
# the ages I feed the model match training and don't shift when the year rolls over.
REFERENCE_YEAR = 2026

# The training listings top out at model year 2022 (dataset year.max). Anything newer
# is outside what the model actually saw, so we default here and warn past it.
MAX_TRAINED_YEAR = 2022

# Fallbacks for the fields someone might leave blank. "unknown unknown" is what the
# model saw when both model and trim were missing; "unknown" is what empty
# region/city values were filled with during training.
DEFAULT_MODEL_TRIM = "unknown unknown"
DEFAULT_REGION = "unknown"
DEFAULT_CITY = "unknown"

# set_page_config has to be the first Streamlit call
st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="centered")


@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


META = load_json(META_PATH)
OPTIONS = load_json(OPTIONS_PATH)
EXPECTED_COLUMNS = META["expected_columns"]
MAE = float(META["test_metrics"]["MAE"])
pipe = load_pipeline()


# The raw data labels multi-engine listings with several fuels joined by "/" or ";",
# inconsistently, so "premium unleaded / unleaded" and "premium unleaded; unleaded" are
# the same real-world category split in two. These helpers collapse such duplicates for
# the dropdown while keeping a path back to the exact strings the model was trained on.
def _fuel_key(value):
    """Delimiter- and order-agnostic key: the set of fuels named in the string."""
    parts = re.split(r"\s*[/;]\s*", value.strip())
    return tuple(sorted({p for p in parts if p}))


def _fuel_display_map(raw_values):
    """Map one clean display label -> one raw fuel string the model actually saw.

    The fitted OneHotEncoder only knows the original raw strings; a merged label would
    hit handle_unknown="ignore" and zero out the whole fuel_type encoding. So each group
    of equivalent strings resolves to a single canonical member (alphabetically first).
    """
    groups = {}
    for value in raw_values:
        groups.setdefault(_fuel_key(value), []).append(value)
    display_to_raw = {
        " / ".join(part.title() for part in key): min(members)
        for key, members in groups.items()
    }
    return dict(sorted(display_to_raw.items()))


def build_feature_row(raw: dict) -> pd.DataFrame:
    """Build a one-row frame with the columns in the order the model expects."""
    year = float(raw["year"])
    miles = float(raw["miles"])

    car_age = REFERENCE_YEAR - year
    if car_age < 0:                       # typo'd future year, let the imputer handle it
        car_age = float("nan")
    miles_per_year = miles / car_age if car_age > 0 else float("nan")

    engine_size = float(raw["engine_size"])
    if engine_size <= 0:                  # treat 0 L as "not sure" and let it impute
        engine_size = float("nan")

    row = {
        "miles": miles,
        "year": year,
        "make": raw["make"],
        "body_type": raw["body_type"],
        "vehicle_type": raw["vehicle_type"],
        "drivetrain": raw["drivetrain"],
        "transmission": raw["transmission"],
        "fuel_type": raw["fuel_type"],
        "engine_size": engine_size,
        "engine_block": raw["engine_block"],
        "city": raw.get("city", DEFAULT_CITY),
        "province": raw["province"],
        "miles_was_zero": 1 if miles == 0 else 0,
        "car_age": car_age,
        "miles_per_year": miles_per_year,
        "model_trim": raw.get("model_trim", DEFAULT_MODEL_TRIM),
        "region": raw.get("region", DEFAULT_REGION),
    }
    # match the training column order before predicting
    return pd.DataFrame([row]).reindex(columns=EXPECTED_COLUMNS)


def _reference_row() -> dict:
    return dict(
        year=2017, miles=80000, make="ford", body_type="suv", vehicle_type="truck",
        drivetrain="4wd", transmission="automatic", fuel_type="unleaded",
        engine_size=2.0, engine_block="i", province="on",
    )


# Sanity check on load: predict a known car and stop if the number looks wrong,
# which usually means a dependency version broke the loaded model.
if "selftest_done" not in st.session_state:
    ref = build_feature_row(_reference_row())
    assert list(ref.columns) == EXPECTED_COLUMNS, "column order doesn't match the model"
    ref_price = float(pipe.predict(ref)[0])
    if not (4000 < ref_price < 70000):
        st.error(
            f"Startup check failed: reference prediction ${ref_price:,.0f} is out of "
            "range. The loaded model is probably wrong - not serving predictions."
        )
        st.stop()
    st.session_state["selftest_done"] = True


st.title("🚗 Used Car Price Estimator")
st.write(
    "Estimate the market price of a used car, from an XGBoost model trained on "
    "~358,000 Canadian listings."
)

low = OPTIONS["low_cardinality"]
make_trims = OPTIONS["make_to_model_trims"]
prov_cities = OPTIONS["province_to_cities"]
FUEL_DISPLAY_TO_RAW = _fuel_display_map(low["fuel_type"])
NOT_LISTED = "— not sure / not listed —"


def _titlecase_or_sentinel(value):
    return value if value == NOT_LISTED else value.title()


# Not using st.form here: a form only updates on submit, so the make->trim and
# province->city dropdowns would never refresh. Rendering the widgets directly lets
# the estimate recompute on each change, which is cheap since the model is cached.
c1, c2, c3 = st.columns(3)
with c1:
    year = st.number_input(
        "Year", min_value=1981, max_value=REFERENCE_YEAR, value=MAX_TRAINED_YEAR, step=1,
        help=f"The model was trained on listings up to model year {MAX_TRAINED_YEAR}; "
             "newer years are extrapolated.",
    )
    make = st.selectbox("Make", low["make"], format_func=str.title)
    model_trim_choice = st.selectbox(
        "Model & trim", [NOT_LISTED] + make_trims.get(make, []),
        format_func=_titlecase_or_sentinel, help="Filtered to the selected make.",
    )
with c2:
    miles = st.number_input("Mileage (mi)", min_value=0, max_value=500_000, value=60_000, step=1_000)
    body_type = st.selectbox("Body type", low["body_type"], format_func=str.title)
    fuel_choice = st.selectbox("Fuel type", list(FUEL_DISPLAY_TO_RAW))
with c3:
    engine_size = st.number_input("Engine size (L) — 0 if unknown", min_value=0.0, max_value=10.0, value=2.4, step=0.1)
    transmission = st.selectbox("Transmission", low["transmission"], format_func=str.title)
    drivetrain = st.selectbox("Drivetrain", low["drivetrain"], format_func=str.upper)

c4, c5, c6 = st.columns(3)
with c4:
    vehicle_type = st.selectbox("Vehicle class", low["vehicle_type"], format_func=str.title)
    engine_block = st.selectbox("Engine block", low["engine_block"], format_func=str.upper)
with c5:
    province = st.selectbox("Province", low["province"], format_func=str.upper)
    city_choice = st.selectbox(
        "City", [NOT_LISTED] + prov_cities.get(province, []),
        format_func=_titlecase_or_sentinel, help="Filtered to the selected province.",
    )
with c6:
    postal = st.text_input(
        "Postal code (optional)", max_chars=7, placeholder="e.g. M5V 2T6",
        help="Only the first 3 characters (the FSA) are used - that's the region the model learned.",
    )

# turn the dropdown picks into the values the model was trained on
fuel_type = FUEL_DISPLAY_TO_RAW[fuel_choice]
model_trim = DEFAULT_MODEL_TRIM if model_trim_choice == NOT_LISTED else model_trim_choice
city = DEFAULT_CITY if city_choice == NOT_LISTED else city_choice
postal_clean = postal.strip().lower()
region = postal_clean[:3] if len(postal_clean) >= 3 else DEFAULT_REGION

raw = dict(
    year=year, miles=miles, make=make, body_type=body_type, vehicle_type=vehicle_type,
    drivetrain=drivetrain, transmission=transmission, fuel_type=fuel_type,
    engine_size=engine_size, engine_block=engine_block, province=province,
    model_trim=model_trim, city=city, region=region,
)
price = float(pipe.predict(build_feature_row(raw))[0])
low_band = max(0.0, price - MAE)
high_band = price + MAE

st.divider()
if year > MAX_TRAINED_YEAR:
    st.warning(
        f"ℹ️ This model was trained on listings up to model year {MAX_TRAINED_YEAR}. "
        f"A {int(year)} vehicle is beyond the training data, so this estimate is an "
        "extrapolation and may be less reliable."
    )
st.metric("Estimated price", f"${price:,.0f}")
st.caption(
    f"Likely range **${low_band:,.0f} – ${high_band:,.0f}** (±${MAE:,.0f} average error) · "
    "updates live as you adjust the inputs"
)

bands = META["accuracy_bands"]
with st.expander("About this estimate"):
    st.markdown(
        f"""
- Typical error (MAE): **${MAE:,.0f}**; R² **{META['test_metrics']['R2']:.3f}**.
- **{bands['within_20_percent_pct']:.0f}%** of estimates land within 20% of the real price,
  **{bands['within_10_percent_pct']:.0f}%** within 10%.
- **Weakest** on luxury/exotic makes (Ferrari, Porsche, Maserati) and cars **15+ years old**,
  where errors are proportionally larger. Best on mainstream 4–10 year-old vehicles.
- Leaving model/trim, city, or postal code unset just falls back to the model's
  "unknown" handling — the estimate still works, with less trim/location signal.
"""
    )
