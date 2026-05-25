import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Boco Rock Wind Power Predictor",
    page_icon="🌬️",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return model, feature_cols

model, FEATURE_COLS = load_model()

# ── Feature engineering (mirrors notebook exactly) ───────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Wind speed & direction
    df["ws_10"]         = np.sqrt(df["u10"]**2  + df["v10"]**2)
    df["ws_100"]        = np.sqrt(df["u100"]**2 + df["v100"]**2)
    df["ws_100_sq"]     = df["ws_100"]**2
    df["ws_100_cubed"]  = df["ws_100"]**3
    df["wd_10"]         = (180 + (180/np.pi) * np.arctan2(df["u10"],  df["v10"]))  % 360
    df["wd_100"]        = (180 + (180/np.pi) * np.arctan2(df["u100"], df["v100"])) % 360

    # Air density (NaN if sp/t2m absent — XGBoost handles natively)
    if "sp" in df.columns and "t2m" in df.columns:
        df["air_density"]         = df["sp"] / (287.05 * df["t2m"])
        df["wind_kinetic_energy"] = 0.5 * df["air_density"] * df["ws_100_cubed"]
        df["density_ws2"]         = df["air_density"] * df["ws_100"]**2
    else:
        df["air_density"]         = np.nan
        df["wind_kinetic_energy"] = np.nan
        df["density_ws2"]         = np.nan

    # Lag & rolling features
    df["ws_100_lag_1"]       = df["ws_100"].shift(1)
    df["ws_100_lag_2"]       = df["ws_100"].shift(2)
    df["ws_100_lag_3"]       = df["ws_100"].shift(3)
    df["ws_100_lag_6"]       = df["ws_100"].shift(6)
    df["ws_100_lead_1"]      = df["ws_100"].shift(-1)
    df["ws_100_roll_std_3h"] = df["ws_100"].rolling(3,  min_periods=1).std()
    df["ws_100_roll_mean_3h"]  = df["ws_100"].rolling(3,  min_periods=1).mean()
    df["ws_100_roll_mean_6h"]  = df["ws_100"].rolling(6,  min_periods=1).mean()
    df["ws_100_roll_mean_24h"] = df["ws_100"].rolling(24, min_periods=1).mean()
    df["ws_100_diff_1h"]     = df["ws_100"].diff(1)
    df["ws_100_dev_3h"]      = df["ws_100"] - df["ws_100_roll_mean_3h"]
    df["ws_100_dev_24h"]     = df["ws_100"] - df["ws_100_roll_mean_24h"]
    df["turbulence_intensity"] = df["ws_100_roll_std_3h"] / (df["ws_100_roll_mean_3h"] + 1e-6)
    df["wind_shear"] = df["ws_100"] / (df["ws_10"] + 1e-6)
    df["wind_veer"]  = df["wd_100"] - df["wd_10"]

    # Cyclic time encoding
    df["hour"]      = df["time"].dt.hour
    df["month"]     = df["time"].dt.month
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24.0)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["doy_sin"]   = np.sin(2 * np.pi * df["time"].dt.dayofyear / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["time"].dt.dayofyear / 365)

    df = df.bfill().ffill()
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌬️ Model Info")
    st.markdown("""
| Metric | Value |
|--------|-------|
| Model | XGBoost |
| Trained on | 2016–2023 |
| Test year | 2024 |
| R² | 0.7694 |
| MAE | 12.19 MW |
| RMSE | 17.17 MW |
| Farm capacity | ~113 MW |
    """)
    st.markdown("---")
    st.markdown("### 📋 Required CSV columns")
    st.code("time\nu10\nv10\nu100\nv100", language="text")
    st.caption("ERA5 hourly wind data in UTC. App converts to AEST automatically if needed.")
    st.markdown("---")
    st.markdown("**ENGG2112 — Team 12**")

# ── Main header ───────────────────────────────────────────────────────────────
st.title("🌬️ Boco Rock Wind Power Predictor")
st.markdown(
    "Upload ERA5 wind data for **Boco Rock Wind Farm** (NSW, Australia) "
    "and the XGBoost model will predict hourly power output in MW."
)

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload ERA5 wind CSV",
    type=["csv"],
    help="Must contain columns: time, u10, v10, u100, v100"
)

st.caption("Don't have a file? Download the sample below to try it out.")
with open("sample_boco_week.csv", "rb") as f:
    st.download_button(
        "⬇️ Download sample CSV (Boco Rock, Jan 2024 — 1 week)",
        data=f,
        file_name="sample_boco_week.csv",
        mime="text/csv"
    )

if uploaded is not None:
    # ── Parse ─────────────────────────────────────────────────────────────────
    try:
        raw = pd.read_csv(uploaded)
        raw["time"] = pd.to_datetime(raw["time"])
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        st.stop()

    required = {"time", "u10", "v10", "u100", "v100"}
    missing  = required - set(raw.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    st.success(f"✅ Loaded {len(raw):,} rows  |  {raw['time'].min().date()} → {raw['time'].max().date()}")

    with st.expander("Preview uploaded data"):
        st.dataframe(raw.head(10), use_container_width=True)

    # ── Feature engineering + prediction ──────────────────────────────────────
    with st.spinner("Engineering features and running XGBoost..."):
        df_feat = engineer_features(raw)
        X       = df_feat[FEATURE_COLS]
        preds   = np.clip(model.predict(X), 0, None)  # no negative power

    df_out = raw[["time"]].copy()
    df_out["wind_speed_100m_ms"] = df_feat["ws_100"].values
    df_out["predicted_power_mw"] = preds

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ Predicted Output Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Power",  f"{preds.mean():.1f} MW")
    col2.metric("Peak Power",     f"{preds.max():.1f} MW")
    col3.metric("Min Power",      f"{preds.min():.1f} MW")
    n_hours = len(preds)
    col4.metric("Total Energy",   f"{preds.sum() / 1000:.2f} GWh",
                help=f"Over {n_hours} hours")

    # ── Predictions table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔢 Hourly Predictions")

    display_df = df_out.copy()
    display_df.columns = ["Time", "Wind Speed at 100m (m/s)", "Predicted Power (MW)"]
    display_df["Time"] = display_df["Time"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["Wind Speed at 100m (m/s)"] = display_df["Wind Speed at 100m (m/s)"].round(2)
    display_df["Predicted Power (MW)"] = display_df["Predicted Power (MW)"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        hide_index=True,
        column_config={
            "Predicted Power (MW)": st.column_config.ProgressColumn(
                "Predicted Power (MW)",
                help="Predicted power output",
                min_value=0,
                max_value=113,
                format="%.1f MW",
            ),
        }
    )

    # ── Time series ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Predicted Power — Time Series")

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_out["time"],
        y=df_out["predicted_power_mw"],
        mode="lines",
        name="Predicted Power",
        line=dict(color="#457B9D", width=1.8)
    ))
    fig_ts.update_layout(
        xaxis_title="Time",
        yaxis_title="Predicted Power (MW)",
        yaxis=dict(range=[0, 120]),
        hovermode="x unified",
        height=380,
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Power curve ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🌀 Power Curve — Wind Speed vs. Predicted Power")

    fig_pc = go.Figure()
    fig_pc.add_trace(go.Scatter(
        x=df_out["wind_speed_100m_ms"],
        y=df_out["predicted_power_mw"],
        mode="markers",
        marker=dict(color="#E63946", size=5, opacity=0.5),
        name="Predicted"
    ))
    fig_pc.update_layout(
        xaxis_title="Wind Speed at 100 m (m/s)",
        yaxis_title="Predicted Power (MW)",
        yaxis=dict(range=[0, 120]),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_pc, use_container_width=True)

    # ── Download predictions ──────────────────────────────────────────────────
    st.markdown("---")
    csv_out = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download predictions as CSV",
        data=csv_out,
        file_name="boco_rock_predictions.csv",
        mime="text/csv"
    )

else:
    # Placeholder when nothing is uploaded
    st.info("👆 Upload a CSV file above to get started. Use the sample file if you just want to see it work.")
    st.markdown("""
    ### How it works
    1. **Upload** an ERA5 hourly wind CSV with columns `time, u10, v10, u100, v100`
    2. The app engineers **30 features** — wind speed, direction, lag windows, turbulence intensity, cyclic time encoding
    3. The trained **XGBoost model** predicts power output in MW for each hour
    4. Results are shown as a time series and power curve, and can be downloaded

    > ⚠️ This model was trained on Boco Rock ERA5 data (2016–2023). Predictions for other wind farm sites will be less accurate — the model has learned site-specific wind patterns.
    """)
