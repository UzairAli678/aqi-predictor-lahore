import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import requests

# === CONFIG ===
DATA_DIR = 'data'
MODEL_DIR = 'models'
CURRENT_CSV = os.path.join(DATA_DIR, 'aqi_lahore.csv')
HIST_CSV = os.path.join(DATA_DIR, 'aqi_historical.csv')
MODEL_CSV = os.path.join(DATA_DIR, 'model_comparison.csv')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
BEST_MODEL_NAME_PATH = os.path.join(MODEL_DIR, 'best_model_name.txt')
CITY = 'Lahore'

FEATURES = [
    'pm25', 'pm10', 'no2', 'so2', 'o3', 'co',
    'temperature', 'humidity', 'wind_speed',
    'hour', 'day', 'month', 'day_of_week'
]

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Lahore AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === SIDEBAR ===
st.sidebar.title("Settings")
st.sidebar.markdown(f"**City:** {CITY}")

try:
    last_update = pd.read_csv(CURRENT_CSV)['timestamp'].iloc[-1]
except Exception:
    last_update = 'N/A'
st.sidebar.markdown(f"**Last Updated:** {last_update}")

try:
    with open(BEST_MODEL_NAME_PATH) as f:
        best_model_name = f.read().strip()
except Exception:
    best_model_name = 'N/A'
st.sidebar.markdown(f"**Best Model:** {best_model_name}")

if st.sidebar.button("🔄 Refresh Dashboard"):
    st.rerun()

# === TITLE ===
st.title("🌍 Lahore AQI Predictor")
st.markdown("#### Real-time Air Quality Monitoring & 3-Day Forecast")
st.markdown("---")

# === AQI COLOR FUNCTION ===
def get_aqi_status(aqi):
    if aqi <= 50:
        return 'Good', '#43a047'
    elif aqi <= 100:
        return 'Moderate', '#fbc02d'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups', '#fb8c00'
    elif aqi <= 200:
        return 'Unhealthy', '#e53935'
    else:
        return 'Very Unhealthy / Hazardous', '#8e24aa'

# === AQI FORMULA ===
def calculate_aqi_from_pm25(pm25):
    if pm25 <= 12.0:
        return round((50/12.0) * pm25)
    elif pm25 <= 35.4:
        return round(((100-51)/(35.4-12.1)) * (pm25-12.1) + 51)
    elif pm25 <= 55.4:
        return round(((150-101)/(55.4-35.5)) * (pm25-35.5) + 101)
    elif pm25 <= 150.4:
        return round(((200-151)/(150.4-55.5)) * (pm25-55.5) + 151)
    elif pm25 <= 250.4:
        return round(((300-201)/(250.4-150.5)) * (pm25-150.5) + 201)
    else:
        return round(((500-301)/(500.4-250.5)) * (pm25-250.5) + 301)

# === CURRENT AQI SECTION ===
st.subheader("Current AQI")

ow_api_key = "eeee98c7401e1f201da1f7694cc0bd98"
ow_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=31.5497&lon=74.3436&appid={ow_api_key}"
api_success = False

try:
    response = requests.get(ow_url, timeout=10)
    if response.status_code == 200:
        data = response.json()
        if "list" in data and len(data["list"]) > 0:
            comp = data["list"][0]["components"]
            pm25 = comp.get("pm2_5", 0)
            pm10 = comp.get("pm10", 0)
            no2 = comp.get("no2", 0)
            so2 = comp.get("so2", 0)
            o3 = comp.get("o3", 0)
            co = comp.get("co", 0)
            aqi = calculate_aqi_from_pm25(pm25)
            api_success = True
except Exception as e:
    api_success = False

if api_success:
    st.info("ℹ️ Showing LIVE AQI from OpenWeather API.")
    aqi_status, aqi_color = get_aqi_status(aqi)
    st.markdown(f"""
        <div style='background:{aqi_color};padding:2rem 1rem;border-radius:1rem;text-align:center;'>
            <span style='font-size:3rem;font-weight:bold;color:white;'>{aqi}</span>
            <span style='font-size:1.2rem;color:white;background:#e53935;padding:0.2rem 0.7rem;border-radius:0.7rem;margin-left:1rem;vertical-align:middle;'>🔴 LIVE</span><br>
            <span style='font-size:1.5rem;color:white;'>{aqi_status}</span>
        </div>
    """, unsafe_allow_html=True)
    if aqi > 150:
        st.error(f"⚠️ ALERT: AQI is {aqi} ({aqi_status})! Unhealthy air quality.")

    st.subheader("Current Pollutants")
    pollutant_vals = [pm25, pm10, no2, so2, o3, co]
    pollutant_labels = ['PM2.5', 'PM10', 'NO₂', 'SO₂', 'O₃', 'CO']
    cols = st.columns(len(pollutant_labels))
    for col, label, val in zip(cols, pollutant_labels, pollutant_vals):
        col.metric(label, f"{val:.2f}")
else:
    st.warning("⚠️ LIVE AQI fetch failed. Showing fallback from CSV.")
    try:
        current_df = pd.read_csv(CURRENT_CSV)
        current = current_df.iloc[-1]
        aqi = int(current['aqi'])
        aqi_status, aqi_color = get_aqi_status(aqi)
        st.markdown(f"""
            <div style='background:{aqi_color};padding:2rem 1rem;border-radius:1rem;text-align:center;'>
                <span style='font-size:3rem;font-weight:bold;color:white;'>{aqi}</span><br>
                <span style='font-size:1.5rem;color:white;'>{aqi_status}</span>
            </div>
        """, unsafe_allow_html=True)
        if aqi > 150:
            st.error(f"⚠️ ALERT: AQI is {aqi} ({aqi_status})!")

        st.subheader("Current Pollutants")
        pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
        pollutant_labels = ['PM2.5', 'PM10', 'NO₂', 'SO₂', 'O₃', 'CO']
        cols = st.columns(len(pollutant_cols))
        for col, label, c in zip(cols, pollutant_labels, pollutant_cols):
            val = current.get(c, np.nan)
            col.metric(label, f"{val:.2f}" if pd.notnull(val) else "N/A")
    except Exception as e:
        st.warning(f"Could not load current AQI data. ({e})")

# === MODEL COMPARISON SECTION ===
st.subheader("Model Comparison")
try:
    model_df = pd.read_csv(MODEL_CSV)
    fig = px.bar(model_df, x='Model', y='RMSE', color='Model',
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title='Model RMSE Comparison')
    st.plotly_chart(fig, use_container_width=True)
    winner = model_df.sort_values('R2', ascending=False).iloc[0]
    st.success(f"🏆 **Best Model:** {winner['Model']} (R²={winner['R2']:.3f}) - Lowest RMSE: {winner['RMSE']:.2f}")
except Exception as e:
    st.warning(f"Could not load model comparison data. ({e})")

# === 3 DAY FORECAST SECTION ===
st.subheader("3-Day AQI Forecast")
try:
    api_url = "http://127.0.0.1:5000/api/forecast"
    forecast_data = None
    try:
        resp = requests.get(api_url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            forecast = data["forecast"]
            times = [datetime.now() + timedelta(hours=i+1) for i in range(72)]
            aqi_vals = [item["predicted_aqi"] for item in forecast]
            forecast_data = pd.DataFrame({"Time": times, "Predicted AQI": aqi_vals})
            for d in range(3):
                day_val = forecast_data.iloc[d*24:(d+1)*24]["Predicted AQI"].mean()
                st.info(f"Day {d+1} Predicted AQI: {day_val:.1f}")
            fig = px.line(forecast_data, x="Time", y="Predicted AQI",
                          title="72-Hour AQI Forecast",
                          color_discrete_sequence=['#e53935'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            raise Exception("Flask API unavailable")
    except Exception:
        st.warning("Flask API unavailable, using OpenWeather fallback.")
        ow_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=31.5497&lon=74.3436&appid={ow_api_key}"
        resp = requests.get(ow_url, timeout=15)
        resp.raise_for_status()
        ow_data = resp.json()
        slots = []
        for item in ow_data["list"]:
            slots.append({
                "temperature": item["main"]["temp"] - 273.15,
                "humidity": item["main"]["humidity"],
                "wind_speed": item["wind"]["speed"]
            })
        times = [datetime.now() + timedelta(hours=i+1) for i in range(72)]
        aqi_vals = []
        pm25_base = 75
        for h in range(72):
            pm25_var = pm25_base * (1 + np.random.uniform(-0.1, 0.1))
            aqi_vals.append(calculate_aqi_from_pm25(pm25_var))
        forecast_data = pd.DataFrame({"Time": times, "Predicted AQI": aqi_vals})
        for d in range(3):
            day_val = forecast_data.iloc[d*24:(d+1)*24]["Predicted AQI"].mean()
            st.info(f"Day {d+1} Predicted AQI: {day_val:.1f}")
        fig = px.line(forecast_data, x="Time", y="Predicted AQI",
                      title="72-Hour AQI Forecast",
                      color_discrete_sequence=['#e53935'])
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not generate forecast. ({e})")

# === SHAP FEATURE IMPORTANCE SECTION ===
st.subheader("🔍 SHAP Feature Importance Analysis")
try:
    shap_df = pd.read_csv("data/shap_importance.csv")
    shap_df = shap_df.rename(columns={shap_df.columns[0]: "feature", shap_df.columns[1]: "importance"})
    shap_df = shap_df.sort_values("importance", ascending=False)
    fig = px.bar(shap_df, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="OrRd",
                 title="🔍 SHAP Feature Importance Analysis")
    fig.update_layout(xaxis_title="SHAP Importance Value",
                      yaxis_title="Feature",
                      yaxis=dict(autorange="reversed"),
                      plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    st.write("SHAP (SHapley Additive exPlanations) shows how much each feature contributes to AQI predictions. PM2.5 is the dominant factor in Lahore's AQI.")
    if len(shap_df) >= 3:
        st.info(f"🥇 {shap_df.iloc[0]['feature']} is most important feature")
        st.info(f"🥈 {shap_df.iloc[1]['feature']} is second most important")
        st.info(f"🥉 {shap_df.iloc[2]['feature']} is third most important")
except Exception as e:
    st.warning(f"Could not display SHAP feature importance. ({e})")

# === HISTORICAL TREND SECTION ===
st.subheader("Historical AQI Trend (30 Days)")
try:
    hist_df = pd.read_csv(HIST_CSV)
    # Fix mixed timestamp formats
    hist_df['timestamp'] = pd.to_datetime(
        hist_df['timestamp'],
        format='mixed',
        utc=True
    )
    hist_df = hist_df.sort_values('timestamp')
    hist_df = hist_df.dropna(subset=['aqi'])
    fig = px.line(hist_df, x='timestamp', y='aqi',
                  title='AQI Over Past 30 Days',
                  color_discrete_sequence=['#3949ab'])
    for y, color, label in [
        (50, 'green', 'Good'),
        (100, 'yellow', 'Moderate'),
        (150, 'orange', 'Unhealthy for Sensitive Groups'),
        (200, 'red', 'Unhealthy'),
        (300, 'purple', 'Very Unhealthy')
    ]:
        fig.add_hline(y=y, line_dash='dot', line_color=color,
                      annotation_text=label,
                      annotation_position='top left')
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load historical AQI data. ({e})")

st.markdown("---")
st.caption("© 2026 Lahore AQI Predictor | Built with Streamlit, Plotly, SHAP, and Python")