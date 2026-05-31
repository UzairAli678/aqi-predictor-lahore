from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os

app = Flask(__name__)
CORS(app)

# Load model at startup
try:
    model = joblib.load('models/best_model.pkl')
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_load_error = str(e)

# Helper: AQI calculation from pm2.5

def calculate_aqi(pm25):
    try:
        pm25 = float(pm25)
        if pm25 <= 12.0:
            return round((50/12.0) * pm25)
        elif pm25 <= 35.4:
            return round(((100-51)/(35.4-12.1)) * (pm25-12.1) + 51)
        elif pm25 <= 55.4:
            return round(((150-101)/(55.4-35.5)) * (pm25-35.5) + 101)
        elif pm25 <= 150.4:
            return round(((200-151)/(150.4-55.5)) * (pm25-55.5) + 151)
        else:
            return round(((300-201)/(250.4-150.5)) * (pm25-150.5) + 201)
    except Exception:
        return None

def get_status(aqi):
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "AQI Predictor API",
        "status": "running",
        "city": "Lahore"
    })

@app.route('/api/current-aqi', methods=['GET'])
def current_aqi():
    try:
        url = "http://api.openweathermap.org/data/2.5/air_pollution?lat=31.5497&lon=74.3436&appid=eeee98c7401e1f201da1f7694cc0bd98"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        comp = data['list'][0]['components']
        pm25 = comp.get('pm2_5', None)
        pm10 = comp.get('pm10', None)
        no2 = comp.get('no2', None)
        so2 = comp.get('so2', None)
        o3 = comp.get('o3', None)
        co = comp.get('co', None)
        timestamp = datetime.utcfromtimestamp(data['list'][0]['dt']).strftime('%Y-%m-%d %H:%M:%S')
        aqi = calculate_aqi(pm25)
        status = get_status(aqi)
        return jsonify({
            "city": "Lahore",
            "aqi": aqi,
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "so2": so2,
            "o3": o3,
            "co": co,
            "status": status,
            "timestamp": timestamp
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast', methods=['GET'])
def forecast():
    try:
        if not model_loaded:
            return jsonify({"error": "Model not loaded", "details": model_load_error}), 500

        # 1. Fetch current weather
        weather_url = "http://api.openweathermap.org/data/2.5/weather?lat=31.5497&lon=74.3436&appid=eeee98c7401e1f201da1f7694cc0bd98"
        weather_resp = requests.get(weather_url, timeout=10)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        temperature = weather_data["main"]["temp"] - 273.15
        humidity = weather_data["main"]["humidity"]
        wind_speed = weather_data["wind"]["speed"]

        # 2. Fetch current pollution
        pollution_url = "http://api.openweathermap.org/data/2.5/air_pollution?lat=31.5497&lon=74.3436&appid=eeee98c7401e1f201da1f7694cc0bd98"
        pollution_resp = requests.get(pollution_url, timeout=10)
        pollution_resp.raise_for_status()
        pollution_data = pollution_resp.json()
        components = pollution_data["list"][0]["components"]
        pm25 = components["pm2_5"]
        pm10 = components["pm10"]
        no2 = components["no2"]
        so2 = components["so2"]
        o3 = components["o3"]
        co = components["co"]

        now = datetime.now()
        current_hour = now.hour
        current_day = now.day
        current_month = now.month
        current_day_of_week = now.weekday()

        # 3. Generate 72-hour forecast with exact feature order
        features_list = []
        for i in range(72):
            dt = now + timedelta(hours=i)
            features_list.append({
                'pm25': pm25,
                'pm10': pm10,
                'no2': no2,
                'so2': so2,
                'o3': o3,
                'co': co,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'hour': (current_hour + i) % 24,
                'day': dt.day,
                'month': dt.month,
                'day_of_week': dt.weekday()
            })
        features = pd.DataFrame(features_list, columns=[
            'pm25', 'pm10', 'no2', 'so2', 'o3', 'co',
            'temperature', 'humidity', 'wind_speed',
            'hour', 'day', 'month', 'day_of_week'])

        preds = model.predict(features)
        preds = [int(np.round(x)) for x in preds]
        forecast_list = []
        for i, pred_aqi in enumerate(preds):
            status = get_status(pred_aqi)
            forecast_list.append({
                "hour": i+1,
                "predicted_aqi": pred_aqi,
                "status": status
            })
        day1_avg = int(np.round(np.mean(preds[:24])))
        day2_avg = int(np.round(np.mean(preds[24:48])))
        day3_avg = int(np.round(np.mean(preds[48:72])))
        return jsonify({
            "city": "Lahore",
            "forecast": forecast_list,
            "day1_avg": day1_avg,
            "day2_avg": day2_avg,
            "day3_avg": day3_avg
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    try:
        df = pd.read_csv('data/model_comparison.csv')
        best_model = df.loc[df['is_best'] == 1, 'model'].values[0] if 'is_best' in df.columns else None
        return jsonify({
            "model_comparison": df.to_dict(orient='records'),
            "best_model": best_model
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if model_loaded else "error",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
