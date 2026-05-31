import requests
import pandas as pd
from datetime import datetime
import os

OPENWEATHER_KEY = "eeee98c7401e1f201da1f7694cc0bd98"
LAT = 31.5497
LON = 74.3436

def calculate_aqi(pm25):
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

def fetch_openweather_data():
    try:
        # Air Pollution
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
        air_resp = requests.get(air_url, timeout=10)
        air_resp.raise_for_status()
        comp = air_resp.json()["list"][0]["components"]
        pm25 = comp["pm2_5"]
        pm10 = comp["pm10"]
        no2 = comp["no2"]
        so2 = comp["so2"]
        o3 = comp["o3"]
        co = comp["co"]
        aqi = calculate_aqi(pm25)

        # Weather
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
        weather_resp = requests.get(weather_url, timeout=10)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        temperature = round(weather_data["main"]["temp"] - 273.15, 2)
        humidity = weather_data["main"]["humidity"]
        wind_speed = weather_data["wind"]["speed"]

        # Timestamp
        timestamp = datetime.now()

        # Row
        row = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "so2": so2,
            "o3": o3,
            "co": co,
            "aqi": aqi,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "hour": timestamp.hour,
            "day": timestamp.day,
            "month": timestamp.month,
            "day_of_week": timestamp.weekday()
        }

        # Print
        print("\n✅ Latest AQI & Weather Data for Lahore")
        for key, value in row.items():
            print(f"✅ {key}: {value}")

        # Save
        os.makedirs("data", exist_ok=True)
        pd.DataFrame([row]).to_csv("data/aqi_lahore.csv", index=False)

        hist_path = "data/aqi_historical.csv"
        if os.path.exists(hist_path):
            hist_df = pd.read_csv(hist_path)
            hist_df = pd.concat([hist_df, pd.DataFrame([row])], ignore_index=True)
        else:
            hist_df = pd.DataFrame([row])
        hist_df.to_csv(hist_path, index=False)

        print("✅ Data saved successfully!\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    fetch_openweather_data()