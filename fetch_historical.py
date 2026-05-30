import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os

def fetch_openaq_data(api_key, city="Lahore", country="PK", parameters=None, days=365):
    if parameters is None:
        parameters = ["pm25", "pm10", "no2", "so2", "o3", "co"]
    headers = {"X-API-Key": api_key}
    base_url = "https://api.openaq.org/v3/"
    # Step 1: Get Lahore location IDs
    try:
        loc_url = f"{base_url}locations?city={city}&country={country}&limit=100"
        loc_resp = requests.get(loc_url, headers=headers)
        loc_resp.raise_for_status()
        loc_data = loc_resp.json()
        location_ids = [loc["id"] for loc in loc_data.get("results", [])]
        if not location_ids:
            print("No location IDs found for Lahore in OpenAQ.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching OpenAQ locations: {e}")
        return pd.DataFrame()
    # Step 2: Fetch measurements for each location
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    all_rows = []
    for loc_id in location_ids:
        page = 1
        while True:
            try:
                meas_url = (
                    f"{base_url}measurements?location_id={loc_id}"
                    f"&parameter={','.join(parameters)}"
                    f"&date_from={start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    f"&date_to={end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    f"&limit=100&page={page}"
                )
                resp = requests.get(meas_url, headers=headers)
                if resp.status_code == 429:
                    print("OpenAQ rate limit hit, sleeping 1s...")
                    time.sleep(1)
                    continue
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break
                for r in results:
                    row = {
                        "timestamp": r["date"]["utc"],
                        "parameter": r["parameter"],
                        "value": r["value"],
                        "unit": r["unit"],
                        "location_id": loc_id
                    }
                    all_rows.append(row)
                if len(results) < 100:
                    break
                page += 1
            except Exception as e:
                print(f"Error fetching OpenAQ measurements: {e}")
                break
    if not all_rows:
        print("No OpenAQ data fetched.")
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    # Pivot to wide format: one row per timestamp
    df = df.pivot_table(index=["timestamp", "location_id"], columns="parameter", values="value").reset_index()
    # Drop duplicate timestamps (keep first)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df

def fetch_openweather_air(api_key, lat, lon, days=365):
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    end_date = int(datetime.utcnow().timestamp())
    start_date = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    params = {
        "lat": lat,
        "lon": lon,
        "start": start_date,
        "end": end_date,
        "appid": api_key
    }
    try:
        resp = requests.get(base_url, params=params)
        if resp.status_code == 429:
            print("OpenWeather air rate limit hit, sleeping 1s...")
            time.sleep(1)
            resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for item in data.get("list", []):
            dt = datetime.utcfromtimestamp(item["dt"]).strftime('%Y-%m-%dT%H:%M:%SZ')
            main = item.get("main", {})
            components = item.get("components", {})
            row = {
                "timestamp": dt,
                "pm25": components.get("pm2_5"),
                "pm10": components.get("pm10"),
                "no2": components.get("no2"),
                "so2": components.get("so2"),
                "o3": components.get("o3"),
                "co": components.get("co"),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        print(f"Error fetching OpenWeather air pollution: {e}")
        return pd.DataFrame()

def fetch_openweather_weather(api_key, lat, lon, days=365):
    base_url = "http://history.openweathermap.org/data/2.5/history/city"
    end_date = int(datetime.utcnow().timestamp())
    start_date = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    params = {
        "lat": lat,
        "lon": lon,
        "type": "hour",
        "start": start_date,
        "end": end_date,
        "appid": api_key
    }
    try:
        resp = requests.get(base_url, params=params)
        if resp.status_code == 429:
            print("OpenWeather weather rate limit hit, sleeping 1s...")
            time.sleep(1)
            resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for item in data.get("list", []):
            dt = datetime.utcfromtimestamp(item["dt"]).strftime('%Y-%m-%dT%H:%M:%SZ')
            main = item.get("main", {})
            wind = item.get("wind", {})
            row = {
                "timestamp": dt,
                "temperature": main.get("temp"),
                "humidity": main.get("humidity"),
                "wind_speed": wind.get("speed"),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        print(f"Error fetching OpenWeather weather: {e}")
        return pd.DataFrame()

def calculate_aqi_pm25(pm25):
    # US EPA breakpoints
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    if pd.isna(pm25):
        return np.nan
    for bp in breakpoints:
        if bp[0] <= pm25 <= bp[1]:
            aqi = ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (pm25 - bp[0]) + bp[2]
            return round(aqi)
    return np.nan

def main():
    # API keys
    OPENAQ_KEY = "86f68a20a988a3a14537bf5d3a37f0f3bffd818442896003d28b98dd8e565b9c"
    OPENWEATHER_KEY = "eeee98c7401e1f201da1f7694cc0bd98"
    LAT, LON = 31.5497, 74.3436
    DAYS = 365
    print("Fetching OpenAQ data...")
    openaq_df = fetch_openaq_data(OPENAQ_KEY, days=DAYS)
    print(f"OpenAQ rows: {len(openaq_df)}")
    print("Fetching OpenWeather air pollution data...")
    ow_air_df = fetch_openweather_air(OPENWEATHER_KEY, LAT, LON, days=DAYS)
    print(f"OpenWeather air rows: {len(ow_air_df)}")
    print("Fetching OpenWeather weather data...")
    ow_weather_df = fetch_openweather_weather(OPENWEATHER_KEY, LAT, LON, days=DAYS)
    print(f"OpenWeather weather rows: {len(ow_weather_df)}")
    # Merge all data on timestamp
    df = openaq_df.copy()
    # Merge with OpenWeather air (fill missing pollutants)
    if not ow_air_df.empty:
        df = pd.merge(df, ow_air_df, on="timestamp", how="outer", suffixes=("", "_ow"))
        for col in ["pm25", "pm10", "no2", "so2", "o3", "co"]:
            df[col] = df[col].combine_first(df.get(f"{col}_ow"))
            if f"{col}_ow" in df:
                df.drop(columns=[f"{col}_ow"], inplace=True)
    # Merge with OpenWeather weather
    if not ow_weather_df.empty:
        df = pd.merge(df, ow_weather_df, on="timestamp", how="left")
    # Add time features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    # Calculate AQI if not present
    if "aqi" not in df:
        df["aqi"] = df["pm25"].apply(calculate_aqi_pm25)
    # Sort and drop duplicates
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    # Save
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/aqi_historical.csv", index=False)
    # Print summary
    if not df.empty:
        print(f"Total rows fetched: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Columns: {list(df.columns)}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    main()
