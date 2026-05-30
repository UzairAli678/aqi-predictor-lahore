

# AQI Fetcher for Lahore (Real Data Only)
import os
import requests
import pandas as pd
from datetime import datetime

DATA_DIR = 'data'
CURRENT_CSV = os.path.join(DATA_DIR, 'aqi_lahore.csv')
HISTORICAL_CSV = os.path.join(DATA_DIR, 'aqi_historical.csv')
API_KEY = '81639ed2195e846449b2be120efc8292828c5f59'
URL = f'https://api.waqi.info/feed/lahore/?token={API_KEY}'

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def fetch_aqi():
    try:
        resp = requests.get(URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('status') != 'ok':
            print(f"API error: {data.get('data', data)}")
            return None
        return data['data']
    except Exception as e:
        print(f"Error fetching AQI: {e}")
        return None

def extract_fields(data):
    iaqi = data.get('iaqi', {})
    now = datetime.now()
    row = {
        'aqi': data.get('aqi'),
        'pm25': iaqi.get('pm25', {}).get('v', None),
        'pm10': iaqi.get('pm10', {}).get('v', None),
        'no2': iaqi.get('no2', {}).get('v', None),
        'so2': iaqi.get('so2', {}).get('v', None),
        'o3': iaqi.get('o3', {}).get('v', None),
        'co': iaqi.get('co', {}).get('v', None),
        'temperature': iaqi.get('t', {}).get('v', None),
        'humidity': iaqi.get('h', {}).get('v', None),
        'wind_speed': iaqi.get('w', {}).get('v', None),
        'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
        'hour': now.hour,
        'day': now.day,
        'month': now.month,
        'day_of_week': now.weekday()
    }
    return row

def save_current(row):
    df = pd.DataFrame([row])
    df.to_csv(CURRENT_CSV, mode='w', header=True, index=False)

def save_historical(row):
    df = pd.DataFrame([row])
    if os.path.exists(HISTORICAL_CSV):
        df.to_csv(HISTORICAL_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(HISTORICAL_CSV, mode='w', header=True, index=False)

def print_values(row):
    print(f"✅ AQI: {row['aqi']}")
    print(f"✅ PM2.5: {row['pm25']}")
    print(f"✅ PM10: {row['pm10']}")
    print(f"✅ NO2: {row['no2']}")
    print(f"✅ SO2: {row['so2']}")
    print(f"✅ O3: {row['o3']}")
    print(f"✅ CO: {row['co']}")
    print(f"✅ Temperature: {row['temperature']}")
    print(f"✅ Humidity: {row['humidity']}")
    print(f"✅ Wind Speed: {row['wind_speed']}")
    print(f"✅ Timestamp: {row['timestamp']}")
    print(f"✅ Hour: {row['hour']}")
    print(f"✅ Day: {row['day']}")
    print(f"✅ Month: {row['month']}")
    print(f"✅ Day of Week: {row['day_of_week']}")

def main():
    ensure_data_dir()
    data = fetch_aqi()
    if not data:
        print("Failed to fetch AQI data from API.")
        return
    row = extract_fields(data)
    save_current(row)
    save_historical(row)
    print_values(row)

if __name__ == "__main__":
    main()
