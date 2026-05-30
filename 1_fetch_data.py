import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

DATA_DIR = 'data'
CURRENT_CSV = os.path.join(DATA_DIR, 'aqi_lahore.csv')
HISTORICAL_CSV = os.path.join(DATA_DIR, 'aqi_historical.csv')

FIELDS = [
    'aqi', 'pm25', 'pm10', 'no2', 'so2', 'o3', 'co',
    'temperature', 'humidity', 'wind_speed',
    'timestamp', 'hour', 'day', 'month', 'day_of_week'
]

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_env():
    load_dotenv()
    api_key = os.getenv('AQICN_API_KEY')
    city = os.getenv('CITY', 'lahore')
    if not api_key:
        raise ValueError('AQICN_API_KEY not found in .env file')
    return api_key, city

def fetch_aqi(api_key, city, date=None):
    if date:
        url = f'https://api.waqi.info/feed/@{city}/?token={api_key}&date={date}'
    else:
        url = f'https://api.waqi.info/feed/{city}/?token={api_key}'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('status') != 'ok':
            raise ValueError(f"API error: {data.get('data', data)}")
        return data['data']
    except Exception as e:
        print(f"Error fetching AQI data: {e}")
        return None

def extract_fields(data):
    iaqi = data.get('iaqi', {})
    main = {
        'aqi': data.get('aqi'),
        'pm25': iaqi.get('pm25', {}).get('v'),
        'pm10': iaqi.get('pm10', {}).get('v'),
        'no2': iaqi.get('no2', {}).get('v'),
        'so2': iaqi.get('so2', {}).get('v'),
        'o3': iaqi.get('o3', {}).get('v'),
        'co': iaqi.get('co', {}).get('v'),
        'temperature': iaqi.get('t', {}).get('v'),
        'humidity': iaqi.get('h', {}).get('v'),
        'wind_speed': iaqi.get('w', {}).get('v'),
    }
    # Time features
    ts = data.get('time', {}).get('s')
    if ts:
        dt = pd.to_datetime(ts)
    else:
        dt = datetime.now()
    main['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
    main['hour'] = dt.hour
    main['day'] = dt.day
    main['month'] = dt.month
    main['day_of_week'] = dt.weekday()
    return main

def save_to_csv(row, csv_path):
    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

def fetch_and_save_current(api_key, city):
    data = fetch_aqi(api_key, city)
    if data:
        row = extract_fields(data)
        save_to_csv(row, CURRENT_CSV)
        print(f"Current AQI data saved to {CURRENT_CSV}")
        return 1
    return 0

def fetch_and_save_historical(api_key, city, days=30):
    count = 0
    today = datetime.now()
    for i in range(1, days+1):
        date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        data = fetch_aqi(api_key, city)
        if data:
            row = extract_fields(data)
            row['timestamp'] = date + ' 12:00:00'  # Approximate midday
            save_to_csv(row, HISTORICAL_CSV)
            count += 1
    print(f"Historical AQI data for {count} days saved to {HISTORICAL_CSV}")
    return count

def main():
    ensure_data_dir()
    try:
        api_key, city = load_env()
        print(f"Fetching current AQI data for {city.title()}...")
        n1 = fetch_and_save_current(api_key, city)
        print(f"Fetching historical AQI data for past 30 days...")
        n2 = fetch_and_save_historical(api_key, city, days=30)
        print(f"Success! {n1} current and {n2} historical records saved.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
