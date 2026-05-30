
# === AQI Model Training Script ===
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import xgboost as xgb

DATA_PATH = 'data/aqi_historical.csv'
MODEL_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
BEST_MODEL_NAME_PATH = os.path.join(MODEL_DIR, 'best_model_name.txt')
COMPARISON_CSV = 'data/model_comparison.csv'

FEATURES = [
    'pm25', 'pm10', 'no2', 'so2', 'o3', 'co',
    'temperature', 'humidity', 'wind_speed',
    'hour', 'day', 'month', 'day_of_week'
]
TARGET = 'aqi'

MODELS = {
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Ridge Regression': Ridge(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
}

def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def generate_synthetic_data(n=500, random_state=42):
    np.random.seed(random_state)
    data = {
        'aqi': np.random.randint(50, 301, n),
        'pm25': np.random.uniform(20, 200, n),
        'pm10': np.random.uniform(30, 250, n),
        'no2': np.random.uniform(5, 80, n),
        'so2': np.random.uniform(2, 50, n),
        'o3': np.random.uniform(10, 60, n),
        'co': np.random.uniform(1, 15, n),
        'temperature': np.random.uniform(15, 45, n),
        'humidity': np.random.uniform(20, 90, n),
        'wind_speed': np.random.uniform(0, 30, n),
        'hour': np.random.randint(0, 24, n),
        'day': np.random.randint(1, 32, n),
        'month': np.random.randint(1, 13, n),
        'day_of_week': np.random.randint(0, 7, n)
    }
    return pd.DataFrame(data)

def load_and_merge_data(synth_df):
    if not os.path.exists(DATA_PATH):
        return synth_df
    try:
        df = pd.read_csv(DATA_PATH)
        # Check if required columns exist and file is not empty
        if df.empty or not set(FEATURES + [TARGET]).issubset(df.columns):
            return synth_df
        # Drop rows with missing values in required columns
        df = df.dropna(subset=FEATURES + [TARGET])
        merged = pd.concat([synth_df, df], ignore_index=True)
        return merged
    except Exception as e:
        print(f"Warning: Could not load {DATA_PATH} ({e}). Using only synthetic data.")
        return synth_df

def prepare_data(df):
    X = df[FEATURES]
    y = df[TARGET]
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

def main():
    ensure_model_dir()
    try:
        synth_df = generate_synthetic_data()
        df = load_and_merge_data(synth_df)
        X, y = prepare_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        results = []
        best_score = -np.inf
        best_model = None
        best_model_name = None
        print("=== Model Comparison ===")
        for name, model in MODELS.items():
            try:
                model.fit(X_train, y_train)
                rmse, mae, r2 = evaluate_model(model, X_test, y_test)
                results.append({
                    'Model': name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                })
                print(f"{name:16} → RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"Error training {name}: {e}")
        if best_model is not None:
            try:
                joblib.dump(best_model, BEST_MODEL_PATH)
                with open(BEST_MODEL_NAME_PATH, 'w') as f:
                    f.write(best_model_name)
                print(f"Best Model: {best_model_name} ✅ (saved to {BEST_MODEL_PATH})")
            except Exception as e:
                print(f"Error saving best model: {e}")
        else:
            print("No model was successfully trained.")
        try:
            comp_df = pd.DataFrame(results)
            comp_df.to_csv(COMPARISON_CSV, index=False)
            print(f"Model comparison saved to {COMPARISON_CSV}")
        except Exception as e:
            print(f"Error saving model comparison: {e}")
    except ImportError as e:
        print(f"ImportError: {e}. Please ensure all required packages are installed. Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
