import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

MODEL_PATH = 'models/best_model.pkl'
DATA_PATH = 'data/aqi_historical.csv'
PLOT_DIR = 'data'
IMPORTANCE_CSV = os.path.join(PLOT_DIR, 'shap_importance.csv')

feature_cols = [
    'pm25', 'pm10', 'no2', 'so2', 'o3',
    'co', 'temperature', 'humidity',
    'wind_speed', 'hour', 'day',
    'month', 'day_of_week'
]

def save_plot(filename):
    try:
        plt.savefig(os.path.join(PLOT_DIR, filename), bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    # 1. Load model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    # 2. Load data
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    # 3. Prepare features
    try:
        X = df[feature_cols].copy().fillna(df[feature_cols].mean())
        y = df['aqi']
        X = X.iloc[:500]
        y = y.iloc[:500]
        print(f"Prepared features: {X.shape}")
    except Exception as e:
        print(f"Error preparing features: {e}")
        return
    # 4. Calculate SHAP values
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        print("SHAP values calculated.")
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return
    # 5. SHAP plots
    try:
        # Summary Bar Plot
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        save_plot('shap_summary_bar.png')
        # Summary Dot Plot
        shap.summary_plot(shap_values, X, show=False)
        save_plot('shap_summary_dot.png')
        # Dependence Plot for pm25
        shap.dependence_plot("pm25", shap_values, X, show=False)
        save_plot('shap_dependence_pm25.png')
    except Exception as e:
        print(f"Error creating SHAP plots: {e}")
    # 6. Top 5 features by SHAP
    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = pd.Series(mean_abs_shap, index=feature_cols)
        importance = importance.sort_values(ascending=False)
        print("Top 5 Features by SHAP importance:")
        for i, (feat, val) in enumerate(importance.head(5).items(), 1):
            print(f"{i}. {feat}: {val:.4f}")
        # 7. Save importance
        importance.to_csv(IMPORTANCE_CSV, header=['mean_abs_shap'])
        print(f"Saved SHAP importance to {IMPORTANCE_CSV}")
    except Exception as e:
        print(f"Error in SHAP importance: {e}")

if __name__ == "__main__":
    main()
