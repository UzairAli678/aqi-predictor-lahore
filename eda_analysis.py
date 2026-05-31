import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'data/aqi_historical.csv'
PLOT_DIR = 'data'

# 1. LOAD DATA
def load_data(path):
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
        print(f"Data shape: {df.shape}")
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nBasic statistics:")
        print(df.describe(include='all'))
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_plot(fig, filename):
    try:
        fig.savefig(os.path.join(PLOT_DIR, filename), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    if df is None or df.empty:
        print("No data to analyze.")
        return

    # 2. PLOT 1 - AQI Distribution
    try:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df['aqi'].dropna(), bins=30, kde=True, ax=ax, color='skyblue')
        ax.set_title('AQI Distribution in Lahore')
        ax.set_xlabel('AQI')
        save_plot(fig, 'eda_aqi_distribution.png')
    except Exception as e:
        print(f"Error in AQI Distribution plot: {e}")

    # 3. PLOT 2 - AQI Over Time
    try:
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['timestamp'], df['aqi'], color='orange')
        ax.set_title('AQI Trend Over Time')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('AQI')
        save_plot(fig, 'eda_aqi_trend.png')
    except Exception as e:
        print(f"Error in AQI Trend plot: {e}")

    # 4. PLOT 3 - Monthly AQI Average
    try:
        monthly_avg = df.groupby('month')['aqi'].mean()
        fig, ax = plt.subplots(figsize=(8,5))
        monthly_avg.plot(kind='bar', ax=ax, color='teal')
        ax.set_title('Monthly Average AQI')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average AQI')
        save_plot(fig, 'eda_monthly_aqi.png')
    except Exception as e:
        print(f"Error in Monthly AQI plot: {e}")

    # 5. PLOT 4 - Hourly AQI Pattern
    try:
        hourly_avg = df.groupby('hour')['aqi'].mean()
        fig, ax = plt.subplots(figsize=(8,5))
        hourly_avg.plot(kind='bar', ax=ax, color='purple')
        ax.set_title('Hourly AQI Pattern')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average AQI')
        save_plot(fig, 'eda_hourly_aqi.png')
    except Exception as e:
        print(f"Error in Hourly AQI plot: {e}")

    # 6. PLOT 5 - Correlation Heatmap
    try:
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        save_plot(fig, 'eda_correlation.png')
    except Exception as e:
        print(f"Error in Correlation Heatmap: {e}")

    # 7. PLOT 6 - Pollutants vs AQI
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18,5))
        sns.scatterplot(x='pm25', y='aqi', data=df, ax=axes[0], color='red')
        axes[0].set_title('PM2.5 vs AQI')
        sns.scatterplot(x='pm10', y='aqi', data=df, ax=axes[1], color='blue')
        axes[1].set_title('PM10 vs AQI')
        sns.scatterplot(x='no2', y='aqi', data=df, ax=axes[2], color='green')
        axes[2].set_title('NO2 vs AQI')
        for ax in axes:
            ax.set_xlabel(ax.get_title().split(' vs ')[0])
            ax.set_ylabel('AQI')
        fig.suptitle('Pollutants vs AQI')
        save_plot(fig, 'eda_pollutants_vs_aqi.png')
    except Exception as e:
        print(f"Error in Pollutants vs AQI plot: {e}")

    # 8. Print KEY FINDINGS
    try:
        peak_month = int(df.groupby('month')['aqi'].mean().idxmax())
        peak_hour = int(df.groupby('hour')['aqi'].mean().idxmax())
        corr = df.corr(numeric_only=True)
        corr_aqi = corr['aqi'].drop('aqi').abs()
        most_corr_feature = corr_aqi.idxmax()
        avg_aqi = df['aqi'].mean()
        max_aqi = df['aqi'].max()
        min_aqi = df['aqi'].min()
        print("\nKEY FINDINGS:")
        print(f"Peak AQI month: {peak_month}")
        print(f"Peak AQI hour: {peak_hour}")
        print(f"Most correlated feature: {most_corr_feature}")
        print(f"Average AQI: {avg_aqi:.2f}")
        print(f"Max AQI: {max_aqi}")
        print(f"Min AQI: {min_aqi}")
    except Exception as e:
        print(f"Error in KEY FINDINGS: {e}")

if __name__ == "__main__":
    main()
