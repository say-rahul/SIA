import os
import csv
import cv2
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

# Constants
HISTORY_FILE = "temperature_history.csv"
LOG_FILE = "alert_log.csv"
PADDY_TEMP_FILE = "paddy_temp_reference.csv"
WEATHER_API_KEY = "3370b2a0cc1a652422e836ec82daf9b2"
CITY = "Chengalpattu"
COUNTRY = "Tamil Nadu"
THRESHOLD_MARGIN = 5.0
TREND_DAYS = 3
THERMAL_VIDEO_FILE = "thermal_video.mp4"

# Calibration parameters for thermal video
TEMP_MIN = 10.0
TEMP_MAX = 50.0

# Weather-based watering advice
WEATHER_CONDITIONS = {
    "Rain": "Don't water, rain expected",
    "Heavy Rain": "No watering needed, heavy rain expected",
    "Sunny": "Mild risk, monitor temperature",
    "Clear": "Consider watering if temperature is high",
    "Extreme": "High risk, water immediately"
}

# Extract temperature from thermal video using grayscale intensity
def extract_temperature_from_thermal_video(video_path):
    cap = cv2.VideoCapture(video_path)
    temperature = None
    frame_count = 0
    temps = []

    if not cap.isOpened():
        print("Error opening video file.")
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_intensity = np.mean(gray)
            temp = TEMP_MIN + ((avg_intensity / 255.0) * (TEMP_MAX - TEMP_MIN))
            temps.append(temp)

    cap.release()
    return sum(temps) / len(temps) if temps else None

# Load historical temperature and watering data
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE, parse_dates=['date'])
    return pd.DataFrame(columns=['date', 'temperature', 'watered'])

# Calculate average temperature on watered days to use as baseline
def compute_baseline(history):
    if not history.empty:
        watered_temps = history[history['watered'] == True]['temperature']
        return watered_temps.mean() if not watered_temps.empty else 28.0
    return 28.0

# Use LSTM model to predict trend
def compute_trend(history):
    if len(history) < TREND_DAYS:
        return 0.0

    df = history.sort_values('date').tail(30)  # Use recent 30 days
    temps = df['temperature'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    temps_scaled = scaler.fit_transform(temps)

    X, y = [], []
    for i in range(5, len(temps_scaled)):
        X.append(temps_scaled[i-5:i])
        y.append(temps_scaled[i])

    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        return 0.0

    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=1, verbose=0)

    prediction = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]), verbose=0)
    predicted_temp = scaler.inverse_transform(prediction)[0][0]
    last_temp = df['temperature'].values[-1]

    return (predicted_temp - last_temp) / 1.0  # slope proxy

# Train an Isolation Forest model to detect abnormal temperature readings
def train_anomaly_detector(history):
    if len(history[history['watered'] == True]) < 5:
        return None
    X = history[history['watered'] == True]['temperature'].values.reshape(-1, 1)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    return model

# Check if a temperature is an outlier using the anomaly detection model
def is_anomalous_temperature(current_temp, anomaly_model):
    if anomaly_model is None:
        return False
    return anomaly_model.predict(np.array([[current_temp]]))[0] == -1

# Fetch tomorrow's weather forecast using OpenWeatherMap API
def get_weather_forecast():
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY},{COUNTRY}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != "200":
            return None, None

        tomorrow = (datetime.now() + timedelta(days=1)).date()
        temps, conditions = [], []

        for entry in data.get("list", []):
            if "main" in entry and "weather" in entry and entry.get("dt_txt"):
                forecast_date = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S").date()
                if forecast_date == tomorrow:
                    temps.append(entry["main"].get("temp", 0))
                    if entry["weather"]:
                        conditions.append(entry["weather"][0].get("main", "Unknown"))

        return max(temps) if temps else None, max(set(conditions), key=conditions.count) if conditions else "Unknown"
    except:
        return None, None

# Check if paddy-specific condition file indicates watering is needed
def check_paddy_temp_condition(current_temp):
    if not os.path.exists(PADDY_TEMP_FILE):
        return False
    df = pd.read_csv(PADDY_TEMP_FILE)
    matched = df[df['needs_water_below'] <= current_temp]
    return not matched.empty

# Compare thermal camera temp to forecast temp
def is_thermal_temp_unreliable(current_temp, forecast_temp, margin=7.0):
    if forecast_temp is None:
        return False
    return abs(current_temp - forecast_temp) > margin

# Determine whether the field needs watering based on several checks
def needs_water(current_temp, watered, baseline, anomaly_model, trend_slope, forecast_temp, weather_condition):
    conditions_triggered = []

    if watered:
        return False, "Already watered today.", conditions_triggered

    if current_temp > (baseline + THRESHOLD_MARGIN):
        conditions_triggered.append("Baseline threshold crossed")

    if is_anomalous_temperature(current_temp, anomaly_model):
        conditions_triggered.append("Anomaly detected")

    if trend_slope > 0.1:
        conditions_triggered.append("Upward trend")

    if forecast_temp and forecast_temp > (baseline + THRESHOLD_MARGIN):
        conditions_triggered.append("Forecast high temp")

    if check_paddy_temp_condition(current_temp):
        conditions_triggered.append("Thermal condition")

    weather_advice = WEATHER_CONDITIONS.get(weather_condition, "No specific advice")

    if len(conditions_triggered) >= 2:
        return True, f"High temperature detected. {weather_advice}", conditions_triggered
    return False, weather_advice, conditions_triggered

# Append alert to CSV log file with timestamp and conditions
def log_alert(current_temp, reason, conditions_triggered, force=False):
    log_data = {
        'Date': datetime.now().strftime("%Y-%m-%d"),
        'Time': datetime.now().strftime("%H:%M:%S"),
        'Temperature': current_temp,
        'Reason': reason,
        'Conditions Triggered': ", ".join(conditions_triggered) if conditions_triggered else "None"
    }
    df = pd.DataFrame([log_data])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

# Check if a log was made in the last 24 hours
def last_log_within_24hrs():
    if not os.path.exists(LOG_FILE):
        return False
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        return False
    last_entry = pd.to_datetime(df['Date'].iloc[-1] + ' ' + df['Time'].iloc[-1])
    return datetime.now() - last_entry < timedelta(hours=24)

# Estimate how often watering is needed based on log intervals and forecast
def calculate_watering_frequency():
    if not os.path.exists(LOG_FILE):
        return "No watering history available."
    df = pd.read_csv(LOG_FILE)
    df = df[df['Date'].notna() & (df['Date'].astype(str).str.strip() != '')]
    if len(df) < 2:
        return "Not enough data to determine frequency."
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    intervals = df['Date'].diff().dt.days.dropna()
    if intervals.empty:
        return "Not enough valid intervals to determine frequency."
    avg_interval = round(intervals.mean())
    next_date = df['Date'].max() + timedelta(days=avg_interval)

    forecast_temp, _ = get_weather_forecast()
    if forecast_temp and forecast_temp < 25:
        next_date += timedelta(days=1)  # delay watering if forecast is cool

    return f"Water every {avg_interval} days. Next watering: {next_date.strftime('%Y-%m-%d')}"

# ----------------------------
# Main program execution
# ----------------------------
if __name__ == '__main__':
    history = load_history()
    baseline = compute_baseline(history)
    trend_slope = compute_trend(history)
    anomaly_model = train_anomaly_detector(history)
    forecast_temp, weather_condition = get_weather_forecast()

    current_temp = extract_temperature_from_thermal_video(THERMAL_VIDEO_FILE)
    if current_temp is None:
        current_temp = float(input("Enter the current field temperature (°C): "))
    else:
        print(f"Extracted temperature: {current_temp:.2f}°C")

    if is_thermal_temp_unreliable(current_temp, forecast_temp):
        print("⚠ Thermal camera temperature may be inaccurate compared to forecast.")
        try:
            current_temp = float(input("Please verify the camera or enter temperature manually (°C): "))
        except ValueError:
            print("Invalid input. Using thermal reading anyway.")

    watered = input("Is the field watered? (yes/no): ").strip().lower() in ['yes', 'y']
    alert, reason, conditions = needs_water(current_temp, watered, baseline, anomaly_model, trend_slope, forecast_temp, weather_condition)

    if alert:
        print(f"\n⚠ ALERT: {reason}")
        print(f"Triggered Conditions: {', '.join(conditions)}")
        print("💧 Action: Please water the field immediately!")
        log_alert(current_temp, reason, conditions)
    else:
        print(f"✅ {reason}")
        if not last_log_within_24hrs():
            log_alert(current_temp, "Daily log - No action required", [], force=True)

    print(calculate_watering_frequency())