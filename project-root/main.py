from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import os
import pandas as pd
import numpy as np
import cv2
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = FastAPI()

# Constants
HISTORY_FILE = "temperature_history.csv"
LOG_FILE = "alert_log.csv"
PADDY_TEMP_FILE = "paddy_temp_reference.csv"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "Chengalpattu"
COUNTRY = "Tamil Nadu"
THRESHOLD_MARGIN = 5.0
TREND_DAYS = 3
TEMP_MIN = 10.0
TEMP_MAX = 50.0

WEATHER_CONDITIONS = {
    "Rain": "Don't water, rain expected",
    "Heavy Rain": "No watering needed, heavy rain expected",
    "Sunny": "Mild risk, monitor temperature",
    "Clear": "Consider watering if temperature is high",
    "Extreme": "High risk, water immediately"
}

# Utility Functions

def extract_temperature_from_thermal_video(video_path):
    cap = cv2.VideoCapture(video_path)
    temps = []
    frame_count = 0
    if not cap.isOpened():
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
    result = sum(temps) / len(temps) if temps else None
    print(f"Extracted average temperature from video: {result}")
    return result

def load_history():
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE, parse_dates=['date'])
        print(f"Loaded history: {len(df)} records")
        return df
    print("No history file found, creating new history DataFrame")
    return pd.DataFrame(columns=['date', 'temperature', 'watered'])

def compute_baseline(history):
    if not history.empty:
        watered_temps = history[history['watered'] == True]['temperature']
        baseline = watered_temps.mean() if not watered_temps.empty else 28.0
        print(f"Computed baseline from history: {baseline}")
        return baseline
    print("History is empty, default baseline: 28.0")
    return 28.0

def compute_trend(history):
    if len(history) < TREND_DAYS:
        print("Not enough history for trend computation")
        return 0.0
    df = history.sort_values('date').tail(30)
    temps = df['temperature'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    temps_scaled = scaler.fit_transform(temps)
    X, y = [], []
    for i in range(5, len(temps_scaled)):
        X.append(temps_scaled[i-5:i])
        y.append(temps_scaled[i])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        print("Not enough data to build trend model")
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
    trend = predicted_temp - last_temp
    print(f"Predicted temp: {predicted_temp}, Last temp: {last_temp}, Trend: {trend}")
    return trend

def train_anomaly_detector(history):
    if len(history[history['watered'] == True]) < 5:
        print("Not enough data to train anomaly detector")
        return None
    X = history[history['watered'] == True]['temperature'].values.reshape(-1, 1)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    print("Trained anomaly detector")
    return model

def is_anomalous_temperature(current_temp, model):
    if model is None:
        return False
    prediction = model.predict(np.array([[current_temp]]))[0] == -1
    print(f"Anomaly detection result for {current_temp}: {prediction}")
    return prediction

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
        max_temp = max(temps) if temps else None
        common_condition = max(set(conditions), key=conditions.count) if conditions else "Unknown"
        print(f"Forecast for tomorrow: Temp={max_temp}, Condition={common_condition}")
        return max_temp, common_condition
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None, None

def check_paddy_temp_condition(current_temp):
    if not os.path.exists(PADDY_TEMP_FILE):
        print("Paddy temp reference file not found")
        return False
    df = pd.read_csv(PADDY_TEMP_FILE)
    matched = df[df['needs_water_below'] <= current_temp]
    print(f"Paddy temp condition check: {not matched.empty}")
    return not matched.empty

def is_thermal_temp_unreliable(current_temp, forecast_temp, margin=7.0):
    unreliable = forecast_temp is not None and abs(current_temp - forecast_temp) > margin
    print(f"Thermal unreliability: {unreliable}")
    return unreliable

def needs_water(current_temp, watered, baseline, anomaly_model, trend_slope, forecast_temp, weather_condition):
    conditions_triggered = []
    print(f"Evaluating need for water: Current Temp = {current_temp}, Watered = {watered}, Baseline = {baseline}")
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
    print(f"Conditions triggered: {conditions_triggered}")
    if len(conditions_triggered) >= 2:
        return True, f"High temperature detected. {weather_advice}", conditions_triggered
    return False, weather_advice, conditions_triggered

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
    print(f"Logged alert: {log_data}")

def last_log_within_24hrs():
    if not os.path.exists(LOG_FILE):
        return False
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        return False
    last_entry = pd.to_datetime(df['Date'].iloc[-1] + ' ' + df['Time'].iloc[-1])
    within_24hrs = datetime.now() - last_entry < timedelta(hours=24)
    print(f"Last log within 24hrs: {within_24hrs}")
    return within_24hrs

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
        next_date += timedelta(days=1)
    print(f"Average watering interval: {avg_interval} days. Next: {next_date}")
    return f"Water every {avg_interval} days. Next watering: {next_date.strftime('%Y-%m-%d')}"

@app.post("/analyze")
async def analyze_field(video: UploadFile = File(...), watered: bool = Form(...)):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(await video.read())
    current_temp = extract_temperature_from_thermal_video("uploaded_video.mp4")
    history = load_history()
    baseline = compute_baseline(history)
    trend_slope = compute_trend(history)
    anomaly_model = train_anomaly_detector(history)
    forecast_temp, weather_condition = get_weather_forecast()
    thermal_warning = is_thermal_temp_unreliable(current_temp, forecast_temp)
    alert, reason, conditions = needs_water(current_temp, watered, baseline, anomaly_model, trend_slope, forecast_temp, weather_condition)
    if alert:
        log_alert(current_temp, reason, conditions)
    elif not last_log_within_24hrs():
        log_alert(current_temp, "Daily log - No action required", [], force=True)
    frequency = calculate_watering_frequency()
    print(f"Analysis complete: Temp={current_temp}, Needs Water={alert}, Reason={reason}")
    return JSONResponse({
        "temperature": round(current_temp, 2),
        "needs_water": alert,
        "reason": reason,
        "conditions_triggered": conditions,
        "thermal_warning": thermal_warning,
        "weather_condition": weather_condition,
        "forecast_temperature": forecast_temp,
        "watering_frequency": frequency
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
