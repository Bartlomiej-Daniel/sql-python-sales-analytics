from prophet import Prophet
from pathlib import Path
from core.data_loader import load_monthly_data
from modeling.forecasting.evaluation import compute_metrics


def train_model(train_data, changepoint_scale=0.05):
    model = Prophet(changepoint_prior_scale=changepoint_scale)
    model.fit(train_data)
    return model

def make_forecast(model, periods, freq = "MS"):
    future = model.make_future_dataframe(periods = periods, freq = freq)
    forecast = model.predict(future)
    return forecast


# Load data
monthly = load_monthly_data()

# Train/ test split
split_index = int(len(monthly) * 0.8)
train = monthly.iloc[:split_index]
test = monthly.iloc[split_index:]

# Tuning
scales = [0.01, 0.05, 0.1, 0.3, 0.5]

for scale in scales:
    model = train_model(train, changepoint_scale=scale)
    forecast = make_forecast(model, periods=len(test))

    forecast_test = forecast.iloc[split_index:]["yhat"]

    mae, mape = compute_metrics(test["y"], forecast_test)

    print(f"\nChangepoint Scale: {scale}")
    print(f"MAE: {mae:,.2f}")
    print(f"MAPE: {mape:.2f}%")