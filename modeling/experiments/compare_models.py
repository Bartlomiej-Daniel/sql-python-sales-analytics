import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sqlalchemy import create_engine
from core.data_loader import load_monthly_data, load_daily_data

# Load data
daily = load_daily_data()
monthly = load_monthly_data()

# Daily model
model_daily = Prophet()
model_daily.fit(daily)

future_daily = model_daily.make_future_dataframe(periods=90)
forecast_daily = model_daily.predict(future_daily)

# Monthly model
model_monthly = Prophet()
model_monthly.fit(monthly)

future_monthly = model_monthly.make_future_dataframe(periods = 90)
forecast_monthly = model_monthly.predict(future_monthly)

# Charts
plt.figure(figsize=(14,6))
plt.plot(daily["ds"], daily["y"], label="Daily Actual", alpha=0.4)
plt.plot(forecast_daily["ds"], forecast_daily["yhat"], label="Daily Forecast")
plt.title("Daily Model")
plt.legend()
plt.show()


plt.figure(figsize=(14,6))
plt.plot(monthly["ds"], monthly["y"], label="Monthly Actual", alpha=0.4)
plt.plot(forecast_monthly["ds"], forecast_monthly["yhat"], label="Monthly Forecast")
plt.title("Monthly Model")
plt.legend()
plt.show()

