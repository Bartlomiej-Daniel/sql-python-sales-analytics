import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sqlalchemy import create_engine
from pathlib import Path

# Path to database
BASE_DIR = Path(__file__).resolve().parent.parent
db_path = BASE_DIR / "db" / "sales.db"

engine = create_engine(f"sqlite:///{db_path}")

# Loading data
df = pd.read_sql("SELECT InvoiceDate, TotalPrice FROM Transactions", engine)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ======================
# DAILY MODEL
# ======================

daily = df.groupby("InvoiceDate")["TotalPrice"].sum().reset_index()
daily.columns = ["ds", "y"]

model_daily = Prophet()
model_daily.fit(daily)

future_daily = model_daily.make_future_dataframe(periods=90)
forecast_daily = model_daily.predict(future_daily)

# ======================
# MONTHLY MODEL WITH TRAIN/TEST SPLIT
# ======================

from sklearn.metrics import mean_absolute_error

# Przygotowanie danych
df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
monthly = df.groupby("Month")["TotalPrice"].sum().reset_index()
monthly.columns = ["ds", "y"]

# Podział 80/20
split_index = int(len(monthly) * 0.8)
train = monthly.iloc[:split_index]
test = monthly.iloc[split_index:]

# Trenowanie modelu
model_monthly = Prophet()
model_monthly.fit(train)

# Tworzymy prognozę tylko na okres testowy
future = model_monthly.make_future_dataframe(periods=len(test), freq="MS")
forecast = model_monthly.predict(future)

# Bierzemy tylko część testową
forecast_test = forecast.iloc[split_index:]

# Liczenie błędów
mae = mean_absolute_error(test["y"], forecast_test["yhat"])
mape = (abs((test["y"] - forecast_test["yhat"]) / test["y"])).mean() * 100

print("\n=== MODEL EVALUATION ===")
print(f"MAE: {mae:,.2f}")
print(f"MAPE: {mape:.2f}%")

# ======================
# CHARTS
# ======================

plt.figure(figsize=(14,6))
plt.plot(daily["ds"], daily["y"], label="Daily Actual", alpha=0.4)
plt.plot(forecast_daily["ds"], forecast_daily["yhat"], label="Daily Forecast")
plt.title("Daily Model")
plt.legend()
plt.show()

"""
plt.figure(figsize=(14,6))
plt.plot(monthly["ds"], monthly["y"], label="Monthly Actual", alpha=0.4)
plt.plot(forecast_monthly["ds"], forecast_monthly["yhat"], label="Monthly Forecast")
plt.title("Monthly Model")
plt.legend()
plt.show()
"""
