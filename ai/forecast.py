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

# Daily aggregation
daily = df.groupby("InvoiceDate")["TotalPrice"].sum().reset_index()
daily.columns = ["ds", "y"]

# Model
model = Prophet()
model.fit(daily)

# Forecast for 90 days
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Charts
fig = model.plot(forecast)
plt.title("Sales Forecast (Next 90 Days)")
plt.show()
