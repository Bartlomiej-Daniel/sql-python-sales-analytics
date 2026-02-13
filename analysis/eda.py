import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
db_path = BASE_DIR / "db" / "sales.db"

engine = create_engine(f"sqlite:///{db_path}")

df = pd.read_sql("SELECT InvoiceDate, TotalPrice FROM Transactions", engine)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Month"] = df["InvoiceDate"].dt.to_period("M")
monthly = df.groupby("Month")["TotalPrice"].sum()

# Chart
plt.figure(figsize=(12,6))
monthly.plot()
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
