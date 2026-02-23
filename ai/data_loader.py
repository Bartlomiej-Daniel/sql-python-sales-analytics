import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

def load_monthly_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    db_path = BASE_DIR / "db" /"sales.db"

    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql("SELECT InvoiceDate, TotalPrice FROM Transactions", engine)

    df["InvoiceData"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceData"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("Month")["TotalPrice"].sum().reset_index()
    monthly.columns = ["ds", "y"]

    return monthly