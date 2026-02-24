import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

def _get_engine():
    BASE_DIR = Path(__file__).resolve().parent.parent
    db_path = BASE_DIR / "db" / "sales.db"
    return create_engine(f"sqlite:///{db_path}")

def load_full_data():
    engine = _get_engine()
    df = pd.read_sql("SELECT * FROM Transactions", engine)
    return df

def load_monthly_data():
    engine = _get_engine
    df = pd.read_sql("SELECT InvoiceDate, TotalPrice FROM Transactions", engine)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("Month")["TotalPrice"].sum().reset_index()
    monthly.columns = ["ds", "y"]

    return monthly