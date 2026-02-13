import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

db_path = BASE_DIR / "db" / "sales.db"

engine = create_engine(f"sqlite:///{db_path}")

count = pd.read_sql("SELECT COUNT(*) as total FROM Transactions", engine)

print("Number of records:")
print(count)

customers = pd.read_sql( "SELECT COUNT(DISTINCT CustomerID) as customers FROM Transactions", engine)

print("Number of unique customers:")
print(customers)

monthly_sales = pd.read_sql("""
    SELECT 
        strftime('%Y-%m', InvoiceDate) AS month,
        SUM(TotalPrice) AS revenue
    FROM Transactions
    GROUP BY month
    ORDER BY month
""", engine)

print("Monthly Sales:")
print(monthly_sales.head())
