import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from core.data_loader import load_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent
db_path = BASE_DIR / "db" / "sales.db"

engine = create_engine(f"sqlite:///{db_path}")

df = pd.read_sql("SELECT * FROM Transactions LIMIT 5", engine)
print(df.columns)