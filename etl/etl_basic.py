import pandas as pd
from sqlalchemy import create_engine

#loading
df = pd.read_excel("../data/Online Retail.xlsx");

print("Data after loading: ")
print(df.head())

#cleaning
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("After cleaning: ", df.shape)

#creating a SQLite database
engine = create_engine("sqlite:///../db/sales.db")

#recording to the database
df.to_sql("Transaction", engine, if_exists="replace", index=False)

print("The data has been saved to the database")