from data_loader import load_monthly_data
from model import train_model, make_forecast
from evaluation import evaluate_prophet, naive_forecast, compute_metrics

# Load data
monthly = load_monthly_data()

# Train/ test split
split_index = int(len(monthly) * 0.8)
train = monthly.iloc[:split_index]
test = monthly.iloc[split_index:]

# Prophet
model = train_model(train)
forecast = make_forecast(model, periods = len(test))
mae_p, mape_p = evaluate_prophet(test, forecast, split_index)

# Naive
naive_preds = naive_forecast(train, test)
mae_n, mape_n = compute_metrics(test["y"], naive_preds)

print("\n PROPHET MODEL")
print(f"MAE: {mae_p:,.2f}")
print(f"MAPE: {mape_p:,.2f}")

print("\n NAIVE MODEL")
print(f"MAE: {mae_n:,.2f}")
print(f"MAPE: {mape_n:,.2f}")


