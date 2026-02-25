from core.data_loader import load_monthly_data
from modeling.forecasting.models import (
    prophet_model,
    linear_regression_model,
    naive_model
)
from modeling.forecasting.evaluation import compute_metrics


# Load data
monthly = load_monthly_data()

# Train/ test split
split_index = int(len(monthly) * 0.8)
train = monthly.iloc[:split_index]
test = monthly.iloc[split_index:]

# Prophet 
prophet_preds = prophet_model(train, test, split_index)
mae_p, mape_p = compute_metrics(test["y"], prophet_preds)

# Linear regression
lr_preds = linear_regression_model(train, test)
mae_lr, mape_lr = compute_metrics(test["y"], lr_preds)

# Naive
naive_preds = naive_model(train, test)
mae_n, mape_n = compute_metrics(test["y"], naive_preds)

print("\nMODEL COMPARISON")
print(f"Prophet MAPE: {mape_p:.2f}%")
print(f"Linear  MAPE: {mape_lr:.2f}%")
print(f"Naive   MAPE: {mape_n:.2f}%")
