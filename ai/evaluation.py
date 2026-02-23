from sklearn.metrics import mean_absolute_error

def compute_metrics(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mape = (abs((true - predicted) / true)).mean() * 100
    return mae, mape

def evaluate_prophet(test, forecast, split_index):
    forecast_test = forecast.iloc[split_index:]

    return compute_metrics(test["y"], forecast_test["yhat"])

def naive_forecast(train, test):
    last_value = train["y"].iloc[-1]
    naive_predictions = [last_value] * len(test)
    return naive_predictions