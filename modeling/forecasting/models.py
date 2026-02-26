from prophet import Prophet
from sklearn.linear_model import LinearRegression
import numpy as np

def prophet_model(train, test, split_index):
    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods = len(test), freq = "MS")
    forecast = model.predict(future)

    return forecast.iloc[split_index:]["yhat"]

def linear_regression_model(train, test):
    train_lr = train.copy()
    test_lr = test.copy()

    train_lr["t"] = np.arange(len(train_lr))
    test_lr["t"] = np.arange(len(train_lr), len(train_lr) + len(test_lr))
    
    lr_model = LinearRegression()
    lr_model.fit(train_lr[["t"]], train_lr["y"])

    return lr_model.predict(test_lr[["t"]])

def naive_model(train, test):
    last_value = train["y"].iloc[-1]
    return [last_value] * len(test)


