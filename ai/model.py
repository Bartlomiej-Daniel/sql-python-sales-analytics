from prophet import Prophet

def train_model(train_data):
    model = Prophet()
    model.fit(train_data)
    return model

def make_forecast(model, periods, freq = "MS"):
    future = model.make_future_dataframe(periods = periods, freq = freq)
    forecast = model.predict(future)
    return forecast