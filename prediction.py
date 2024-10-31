from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Load your trained LSTM model
model = load_model("stock_prediction_model_all.h5")  # Replace with your model's path
scaler = MinMaxScaler(feature_range=(0, 1))

class StockRequest(BaseModel):
    symbol: str
    date: str  # Format: 'YYYY-MM-DD'

class StockResponse(BaseModel):
    predicted_open: float
    predicted_high: float
    predicted_low: float
    predicted_close: float

@app.post("/predict", response_model=StockResponse)
async def predict_stock(request: StockRequest):
    # Fetch the last year of stock data
    stock_data = yf.download(request.symbol, period="1y", interval="1d")
    
    if stock_data.empty:
        raise HTTPException(status_code=404, detail="Stock symbol not found")

    # Ensure the stock data contains enough rows to predict the last 100 days
    if len(stock_data) < 100:
        raise HTTPException(status_code=400, detail="Not enough data to predict")

    # Get the last 100 days of stock data
    data = stock_data[['Open', 'High', 'Low', 'Close']].tail(100).values
    data_scaled = scaler.fit_transform(data)

    # Prepare the input data for prediction
    x_test = []
    x_test.append(data_scaled)  # Use the last 100 days of data
    x_test = np.array(x_test)

    # Make predictions
    y_predicted = model.predict(x_test)
    
    # Inverse transform the predictions
    y_predicted_inverse = scaler.inverse_transform(y_predicted)

    # Create a response
    return StockResponse(
        predicted_open=float(y_predicted_inverse[0][0]),
        predicted_high=float(y_predicted_inverse[0][1]),
        predicted_low=float(y_predicted_inverse[0][2]),
        predicted_close=float(y_predicted_inverse[0][3]),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
