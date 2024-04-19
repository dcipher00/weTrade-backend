import yfinance as yf
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


def get_stock_info(symbol):
    """
    Retrieve detailed information and historical OHLC prices for a given stock symbol.
    Args:
    - symbol (str): Stock symbol for which to retrieve information.
    Returns:
    - info (dict): Detailed information about the stock.
    - history (DataFrame): Historical OHLC prices and other financial data.
    - df_info (DataFrame): Returning dataframe from info dictonary
    """
    # Create Ticker object for the specified symbol
    ticker = yf.Ticker(symbol)
    # Get detailed information about the stock symbol
    info = ticker.info
    # Get historical OHLC prices and other financial data
    history = ticker.history(period="6mo")
    history.reset_index(drop=False, inplace=True)
    history['Date'] = history['Date'].dt.date
    history['symbol'] = symbol

    # Create DataFrame for detailed information
    keys = list(info.keys())
    values = list(info.values())
    df_info = pd.DataFrame({'Key': keys, 'Value': values})
    return df_info, history


# META
# symbol = "META"
# info_meta, history_meta = get_stock_info(symbol)
# # GOOGLE
# symbol = "GOOG"
# info_google, history_google = get_stock_info(symbol)
#
# # AMZN
# symbol = "AMZN"
# info_amzon, history_amazon = get_stock_info(symbol)
# # APPLE
# symbol = "AAPL"
# info_apple, history_apple = get_stock_info(symbol)
#
# symbol = "DNN"
# info_meta, history_dnn = get_stock_info(symbol)

# history_meta.tail()
# data = pd.concat([history_meta, history_amazon, history_apple, history_google, history_dnn], axis=0)

# df = data.reset_index()
# df = df.drop(columns=['Dividends', 'Stock Splits', 'index'])
# df.columns


# Function to predict next day's high and low prices for each company
def predict_next_day_prices(company_df):
    # Shift the high and low prices to get next day's high and low
    company_df['Next_High'] = company_df['High'].shift(-1)
    company_df['Next_Low'] = company_df['Low'].shift(-1)
    print("company next data", company_df['Next_Low'])
    # Drop NaN rows resulting from the shift
    company_df.dropna(inplace=True)

    # Features and target variables
    X = company_df[['Open', 'Close', 'Low', 'High', 'Volume']]
    y_high = company_df['Next_High']
    y_low = company_df['Next_Low']

    # Fit autoregressive model for high price
    model_high = AutoReg(y_high, lags=1).fit()

    # Fit autoregressive model for low price
    model_low = AutoReg(y_low, lags=1).fit()

    # Predict high and low prices for the next day
    predicted_high = model_high.predict(start=len(company_df), end=len(company_df))
    predicted_low = model_low.predict(start=len(company_df), end=len(company_df))

    return predicted_high.values[0], predicted_low.values[0]


# Predict next day's high and low prices for each company
# companies = df['symbol'].unique()
# predictions = {}
# for company in companies:
#     company_df = df[df['symbol'] == company]
#     predicted_high, predicted_low = predict_next_day_prices(company_df)
#     predictions[company] = {'Predicted_High': predicted_high, 'Predicted_Low': predicted_low}
#
# # Print predictions for each company
# for company, values in predictions.items():
#     print(f"Company: {company}, Predicted High: {values['Predicted_High']}, Predicted Low: {values['Predicted_Low']}")