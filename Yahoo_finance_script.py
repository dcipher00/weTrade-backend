import yfinance as yf
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from io import BytesIO
import matplotlib.pyplot as plt
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

aws_secret_access_key = os.getenv("aws_secret_access_key")
aws_access_key_id = os.getenv("aws_access_key_id")


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


def EDA_analysis(data):
    data = data.drop(columns=['symbol'])

    # Initialize Boto3 S3 client
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    bucket_name = 'rjbigdataimages'

    # Store URLs of uploaded images
    uploaded_image_urls = []

    # Time Series Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['High'], label='High', marker='o')
    plt.plot(data['Date'], data['Low'], label='Low', marker='o')
    plt.plot(data['Date'], data['Open'], label='Open', marker='o')
    plt.plot(data['Date'], data['Close'], label='Close', marker='o')
    plt.title('Stock Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    upload_plot(plt, 'time_series.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    # Histograms
    data.hist(figsize=(10, 6))
    plt.tight_layout()
    upload_plot(plt, 'hist.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    print("All images uploaded successfully to S3.")
    print("Uploaded Image URLs:")
    file_urls_dict = {}
    for url in uploaded_image_urls:
        file_name = url.split("/")[-1].split(".")[0]
        file_urls_dict[file_name] = url

    return file_urls_dict


def upload_plot(plt, filename, s3, bucket_name, uploaded_image_urls):
    # Save plot to BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Rewind the buffer

    # Upload to S3
    s3.upload_fileobj(img_data, bucket_name, filename)

    # Create public URL
    image_url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"
    uploaded_image_urls.append(image_url)

    print(f"Image '{filename}' uploaded successfully to S3.")
    print(f"URL: {image_url}")



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