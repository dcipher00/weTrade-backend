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

    # Volume Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Volume'], label='Volume', color='orange', marker='o')
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    upload_plot(plt, 'volume_vs_date.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data[['Open', 'High', 'Low', 'Close']])
    plt.title('Box Plot of Prices')
    upload_plot(plt, 'Box_plots.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    # Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Seasonal Plot (Assuming monthly data)
    data['Month'] = data['Date'].dt.month
    seasonal_data = data.groupby('Month')['High'].mean()
    seasonal_data.plot(kind='bar')
    plt.title('Average High Price by Month')
    plt.xlabel('Month')
    plt.ylabel('Average High Price')
    upload_plot(plt, 'seasonal_plot.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    # Autocorrelation Plot
    plot_acf(data['High'], lags=20)
    upload_plot(plt, 'ACR.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    # Partial Autocorrelation Plot
    plot_pacf(data['High'], lags=20)
    upload_plot(plt, 'PCR.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    # Seasonal Plot (Assuming monthly data)
    data['Month'] = data['Date'].dt.month
    seasonal_data = data.groupby('Month')['High'].mean()
    seasonal_data.plot(kind='bar')
    plt.title('Average High Price by Month')
    plt.xlabel('Month')
    plt.ylabel('Average High Price')
    upload_plot(plt, 'seasonality.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    # Rolling Statistics
    rolling_mean = data['High'].rolling(window=30).mean()  # 30-day rolling mean
    rolling_std = data['High'].rolling(window=30).std()  # 30-day rolling standard deviation
    plt.plot(data['Date'], data['High'], label='High')
    plt.plot(data['Date'], rolling_mean, label='30-Day Rolling Mean')
    plt.plot(data['Date'], rolling_std, label='30-Day Rolling Std')
    plt.legend()
    upload_plot(plt, 'Rolling_mean.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    # Density Plot
    data['High'].plot(kind='kde')
    plt.title('Kernel Density Estimate of High Prices')
    plt.xlabel('Price')
    plt.ylabel('Density')
    upload_plot(plt, 'KDE_high.png', s3, bucket_name, uploaded_image_urls)
    plt.close()

    # Lag Scatter Plot
    plt.scatter(data['High'], data['High'].shift(-1))
    plt.title('Lag Scatter Plot')
    plt.xlabel('High (t)')
    plt.ylabel('High (t+1)')
    upload_plot(plt, 'Scatter.png', s3, bucket_name, uploaded_image_urls)
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