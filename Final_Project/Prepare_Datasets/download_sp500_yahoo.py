
import yfinance as yf
import pandas as pd

def download_sp500_data(start_date, end_date, output_file):
    # Download historical data for the S&P 500 (ticker symbol: ^GSPC)
    sp500 = yf.Ticker("^GSPC")
    df = sp500.history(start=start_date, end=end_date)
    
    # Export the data to a CSV file
    df.to_csv(output_file)
    print(f"Data successfully downloaded and saved to {output_file}")

if __name__ == '__main__':
    start_date = '2014-01-01'
    end_date = '2024-01-01'
    output_file = 'sp500_data.csv'
    download_sp500_data(start_date, end_date, output_file)