import yfinance as yf
import pandas as pd
import argparse 

def download_sp500_data(start_date, end_date, output_file):
    # Download historical data for the S&P 500 (ticker symbol: ^GSPC)
    sp500 = yf.Ticker("^GSPC")
    df = sp500.history(start=start_date, end=end_date)
    
    # Export the data to a CSV file
    df.to_csv(output_file)
    print(f"Data successfully downloaded and saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')
    parser.add_argument('--output_file', type=str, default='sp500_data.csv', help='Output CSV file name')
    args = parser.parse_args()
    
    download_sp500_data(args.start_date, args.end_date, args.output_file)