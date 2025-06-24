def get_history(ticker="AAPL", period="7mo"):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval="1d")
    df.to_csv(f"stocks/data/raw/{ticker}.csv")
    return df

if __name__ == "__main__":
    # 1) call the function
    df = get_history("AAPL", period="6mo")
    # 2) give yourself feedback
    print(f"Downloaded {len(df)} rows for AAPL; saved to data/raw/AAPL.csv")
