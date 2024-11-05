#Code for Risk Management:
from yfinance_data.py import get_stock_data

def calculate_risk_metrics(stock_data, market_data):
    risk_metrics = {}
    for symbol, prices in stock_data.items():
        # Calculate daily returns for stock and market
        stock_returns = prices.pct_change().dropna()
        market_returns = market_data.pct_change().dropna()
        
        # Calculate the standard deviation of returns 
        standard deviation = stock_returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate beta
        lr = LinearRegression()
        lr.fit(market_returns.values.reshape(-1, 1), stock_returns.values)
        beta = lr.coef_[0]
        
        risk_metrics[symbol] = {'volatility': standard deviation, 'beta': beta}
    return risk_metrics

data , market_data = get_stock_data(["AAPL"], datetime.now() - timedelta(days=10), datetime.now(), "^GSPC")

