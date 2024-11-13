import yfinance as yf
import pandas as pd
import numpy as np
import os
 
# Set the working directory to the project directory
PROJECT_DIR = "/home/smehul/project/project/"
os.chdir(PROJECT_DIR)
 
def stochastic_optimization(tickers):
    def separate_tickers(tickers):
        us_tickers = []
        indian_tickers = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            currency = stock.info.get("currency", None)
            if currency == "INR":
                indian_tickers.append(ticker)
            elif currency == "USD":
                us_tickers.append(ticker)
        return us_tickers, indian_tickers
 
    def get_inr_to_usd_rate():
        inr_usd = yf.Ticker("INRUSD=X").history(period="1d")
        return inr_usd['Close'][0] if not inr_usd.empty else None
 
    def get_stock_data(tickers, inr_to_usd_rate):
        market_cap = []
        ticker=tickers[0]
        stock = yf.Ticker(ticker)
        currency = stock.info.get("currency", None)
        market_cap.append(stock.info.get("marketCap", None))
        if currency == "INR" and inr_to_usd_rate:
            market_cap[0] = market_cap[0] / inr_to_usd_rate
        hist = stock.history(period="1mo")  # 1 month daily data
        data=hist[['Close']].rename(columns={'Close': 'Close_'+ticker})
        data[ticker]=hist['Close'].pct_change().dropna()
        data.drop('Close_'+ticker,inplace=True,axis=1)
        tickers.pop(0)
 
        for ticker in tickers:
            i=0
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")  # 1 month daily data
            data[ticker]=hist['Close'].pct_change().dropna()   # Return in percentage
 
            # Get daily market capitalization
            market_cap.append(stock.info.get("marketCap", None))
            currency = stock.info.get("currency", None)
 
            if market_cap[i+1] and currency == "INR" and inr_to_usd_rate:
                market_cap[i+1]= market_cap[i+1] / inr_to_usd_rate
            elif market_cap[i+1] and currency == "USD":
                market_cap[i+1] = market_cap[i+1]
            i+=1
 
 
        return data, market_cap
 
    # Main Execution
    us_tickers, indian_tickers = separate_tickers(tickers)
 
    # Fetch the INR to USD conversion rate
    inr_to_usd_rate = get_inr_to_usd_rate()
    data, market_cap = get_stock_data(tickers, inr_to_usd_rate)
    returns=data
    risk_free_rate = [0,0]
    risk_free_rate[0]=((1+(0.069))**(1/365))-1
    risk_free_rate[1]=((1+(0.049))**(1/365))-1
    risk_free_rate=(risk_free_rate[0]*(len(indian_tickers))+risk_free_rate[1]*(len(us_tickers)))/(len(indian_tickers)+len(us_tickers))
 
    # Sharpe Ratio objective function
    def sharpe_ratio(weights, returns, risk_free_rate):
        portfolio_return = np.dot(weights, returns.mean())   # Annualized return
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() , weights)))  # Annualized volatility
        return (portfolio_return - risk_free_rate) / portfolio_std
 
    # Stochastic Gradient Descent for Sharpe Ratio maximization
    def stochastic_sharpe_optimization(market_cap,returns, risk_free_rate, num_iterations=5000, learning_rate=0.002, tolerance=1e-5):
        num_assets = returns.shape[1]
        num_init=10*(num_assets)
        # Start with equal weights
        sum_weights = np.sum(market_cap)
        weights = market_cap/sum_weights
        best_sharpe = sharpe_ratio(weights, returns, risk_free_rate)
        best_weights = weights
        L=[]
 
        for i in range(num_iterations):
            # Add a random small perturbation to weights for stochasticity
            perturbation = np.random.normal(0, 0.1, num_assets)
            new_weights = weights + learning_rate * perturbation
            new_weights = np.clip(new_weights, 0, 1)  # Ensure weights are between 0 and 1
            new_weights /= np.sum(new_weights)  # Normalize to sum to 1
 
            # Calculate new Sharpe Ratio
            new_sharpe = sharpe_ratio(new_weights, returns, risk_free_rate)
            
            # If the new Sharpe Ratio is better, update the best weights
            if new_sharpe > best_sharpe:
                best_sharpe = new_sharpe
                best_weights = new_weights
            
            # Update current weights for the next iteration
            weights = new_weights
 
            # Convergence check
            if np.abs(new_sharpe - best_sharpe) < tolerance:
                break
        L.append((best_weights,best_sharpe))
        for i in range(num_init):
            weights = np.random.rand(num_assets)
            weights /= np.sum(weights)
            best_sharpe = sharpe_ratio(weights, returns, risk_free_rate)
            best_weights = weights
            for i in range(num_iterations):
                # Add a random small perturbation to weights for stochasticity
                perturbation = np.random.normal(0, 0.1, num_assets)
                new_weights = weights + learning_rate * perturbation
                new_weights = np.clip(new_weights, 0, 1)  # Ensure weights are between 0 and 1
                new_weights /= np.sum(new_weights)  # Normalize to sum to 1
 
                # Calculate new Sharpe Ratio
                new_sharpe = sharpe_ratio(new_weights, returns, risk_free_rate)
                
                # If the new Sharpe Ratio is better, update the best weights
                if new_sharpe > best_sharpe:
                    best_sharpe = new_sharpe
                    best_weights = new_weights
                
                # Update current weights for the next iteration
                weights = new_weights
 
                # Convergence check
                if np.abs(new_sharpe - best_sharpe) < tolerance:
                    break
            L.append((best_weights,best_sharpe))
        best_sharpe=L[0][1]
 
        for i in range(1,len(L)):
            if L[i][1]>best_sharpe:
                best_sharpe=L[i][1]
                best_weights=L[i][0]
        return best_weights, best_sharpe      
        
 
    # Run optimization
    optimal_weights, optimal_sharpe = stochastic_sharpe_optimization(market_cap,returns, risk_free_rate)
    return optimal_weights, optimal_sharpe
 
if __name__ == "__main__":
    try:
        df = pd.read_excel(os.path.join(PROJECT_DIR, 'user_data.xlsx'))
        tickers = df['Stock Ticker'].tolist()
        print(df['Total Value ($)'])
        total_money = df['Total Value ($)'].sum()
        optimal_weights, optimal_sharpe = stochastic_optimization(tickers)
        print("sum of weights",sum(optimal_weights))
        print("total money",total_money)
        optimal_money_split = [total_money * weight for weight in optimal_weights]
 
        # Save optimal weights and Sharpe ratio to a CSV file
        tickers = df['Stock Ticker'].tolist()
        split_df = pd.DataFrame({
            'Ticker': tickers,
            'Optimal Money Split ($)': optimal_money_split
        })
        output_csv_path = os.path.join(PROJECT_DIR, 'optimal_money_split.csv')
        split_df.to_csv(output_csv_path, index=False)
        print(f"Optimal money split saved to '{output_csv_path}'.")
 
        # Save the Sharpe ratio
        optimal_sharpe_path = os.path.join(PROJECT_DIR, 'optimal_sharpe.txt')
        with open(optimal_sharpe_path, 'w') as f:
            f.write(str(optimal_sharpe))
        print(f"Optimal Sharpe Ratio saved to '{optimal_sharpe_path}'.")
        
        # Save optimal_money_split 
        optimal_money_split_path = os.path.join(PROJECT_DIR, 'optimal_money_split.txt')
        with open(optimal_money_split_path, 'w') as f:
            f.write(str(optimal_money_split))
        print(f"Optimal Money Split saved to '{optimal_money_split_path}'.")
 
    except Exception as e:
        print(f"An error occurred: {e}")
 
