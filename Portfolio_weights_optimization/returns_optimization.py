#This file was written by Ramanan
import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
import os
import sys
#This file was written by Ramanan
PROJECT_DIR = "/home/ramanant/myenv/project/"
os.chdir(PROJECT_DIR)
if len(sys.argv) > 1:
    risk_tol = float(sys.argv[1])
else:
    risk_tol = 0.5
 
def convex_optimisation(tickers,risk_tol):
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
            elif currency == None:
                print(f"Couldn't find currency for {ticker}. Skipping...")
        return us_tickers, indian_tickers
    def softmaxm(x):
        exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    def softmaxv(x):
        exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    def timezone(us_tickers,indian_tickers):
        if len(us_tickers)>0 and len(indian_tickers)>0:
            ticker1=us_tickers[0]
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=30)

# Download data
            hist = yf.download(ticker1, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d')) # 1 month daily data
            data1=hist[['Close']].rename(columns={'Close': 'Close_'+ticker1})
            len1=data1.shape[0]
            ticker2=indian_tickers[0]
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=30)
            hist=yf.download(ticker2, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            data2=hist[['Close']].rename(columns={'Close': 'Close_'+ticker2})
            len2=data2.shape[0]
            if len1!=len2:
                timezone=1
            else:
                timezone=0
        else:
            timezone=0
        return timezone

    def get_inr_to_usd_rate():
        inr_usd = yf.Ticker("INRUSD=X").history(period="1d")
        
        return inr_usd['Close'][0] if not inr_usd.empty else None

    def get_stock_data(tickers, inr_to_usd_rate,t):
        market_cap = []
        ticker=tickers[0]
        stock = yf.Ticker(ticker)
        currency = stock.info.get("currency", None)
        market_cap.append(stock.info.get("marketCap", None))
        if currency == "INR" and inr_to_usd_rate:
            market_cap[0] = market_cap[0] * inr_to_usd_rate
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        

# Download data
        if t==1:
            if currency=="INR":
                start_date= start_date - timedelta(days=1)
            
        hist = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d')) # 1 month daily data
        data=hist[['Close']].rename(columns={'Close': 'Close_'+ticker})
        if currency == "INR" and inr_to_usd_rate:
            data['Close_'+ticker]=data['Close_'+ticker]*inr_to_usd_rate
        data[ticker]=data['Close_'+ticker].pct_change()
        data.drop('Close_'+ticker,inplace=True,axis=1)
        data.reset_index(drop=True)
        first=tickers.pop(0)
        
        i=0
        

        for ticker in tickers:
            currency = stock.info.get("currency", None)
            stock = yf.Ticker(ticker)
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=30)
            if t==1:
                if currency=="INR":
                    start_date= start_date - timedelta(days=1)
            hist = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d')) # 1 month daily data
            df=hist['Close']# 1 month daily data
            df.reset_index(drop=True)
            if currency == "INR" and inr_to_usd_rate:
                df=df*inr_to_usd_rate
            df=df.pct_change()
            
            data[ticker]=df.values
             # Return in percentage

            # Get daily market capitalization
            market_cap.append(stock.info.get("marketCap", None))
            

            if market_cap[i+1] and currency == "INR" and inr_to_usd_rate:
                market_cap[i+1]= market_cap[i+1] * inr_to_usd_rate
            elif market_cap[i+1] and currency == "USD":
                market_cap[i+1] = market_cap[i+1]
            
            i+=1
        data=data.dropna()
        

        return data, market_cap,first

    # Main Execution
    us_tickers, indian_tickers = separate_tickers(tickers)
    tickers=us_tickers+indian_tickers
    t=timezone(us_tickers,indian_tickers)

    # Fetch the INR to USD conversion rate
    inr_to_usd_rate = get_inr_to_usd_rate()
    
    if not inr_to_usd_rate:
        print("INR to USD rate not available. Exiting...")
        return
    data, market_cap,first = get_stock_data(tickers, inr_to_usd_rate,t)
    tickers.insert(0,first)
    length=len(market_cap)
    del_L=[]
    for i in range(length):
        if market_cap[i]==None:
            print(f"Market cap for {tickers[i]} not available. Skipping...")
            del_L.append(i)
    for i in del_L:
        market_cap[i]=0
        data.drop(tickers[i],inplace=True,axis=1)
        tickers[i]=0
    market_cap=[x for x in market_cap if x!=0]
    tickers=[x for x in tickers if x!=0]
    returns = data.mean().values
    cov_matrix = data.cov()
    std_dev = data.std()
    corr_matrix1 = (cov_matrix.div(std_dev, axis=0).div(std_dev, axis=1)).values
    corr_matrix2 = (corr_matrix1+1)/2
    corr_matrix=softmaxm(corr_matrix2)
    def stochastic_sharpe_optimization(market_cap,returns,corr_matrix, num_iterations=5000, learning_rate=0.002, tolerance=1e-5):
        num_assets = returns.shape[0]
        num_init=10*(num_assets)
        # Start with equal weights
        sum_weights = np.sum(market_cap)
        weights = market_cap/sum_weights
        best_returns = weights@ returns
        best_weights = weights
        risk=np.sqrt(np.dot(weights.T, softmaxv(np.dot(corr_matrix, weights))))
        L=[]
        if (risk <= risk_tol):
            L.append((best_weights,best_returns))
        for i in range(num_iterations):
            # Add a random small perturbation to weights for stochasticity
            perturbation = np.random.normal(0, 0.1, num_assets)
            new_weights = weights + learning_rate * perturbation
            new_weights = np.clip(new_weights, 0, 1)  # Ensure weights are between 0 and 1
            new_weights /= np.sum(new_weights)  # Normalize to sum to 1

            # Calculate new Sharpe Ratio
            new_returns = new_weights @ returns 
            
            # If the new Sharpe Ratio is better, update the best weights
            if new_returns > best_returns:
                best_returns = new_returns
                best_weights = new_weights
                if np.sqrt(np.dot(new_weights.T, softmaxv(np.dot(corr_matrix, new_weights))))<=risk_tol:
                    L.append((best_weights,best_returns))
                weights = new_weights
            
            # Update current weights for the next iteration
            

            # Convergence check
            

        
        for i in range(num_init):
            weights = np.random.rand(num_assets)
            weights /= np.sum(weights)
            best_returns = weights@ returns
            best_weights = weights
            if np.sqrt(np.dot(weights.T, softmaxv(np.dot(corr_matrix, weights))))<=risk_tol:
                L.append((best_weights,best_returns))
            for i in range(num_iterations):
                # Add a random small perturbation to weights for stochasticity
                perturbation = np.random.normal(0, 0.1, num_assets)
                new_weights = weights + learning_rate * perturbation
                new_weights = np.clip(new_weights, 0, 1)  # Ensure weights are between 0 and 1
                new_weights /= np.sum(new_weights)  # Normalize to sum to 1

                # Calculate new Sharpe Ratio
                new_returns = new_weights@ returns
                
                # If the new Sharpe Ratio is better, update the best weights
                if new_returns > best_returns:
                    best_returns = new_returns
                    best_weights = new_weights
                    if np.sqrt(np.dot(new_weights.T, softmaxv(np.dot(corr_matrix, new_weights))))<=risk_tol:
                        L.append((best_weights,best_returns))
                    weights = new_weights
                
                # Update current weights for the next iteration
                

                # Convergence check
                
            
            
        
        if len(L)!=0:
            best_returns=L[0][1]
        else:
            return [0]*len(market_cap),0

        for i in range(1,len(L)):
            if L[i][1]>best_returns:
                best_returns=L[i][1]
                best_weights=L[i][0]
        return best_weights, best_returns


        
        

    # Run optimization
    optimal_weights, optimal_returns= stochastic_sharpe_optimization(market_cap,returns,corr_matrix)

    print("Optimal Weights:", optimal_weights)
    print("Optimal Returns:", optimal_returns)
    return optimal_weights, optimal_returns,tickers
if __name__ == "__main__":
    try:
        df = pd.read_excel(os.path.join(PROJECT_DIR, 'user_data.xlsx'))
        tickers = df['Ticker'].tolist()
        print(df['Total Value ($)'])
        total_money = df['Total Value ($)'].sum()
        optimal_weights, optimal_returns,tickers = convex_optimisation(tickers,risk_tol)
        print("sum of weights",sum(optimal_weights))
        print("total money",total_money)
        optimal_money_split = [total_money * weight for weight in optimal_weights]
 
        # Save optimal weights and Sharpe ratio to a CSV file
        split_df = pd.DataFrame({
            'Ticker': tickers,
            'Optimal Money Split ($)': optimal_money_split
        })
        output_csv_path = os.path.join(PROJECT_DIR, 'optimal_money_split1.csv')
        split_df.to_csv(output_csv_path, index=False)
        print(f"Optimal money split saved to '{output_csv_path}'.")
 
        # Save the Sharpe ratio
        optimal_sharpe_path = os.path.join(PROJECT_DIR, 'optimal_sharpe_path1.txt')
        with open(optimal_sharpe_path, 'w') as f:
            f.write(str(optimal_returns))
        print(f"Optimal Sharpe Ratio saved to '{optimal_sharpe_path}'.")
        
        # Save optimal_money_split 
        optimal_money_split_path = os.path.join(PROJECT_DIR, 'optimal_money_split1.txt')
        with open(optimal_money_split_path, 'w') as f:
            f.write(str(optimal_money_split))
        print(f"Optimal Money Split saved to '{optimal_money_split_path}'.")
 
    except Exception as e:
        print(f"An error occurred: {e}")
