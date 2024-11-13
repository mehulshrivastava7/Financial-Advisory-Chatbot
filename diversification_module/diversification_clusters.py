"""This code is by Mehul Shrivastava"""

"""


This idea is to create a diversified stock portfolio by performing clustering analysis.


Motivation:

The goal of this module is to enhance portfolio diversification by identifying clusters of stocks based on various financial and market features. 

The output of this is used in the recommendation module. 

The module helps us identify the clusters and then recommend the stocks from other clusters which are performing good. All those stocks are taken as input in the recommendation module and then it chooses the stocks from them 
to give the final output.

References:


I am not using anything directly from these papers, but these are what I came across initially which helped me choose the features and discuss importance of diversifications in detail. 

1. "Creating Diversified Portfolios Using Cluster Analysis" by Karina Marvin:

   - Discusses the importance of diversification and how clustering can be used to achieve a balanced portfolio.
   
2. https://medium.com/@facujallia/stock-classification-using-k-means-clustering-8441f75363de

   - I am not sure if a article like this is a source which I can use, if not then below I described a rationale of using these features.

Feature Selection Rationale:

- **Sector:** Categorical variable indicating the industry sector of the stock.
- **Market Cap:** Represents the size of the company; a key indicator of the stock's risk and potential growth.
- **Volatility:** A measure of the stock's risk and stability.

Methodology:
------------
The script follows these main steps:
1. **Data Collection:** Fetches historical stock data using yFinance in batches to handle API rate limits.
2. **Data Preprocessing:** Handles missing values, encodes categorical features, and scales numerical features.
3. **Optimal Clustering:** Determines the optimal number of clusters using the Elbow Method and Silhouette Score.
4. **Clustering Analysis:** Performs K-Means clustering on the preprocessed stock data.
5. **Portfolio Analysis:** Analyzes the user's current stock portfolio and identifies underrepresented clusters.
6. **Stock Recommendation:** Recommends top-performing stocks from unrepresented clusters to enhance diversification.

Output:
-------
1. A CSV file (`indian_stocks_clusters.csv`) containing the clustering results for each stock.

                                                                              """











import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
from tqdm import tqdm
 
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
 
# -------------------------------
# 1. Load Stock Tickers from CSV
# -------------------------------
 
def load_stock_tickers(csv_path):
    """
    Load stock tickers from a CSV file.
 
    Parameters:
        csv_path (str): Path to the CSV file containing stock tickers.
 
    Returns:
        list: List of stock tickers.
    """
    try:
        df = pd.read_csv(csv_path)
        # Ensure the 'SYMBOL' column exists
        if 'SYMBOL' not in df.columns:
            raise ValueError("CSV file must contain a 'SYMBOL' column.")
        
        # Extract all unique symbols
        tickers = df['SYMBOL'].dropna().unique().tolist()
        print(f"Loaded {len(tickers)} tickers from '{csv_path}'.")
        return tickers
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' does not exist.")
        return []
    except Exception as e:
        print(f"Error loading tickers: {e}")
        return []
 
# ----------------------------------------
# 2. Fetch Stock Data in Batches
# ----------------------------------------
 
def fetch_and_save_stock_data(tickers, batch_size=200, start_date='2023-05-01', end_date='2024-11-09', output_csv='sample_indian_stocks_data_full.csv'):
    """
    Fetch stock data in batches and append to a CSV file.
 
    Parameters:
        tickers (list): List of stock tickers.
        batch_size (int): Number of tickers to process in each batch.
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        output_csv (str): Path to the output CSV file.
    """
    # Initialize the CSV file with headers if it doesn't exist
    if not os.path.exists(output_csv):
        headers = ['Ticker', 'Sector', 'Market Cap', 'P/E Ratio', 'Average Return', 'Volatility']
        with open(output_csv, 'w') as f:
            f.write(','.join(headers) + '\n')
        print(f"Created '{output_csv}' with headers.")
    else:
        print(f"Appending to existing '{output_csv}'.")
 
    total_tickers = len(tickers)
    total_batches = int(np.ceil(total_tickers / batch_size))
    
    print(f"\nFetching data in {total_batches} batches of {batch_size} tickers each...\n")
    
    for i in tqdm(range(0, total_tickers, batch_size), desc="Processing Batches"):
        batch_tickers = tickers[i:i + batch_size]
        try:
            # Fetch historical price data for the batch
            for ticker in batch_tickers:
                try:
                    print(f"Processing {ticker}...")
                    stock = yf.Ticker(ticker)
                    
                    # Fetch historical price data
                    hist = stock.history(start=start_date, end=end_date)
                    if hist.empty:
                        print(f"No historical data found for {ticker}. Skipping.")
                        continue
                    
                    # Calculate daily returns
                    hist['Daily Return'] = hist['Close'].pct_change()
                    avg_return = hist['Daily Return'].mean()
                    volatility = hist['Daily Return'].std()
                    
                    # Fetch financial info
                    info = stock.info
                    sector = info.get('sector', 'Unknown')
                    market_cap = info.get('marketCap', np.nan)
                    pe_ratio = info.get('trailingPE', np.nan)
                    
                    # Prepare row data
                    row = {
                        'Ticker': ticker,
                        'Sector': sector,
                        'Market Cap': market_cap if not pd.isna(market_cap) else '',
                        'P/E Ratio': pe_ratio if not pd.isna(pe_ratio) else '',
                        'Average Return': avg_return,
                        'Volatility': volatility
                    }
                    
                    # Append to CSV
                    with open(output_csv, 'a') as f:
                        f.write(f"{row['Ticker']},{row['Sector']},{row['Market Cap']},{row['P/E Ratio']},{row['Average Return']},{row['Volatility']}\n")
                    
                    print(f"Data fetched and saved for {ticker}.\n")
                    
                    # Optional: Sleep to respect API rate limits
                    time.sleep(0.1)
                
                except Exception as e:
                    print(f"Error processing {ticker}: {e}\n")
                    continue
            
            # Optional: Sleep between batches
            time.sleep(1)
        
        except Exception as e:
            print(f"Error fetching batch starting at index {i}: {e}")
            continue
    
    print(f"\nData fetching completed. Consolidated data saved to '{output_csv}'.")
 
# ----------------------------------------
# 3. Preprocess the Consolidated Data
# ----------------------------------------
 
def preprocess_data(input_csv='sample_indian_stocks_data.csv'):
    print("\nPreprocessing data...")
 
    # Load the consolidated CSV
    df = pd.read_csv(input_csv)
    print(f"Loading the latest diversification metrics...")
 
    # 3.1 Handle Missing Values
    # essential_columns = ['Sector', 'Market Cap', 'P/E Ratio', 'Average Return', 'Volatility']
    essential_columns = ['Sector', 'Market Cap','Volatility']
    df_clean = df.dropna(subset=essential_columns).reset_index(drop=True)
 
    # 3.2 Encode Categorical Variables (Sector) using pd.get_dummies
    sector_encoded_df = pd.get_dummies(df_clean['Sector'], prefix='Sector')
 
    # 3.3 Concatenate Encoded Columns with Main DataFrame
    df_final = pd.concat([df_clean.drop('Sector', axis=1), sector_encoded_df], axis=1)
 
    # 3.4 Handle any potential NaN or infinite values in the numerical columns
    numerical_features = ['Market Cap','Volatility']
    
    # Replace infinities with NaN, then drop rows with NaN in these columns
    df_final[numerical_features] = df_final[numerical_features].replace([np.inf, -np.inf], np.nan)
 
    # Drop rows where any of the numerical features have NaN values
    df_final = df_final.dropna(subset=numerical_features)
 
    # 3.5 Feature Scaling
    scaler = StandardScaler()
    df_final[numerical_features] = scaler.fit_transform(df_final[numerical_features])
 
    print("Preprocessing completed.")
    return df_final
 
# ----------------------------------------
# 4. Determine Optimal Number of Clusters
# ----------------------------------------
 
def determine_optimal_clusters(X, max_k=10):
    """
    Determine the optimal number of clusters using the Elbow Method.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        max_k (int): Maximum number of clusters to try.

    Returns:
        int: Optimal number of clusters.
    """
    wcss = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Elbow method to determine optimal K
    optimal_k = 2
    for i in range(1, len(wcss) - 1):
        if (wcss[i - 1] - wcss[i]) < (wcss[i] - wcss[i + 1]):
            optimal_k = i + 2
            break

    return optimal_k

 
# ----------------------------------------
# 5. Perform Clustering
# ----------------------------------------
 
def perform_clustering(X, n_clusters):
    """
    Perform K-Means clustering.
 
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        n_clusters (int): Number of clusters.
 
    Returns:
        KMeans: Fitted KMeans object.
        np.ndarray: Cluster labels.
    """
    # print(f"\nPerforming K-Means clustering with K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X)
    # print("Clustering completed.")
    return kmeans, labels
 
# ----------------------------------------
# 6. Save Clustering Results
# ----------------------------------------
 
def save_clustering_results(df, labels, output_path='indian_stocks_clusters.csv'):
    """
    Save the clustering results to a CSV file.
 
    Parameters:
        df (pd.DataFrame): Preprocessed stock data.
        labels (np.ndarray): Cluster labels.
        output_path (str): Path to save the CSV file.
    """
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    df_with_clusters.to_csv(output_path, index=False)
    # print(f"\nClustering results saved to '{output_path}'.")
 
# ----------------------------------------
# 7. Analyze User Portfolio and Recommend Stocks
# ----------------------------------------
 
def analyze_user_portfolio(clustered_df, user_portfolio, top_n=10):
    """
    Analyze the user's portfolio clusters and recommend stocks from other clusters.
 
    Parameters:
        clustered_df (pd.DataFrame): DataFrame with clustering results.
        user_portfolio (list): List of user's stock tickers.
        top_n (int): Number of stock recommendations per cluster.
 
    Returns:
        pd.DataFrame: Recommended stocks from clusters the user is not invested in.
    """
    print("\nAnalyzing user portfolio...")
 
    # Get company info using yfinance for each ticker
    def get_company_info(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'Company_Name': info.get('longName', 'N/A'),
                'Sector': info.get('sector', 'N/A')
            }
        except:
            return {
                'Company_Name': 'N/A',
                'Sector': 'N/A'
            }
 
    # Filter user's stocks
    user_stocks = clustered_df[clustered_df['Ticker'].isin(user_portfolio)]
 
    if user_stocks.empty:
        print("No matching stocks found in the clustering results for the user portfolio.")
        return pd.DataFrame()
 
    # Display user's stocks and their clusters
    print("\nUser Portfolio Clusters:")
    print(user_stocks[['Ticker', 'Cluster']])
 
    # Identify clusters present in user's portfolio
    user_clusters = user_stocks['Cluster'].unique()
    print(f"\nUser's clusters: {user_clusters}")
 
    # Find clusters where user is not invested
    all_clusters = clustered_df['Cluster'].unique()
    unrepresented_clusters = [c for c in all_clusters if c not in user_clusters]
    print(f"\nClusters without user investment: {unrepresented_clusters}")
 
    if not unrepresented_clusters:
        print("User has investments in all available clusters.")
        return pd.DataFrame()
 
    # Get recommendations from each unrepresented cluster
    recommendations_list = []
    
    for cluster in unrepresented_clusters:
        # Get stocks from this cluster
        cluster_stocks = clustered_df[clustered_df['Cluster'] == cluster].copy()
        
        # Sort by Average Return (descending) and get top N stocks
        top_cluster_stocks = cluster_stocks.sort_values(
            by='Average Return',
            ascending=False
        ).head(top_n)
        
        # Add cluster information
        top_cluster_stocks['Recommendation_Reason'] = f'Top performer from Cluster {cluster}'
        recommendations_list.append(top_cluster_stocks)
 
    # Combine all recommendations
    if recommendations_list:
        final_recommendations = pd.concat(recommendations_list, ignore_index=True)
        
        # Add company information
        company_info = []
        print("\nFetching company information...")
        for ticker in final_recommendations['Ticker']:
            info = get_company_info(ticker)
            company_info.append(info)
        
        # Add new columns
        final_recommendations['Company_Name'] = [info['Company_Name'] for info in company_info]
        final_recommendations['Sector'] = [info['Sector'] for info in company_info]
        
        # Select and reorder columns for output
        columns_to_keep = [
            'Ticker',
            'Company_Name',
            'Sector',
            'Cluster',
            'Average Return',
            'Volatility',
            'Market Cap',
            'P/E Ratio',
        ]
        final_recommendations = final_recommendations[
            [col for col in columns_to_keep if col in final_recommendations.columns]
        ]
        
        # Save recommendations to CSV
        output_file = 'recommended_stocks.csv'
        final_recommendations.to_csv(output_file, index=False)
        print(f"\nSaved {len(final_recommendations)} stock recommendations to '{output_file}'")
        
        # Display summary
        print("\nRecommendations Summary:")
        summary = final_recommendations.groupby('Cluster').size()
        print(f"Number of recommendations per cluster:\n{summary}")
        
        return final_recommendations
    
    return pd.DataFrame()
 
# ----------------------------------------
# 8. Visualize Clusters using PCA
# ----------------------------------------
 
def plot_clusters(df, n_clusters):
    """
    Visualize clusters using PCA for dimensionality reduction.
 
    Parameters:
        df (pd.DataFrame): Preprocessed data with cluster labels.
        n_clusters (int): Number of clusters.
    """
    from sklearn.decomposition import PCA
 
    # Separate features and cluster labels
    X = df.drop(['Ticker', 'Cluster'], axis=1)
    labels = df['Cluster']
 
    # Apply PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(X)
 
    # Create a DataFrame with principal components and cluster labels
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
 
    # Plot the clusters
    # plt.figure(figsize=(10, 7))
    # sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.7)
    # plt.title('Stock Clusters Visualization using PCA')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.legend(title='Cluster')
    # plt.show()
 
# ----------------------------------------
# 9. Main Execution Flow
# ----------------------------------------
 
def main(sample_user_portfolio):
    input_csv = 'sample_indian_stocks_data.csv'
    
    # Step 1: Preprocess the consolidated data
    df_preprocessed = preprocess_data(input_csv=input_csv)
    if df_preprocessed.empty:
        return
    
    # Step 2: Determine the optimal number of clusters
    X = df_preprocessed.drop(['Ticker'], axis=1)
    optimal_k = determine_optimal_clusters(X, max_k=10)
    print(f"Optimal number of clusters determined: {optimal_k}")
    
    # Step 3: Perform clustering
    kmeans, labels = perform_clustering(X, optimal_k)
    
    # Step 4: Save clustering results
    save_clustering_results(df_preprocessed, labels)
    
    # Step 5: Analyze user portfolio and recommend stocks
    clustered_df = pd.read_csv('indian_stocks_clusters.csv')
    recommendations = analyze_user_portfolio(clustered_df, sample_user_portfolio, top_n=10)
    if not recommendations.empty:
        recommendations.to_csv('recommended_stocks.csv', index=False)

if __name__ == "__main__":
    sample_user_portfolio = ['20MICRONS.NS', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    main(sample_user_portfolio)
