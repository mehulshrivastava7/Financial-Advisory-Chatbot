"""This code is by mehul"""
import praw
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nltk
import time
from tqdm import tqdm
from googleapiclient.discovery import build
import os
import sys
import io
import yfinance as yf  # Import yfinance to fetch beta
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
 
# =============================================
# 0. Suppress Unwanted Logs
# =============================================
 
# Suppress TensorFlow and CUDA logs (if any)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
 
# Suppress NLTK download output
original_stdout = sys.stdout
sys.stdout = io.StringIO()
nltk.download('vader_lexicon', quiet=True)
sys.stdout = original_stdout
 
# Suppress other library logs
warnings.filterwarnings("ignore")
 
# =============================================
# 1. Setup and Load Pre-trained BERT Model
# =============================================
 
# Load the pre-trained BERT model and tokenizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForSequenceClassification.from_pretrained('sentiment_weights', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('sentiment_weights')
model.to(device)
model.eval()
 
# =============================================
# 2. Define Sentiment Prediction Function
# =============================================
 
def predict_sentiment(text, tokenizer, model, device):
    """
    Predicts the sentiment of a given text using a pre-trained BERT model.
    
    Parameters:
        text (str): The text to analyze.
        tokenizer: The BERT tokenizer.
        model: The pre-trained BERT model.
        device: The device to run the model on.
    
    Returns:
        tuple: (Sentiment Label, Confidence Score)
    """
    if not text or not isinstance(text, str):
        return 'neutral', 0.0
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    # Map numerical labels to sentiment
    label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = label_mapping.get(predicted_class.item(), 'Neutral')
    confidence_score = confidence.item()
    
    return sentiment, confidence_score
 
# =============================================
# 3. Reddit Sentiment Analysis Function
# =============================================
 
def reddit_sentiment_analysis(client_id, client_secret, user_agent, stock_ticker, company_name, subreddits, reddit_limit=100):
    """
    Fetches Reddit posts related to the stock and performs sentiment analysis.
 
    Parameters:
        client_id (str): Reddit API client ID.
        client_secret (str): Reddit API client secret.
        user_agent (str): Reddit API user agent.
        stock_ticker (str): Stock ticker symbol (e.g., 'RELIANCE.NS').
        company_name (str): Full company name (e.g., 'Reliance Industries').
        subreddits (list): List of subreddit names to search.
        reddit_limit (int): Number of posts to fetch per query per subreddit.
 
    Returns:
        tuple: (DataFrame containing comments and adjusted scores, average adjusted score)
    """
    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
 
    queries = [f"${stock_ticker.split('.')[0]}", company_name]
    posts = []
 
    for subreddit in subreddits:
        for query in queries:
            print(f"Searching in r/{subreddit} for '{query}'...")
            try:
                for submission in reddit.subreddit(subreddit).search(query, limit=reddit_limit, sort='new'):
                    posts.append({
                        'Combined_Text': f"{submission.title} {submission.selftext}"
                    })
            except Exception as e:
                print(f"Error fetching posts from r/{subreddit} with query '{query}': {e}")
                continue
 
    reddit_df = pd.DataFrame(posts)
    print(f"\nFetched {len(reddit_df)} Reddit posts.\n")
 
    # Perform sentiment analysis
    adjusted_scores = []
 
    for text in tqdm(reddit_df['Combined_Text'], desc="Analyzing Reddit Sentiments", disable=True):
        sentiment, confidence = predict_sentiment(text, tokenizer, model, device)
 
        # Assign numeric value based on sentiment
        if sentiment == 'Positive':
            sentiment_value = -1
        elif sentiment == 'Neutral':
            sentiment_value = 0
        else:  # Negative
            sentiment_value = 1
 
        # Calculate adjusted score
        adjusted_score = sentiment_value * confidence
        adjusted_scores.append(adjusted_score)
 
    # Add adjusted scores to DataFrame
    reddit_df['Adjusted_Score'] = adjusted_scores
 
    # Calculate the average of adjusted scores
    average_adjusted_score = reddit_df['Adjusted_Score'].mean()
 
    # Filter to only keep the relevant column
    reddit_output_df = reddit_df[['Combined_Text', 'Adjusted_Score']]
 
    # Save to CSV
    reddit_output_df.to_csv('reddit_stock_sentiment_adjusted_scores.csv', index=False)
    print("Reddit sentiment results saved to 'reddit_stock_sentiment_adjusted_scores.csv'.\n")
 
    return reddit_output_df, average_adjusted_score
 
# =============================================
# 4. YouTube Sentiment Analysis Function
# =============================================
 
def youtube_sentiment_analysis(api_key, stock_symbol, company_name, youtube_limit=5, comments_limit=100):
    """
    Fetches YouTube comments related to the stock and performs sentiment analysis.
    
    Parameters:
        api_key (str): YouTube Data API key.
        stock_symbol (str): Stock symbol (e.g., 'RELIANCE.NS').
        company_name (str): Full company name (e.g., 'Reliance Industries').
        youtube_limit (int): Number of videos to fetch per query.
        comments_limit (int): Number of comments to fetch per video.
    
    Returns:
        pd.DataFrame: DataFrame containing comments and their sentiment scores.
    """
    search_queries = [f"{stock_symbol.split('.')[0]}", company_name]
    video_ids = []
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    for query in search_queries:
        print(f"Searching YouTube for '{query}'...")
        try:
            search_response = youtube.search().list(
                q=query,
                part='id',
                type='video',
                maxResults=youtube_limit,
                order='date'  # Fetch recent videos
            ).execute()
    
            vids = [item['id']['videoId'] for item in search_response.get('items', [])]
            video_ids.extend(vids)
            print(f"Found {len(vids)} videos for query '{query}'.\n")
        except Exception as e:
            print(f"Error searching YouTube for query '{query}': {e}")
            continue
    
    print(f"Total videos found: {len(video_ids)}\n")
    
    # Fetch comments for each video
    all_comments = []
    for vid in tqdm(video_ids, desc="Fetching YouTube Comments", disable=True):
        try:
            comment_response = youtube.commentThreads().list(
                videoId=vid,
                part='snippet',
                maxResults=comments_limit,
                textFormat='plainText',
                order='relevance'
            ).execute()
    
            for item in comment_response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                # Check if the comment mentions the stock symbol or company name
                if stock_symbol.split('.')[0].lower() in comment.lower() or company_name.lower() in comment.lower():
                    sentiment, confidence = predict_sentiment(comment, tokenizer, model, device)
                    all_comments.append({
                        'Comment': comment,
                        'Sentiment': sentiment,
                        'Confidence': confidence
                    })
        except Exception as e:
            print(f"Error fetching comments for video {vid}: {e}")
            continue
        # To respect API rate limits
        time.sleep(1)
    
    youtube_df = pd.DataFrame(all_comments)
    print(f"\nFetched and analyzed {len(youtube_df)} relevant YouTube comments.\n")
    
    # Save to CSV
    youtube_df.to_csv('youtube_stock_sentiment.csv', index=False)
    print("YouTube sentiment results saved to 'youtube_stock_sentiment.csv'.\n")
    
    return youtube_df
 
# =============================================
# 5. Main Function
# =============================================
 
def for_one_stock(lele,lelecompany_name):
    # ----------------------------------------
    # User Inputs
    # ----------------------------------------
    
    # Reddit API Credentials
    reddit_client_id = 'YS3tFotOu28fhCV6qbn9Ag'          # Replace with your client ID
    reddit_client_secret = 'XNgpl9XAxCo326LKUgYKYFMK40o0eA' # Replace with your client secret
    reddit_user_agent = 'stock_sentiment_analysis by /u/Comfortable-Title817'  # Replace with your Reddit username
    
    # YouTube API Key
    youtube_api_key = 'AIzaSyCMqD1NfY7docBY30UWEwkThxQpzN-DEFo'  # Replace with your actual YouTube Data API key
    
    # Stock Information
    stock_ticker = lele # Example: RELIANCE Industries
    company_name = lelecompany_name
    
    # Subreddits to search
    subreddits = ['stocks', 'investing', 'IndianStockMarket']
    
    # ----------------------------------------
    # Reddit Sentiment Analysis
    # ----------------------------------------
    
    reddit_results, average_sentiment_score = reddit_sentiment_analysis(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent,
        stock_ticker=stock_ticker,
        company_name=company_name,
        subreddits=subreddits,
        reddit_limit=10  # Number of posts per query per subreddit
    )
    
    # ----------------------------------------
    # YouTube Sentiment Analysis (Optional)
    # ----------------------------------------
    
    # Uncomment the following lines if you want to perform YouTube sentiment analysis
    # youtube_results = youtube_sentiment_analysis(
    #     api_key=youtube_api_key,
    #     stock_symbol=stock_ticker,
    #     company_name=company_name,
    #     youtube_limit=5,    # Number of videos per query
    #     comments_limit=10   # Number of comments per video
    # )
    
    # ----------------------------------------
    # Fetch Beta Value from yFinance
    # ----------------------------------------
    
    try:
        ticker_obj = yf.Ticker(stock_ticker)
        beta = ticker_obj.info.get('beta', None)
        if beta is None:
            print("Beta value not found for the stock.")
            beta = 0.0  # Assign a default value or handle accordingly
    except Exception as e:
        print(f"Error fetching beta from yFinance: {e}")
        beta = 0.0  # Assign a default value or handle accordingly
    
    # ----------------------------------------
    # Calculate Final Risk Score
    # ----------------------------------------
    
    # Change the sign of the sentiment score
    adjusted_sentiment_score = -average_sentiment_score
    
    # Calculate weighted average: 80% beta and 20% adjusted sentiment score
    final_risk_score = 0.9 * beta + 0.1 * adjusted_sentiment_score
    
    # Determine if it's an increase or decrease
    # if final_risk_score > 0:
    #     risk_trend = "Increase"
    # elif final_risk_score < 0:
    #     risk_trend = "Decrease"
    # else:
    #     risk_trend = "No Change"
    
    # ----------------------------------------
    # Print Final Risk Score
    # ----------------------------------------
    
    print(f"\nFinal Risk Score for {stock_ticker}: {final_risk_score:.2f}")
    
    # Optionally, you can merge or further process these results as needed
 
# if __name__ == "__main__":
# for_one_stock('RELIANCE.NS','Reliance Industries')
# Add this code at the end of your existing file, after the for_one_stock function
 
def calculate_risk_scores(input_csv_path, output_csv_path):
    """
    Calculate risk scores for stocks in the input CSV and save results to output CSV.
    
    Args:
        input_csv_path (str): Path to the input CSV containing recommended stocks
        output_csv_path (str): Path to save the output CSV with risk scores
    """
    # Read the input CSV
    print(f"Reading stocks from {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Initialize new column for risk scores
    df['Risk_Score'] = 0.0
    
    # Calculate risk score for each stock
    print("\nCalculating risk scores for each stock...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ticker = row['Ticker']
        company_name = row['Company_Name']
        
        try:
            # Temporarily redirect stdout to capture the risk score
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Calculate risk score using for_one_stock
            for_one_stock(ticker, company_name)
            
            # Get the output
            output = sys.stdout.getvalue()
            
            # Restore stdout
            sys.stdout = original_stdout
            
            # Extract the risk score from the output
            risk_score_line = [line for line in output.split('\n') if "Final Risk Score" in line]
            if risk_score_line:
                risk_score = float(risk_score_line[0].split(":")[1].strip())
                df.at[idx, 'Risk_Score'] = risk_score
            
            # Clean up any temporary files created by the for_one_stock function
            if os.path.exists('reddit_stock_sentiment_adjusted_scores.csv'):
                os.remove('reddit_stock_sentiment_adjusted_scores.csv')
            if os.path.exists('youtube_stock_sentiment.csv'):
                os.remove('youtube_stock_sentiment.csv')
                
        except Exception as e:
            print(f"\nError processing {ticker}: {str(e)}")
            df.at[idx, 'Risk_Score'] = None
            continue
    
    # Save the results
    print(f"\nSaving results to {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    print("Done!")
 
# Add this at the very end of your file
if __name__ == "__main__":
    input_path = "recommended_stocks.csv"  # Path to your input CSV
    output_path = "recommended_stocks_with_risk.csv"  # Path for output CSV
    
    calculate_risk_scores(input_path, output_path)
