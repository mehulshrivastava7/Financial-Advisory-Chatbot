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

import praw
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import nltk
import time
from tqdm import tqdm
from googleapiclient.discovery import build
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
import nltk
import sys
import io

# Suppress NLTK download output
original_stdout = sys.stdout
sys.stdout = io.StringIO()
nltk.download('vader_lexicon')
sys.stdout = original_stdout

# =============================================
# 1. Setup and Load Pre-trained BERT Model
# =============================================
 
# Download NLTK VADER lexicon (if still needed elsewhere)
nltk.download('vader_lexicon')
 
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
 
    for text in tqdm(reddit_df['Combined_Text'], desc="Analyzing Reddit Sentiments"):
        sentiment, confidence = predict_sentiment(text, tokenizer, model, device)
 
        # Assign numeric value based on sentiment
        if sentiment == 'Positive':
            sentiment_value = 1
        elif sentiment == 'Neutral':
            sentiment_value = 0
        else:  # Negative
            sentiment_value = -1
 
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
    for vid in tqdm(video_ids, desc="Fetching YouTube Comments"):
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
 
def main():
    # ----------------------------------------
    # User Inputs
    # ----------------------------------------
    
    # Reddit API Credentials
    reddit_client_id = 'YS3tFotOu28fhCV6qbn9Ag'         # Replace with your client ID
    reddit_client_secret = 'XNgpl9XAxCo326LKUgYKYFMK40o0eA' # Replace with your client secret
    reddit_user_agent = 'stock_sentiment_analysis by /u/Comfortable-Title817'  # Replace with your Reddit username
    
    # YouTube API Key
    youtube_api_key = 'AIzaSyCMqD1NfY7docBY30UWEwkThxQpzN-DEFo'  # Replace with your actual YouTube Data API key
    
    # Stock Information
    stock_ticker = 'RELIANCE.NS'  # Example: RELIANCE Industries
    company_name = 'Reliance Industries'
    
    # Subreddits to search
    subreddits = ['stocks', 'investing', 'IndianStockMarket']
    
    # ----------------------------------------
    # Reddit Sentiment Analysis
    # ----------------------------------------
    
    reddit_results , average = reddit_sentiment_analysis(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        user_agent=reddit_user_agent,
        stock_ticker=stock_ticker,
        company_name=company_name,
        subreddits=subreddits,
        reddit_limit=10  # Number of posts per query per subreddit
    )
    
    # ----------------------------------------
    # YouTube Sentiment Analysis
    # ----------------------------------------
    
    # youtube_results = youtube_sentiment_analysis(
        # api_key=youtube_api_key,
        # stock_symbol=stock_ticker,
        # company_name=company_name,
        # youtube_limit=5,    # Number of videos per query
        # comments_limit=10  # Number of comments per video
    # )
    
    # ----------------------------------------
    # Final Output
    # ----------------------------------------
    
    # print("=== Sentiment Analysis Completed ===\n")
    # print("Reddit Sentiment Analysis Results:")
    # print(reddit_results.head())
    
    # print("\nYouTube Sentiment Analysis Results:")
    # print(youtube_results.head())
    # Print only the 'Confidence' column to the console
 
    print(average)
 
    # Optionally, you can merge or further process these results as needed
 
if __name__ == "__main__":
    main()
