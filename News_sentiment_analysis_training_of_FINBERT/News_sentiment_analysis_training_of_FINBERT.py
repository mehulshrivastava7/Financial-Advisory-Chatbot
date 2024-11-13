# sentiment_analysis.py

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f"Using device: {device}")

# Load model and tokenizer
try:
    # Load the FinBERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model.to(device)
    model.eval()  # Set model to evaluation mode
    logging.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model/tokenizer: {e}")
    raise e

# Access the API key securely
NEWSAPI_KEY="d18da3e7ddb2465f8d2d5d4f3138d5f4" # Ensure NEWSAPI_KEY is set as an environment variable
if not NEWSAPI_KEY:
    logging.error("NewsAPI Key not found. Please set the NEWSAPI_KEY environment variable.")
    raise ValueError("NewsAPI Key not found.")

label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def fetch_stock_news(tickers, language='en', page_size=100):
    all_articles = []
    for ticker in tickers:
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': ticker,
            'language': language,
            'pageSize': page_size,
            'domains': 'economictimes.indiatimes.com,moneycontrol.com,reuters.com,bloomberg.com,cnbc.com,finance.yahoo.com,thehindubusinessline.com',
            'apiKey': NEWSAPI_KEY
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            logging.info(f"Fetched {len(articles)} articles for ticker '{ticker}'.")
            all_articles.extend(articles)
        except requests.exceptions.RequestException as e:
            logging.error(f"NewsAPI request failed for ticker '{ticker}': {e}")
    return all_articles

def predict_sentiment(text):
    try:
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

        sentiment = label_mapping[predicted_class.item()]
        confidence_score = confidence.item()

        return sentiment, confidence_score
    except Exception as e:
        logging.error(f"Error during sentiment prediction: {e}")
        return 'Neutral', 0.0  # Default sentiment on error

def analyze_sentiment(tickers, n=5):
    articles = fetch_stock_news(tickers)
    logging.info(f"Total articles fetched: {len(articles)}")

    if not articles:
        logging.warning("No articles fetched. Returning empty DataFrame.")
        return pd.DataFrame()

    sentiments = []
    confidences = []

    for idx, article in enumerate(articles, start=1):
        title = article.get('title', '').strip()
        description = article.get('description', '').strip()
        text = f"{title} {description}".strip()
        if text:
            sentiment, confidence = predict_sentiment(text)
            sentiments.append(sentiment)
            confidences.append(confidence)
            logging.debug(f"Article {idx}: Sentiment={sentiment}, Confidence={confidence}")
        else:
            sentiments.append('Neutral')
            confidences.append(0.0)
            logging.debug(f"Article {idx}: No text available. Assigned Neutral sentiment.")

    articles_df = pd.DataFrame(articles)
    articles_df['sentiment'] = sentiments
    articles_df['confidence'] = confidences

    top_articles = articles_df.sort_values(by='confidence', ascending=False).head(n)
    top_articles.to_csv('top_articles.csv', index=False)
    logging.info(f"Saved top {n} articles to 'top_articles.csv'.")

    return top_articles

