"""This code is by Mehul Shrivastava"""

"""
Description:
------------
This script is designed to evaluate the risk of individual stocks by combining traditional financial metrics (beta value) with public sentiment analysis. 
The idea is to enhance stock risk assessment by integrating sentiment scores derived from Reddit and YouTube discussions.(Initially I wanted to use Twitter API but it is not free, so I have using Reddit 
and youtube, the youtube API key code is there but the API key is expired so I commented the code, Reddit key is still working as of now(13 November).

Motivation:
-----------
The primary goal of this module is to calculate a risk score for each stock based on two main factors:
1. **Beta Value:** Represents the stock's volatility relative to the market. A higher beta indicates higher risk and potential for greater returns.
2. **Sentiment Analysis:** Incorporates public sentiment from Reddit and YouTube. 
The rationale is that investor sentiment significantly impacts stock performance, and thus, it should be considered when assessing stock risk.

Why Calculate Risk Scores?
--------------------------
The calculated risk scores are intended to be used in a broader user-risk-assessment module. 
The idea is to tailor stock recommendations based on the user's risk tolerance. If a user has a high risk tolerance, riskier stocks can be recommended;
otherwise, safer stocks will be suggested. This approach aims to provide a more personalized investment recommendation system.

This script is inspired by several research papers that emphasize the importance of sentiment analysis in financial models. One notable reference is:
"Adding Sentiment to Multi-Factor Equity Strategies" by Corne Reniers and Svetlana Borovkova, which highlights the significance of sentiment scores in risk assessment.

Data Source:
------------
The list of stocks is provided via a CSV file, which includes most of the indian stocks available on yfinance.(around 1200 stocks are there in the csv will be there in the references section.) 
This file includes various stock metrics like Market Cap, P/E Ratio, Average Return, Volatility, and Sector information. 
The dataset has been preprocessed with one-hot encoding for certain categorical features, gathered through API calls to yFinance (handled in batches due to API call limits).

Overview of the Code:
---------------------
The script is structured into the following main components:

0. **Setup:** Suppresses unwanted logs and loads the pre-trained BERT model for sentiment analysis.
1. **Sentiment Analysis Function:** Uses a BERT model to predict the sentiment of given text (Reddit posts and YouTube comments).
2. **Reddit Sentiment Analysis:** Fetches recent Reddit posts related to the stock and analyzes their sentiment.
3. **YouTube Sentiment Analysis:** (Optional) Fetches YouTube comments related to the stock and analyzes their sentiment.
4. **Risk Score Calculation:** Combines the beta value from yFinance and the adjusted sentiment score to compute the final risk score for the stock.

How to Use:
-----------
1. Replace the placeholders for Reddit API credentials and YouTube API key with your own credentials.
2. Run the script to calculate risk scores for individual stocks or an entire list from a CSV file.

"""
