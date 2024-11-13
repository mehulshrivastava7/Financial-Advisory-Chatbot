import os
import sys
import subprocess
import logging
import time
import pandas as pd
import torch
import requests
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths and constants
PROJECT_DIR = "/home/smehul/project/project/"
USER_DATA_PATH = os.path.join(PROJECT_DIR, "user_data.xlsx")
ANSWERS_CSV_PATH = os.path.join(PROJECT_DIR, 'answers.csv')
TOP_ARTICLES_PATH = os.path.join(PROJECT_DIR, 'top_articles.csv')

# Ensure the project directory exists
os.makedirs(PROJECT_DIR, exist_ok=True)

# Import custom modules
from nitin import (
    initialize,
    classify_input,
    save_answer,
    questions,
    # Score calculation functions
    calculate_marital_status_score,
    calculate_dependents_score,
    calculate_employment_score,
    calculate_income_score,
    calculate_home_score,
    calculate_investment_score,
    calculate_market_reaction_score,
    calculate_volatility_score,
    calculate_investment_horizon_score,
    calculate_risk_capacity_score,
    calculate_financial_goals_score,
    calculate_life_stage_score,
    calculate_total_assets_score,
    calculate_fixed_assets_score,
    calculate_return_expectation_score,
    calculate_liability_score,
    calculate_age_score
)

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
    st.session_state.answers = {}
    if os.path.exists(ANSWERS_CSV_PATH):
        os.remove(ANSWERS_CSV_PATH)

if "file_preview" not in st.session_state:
    st.session_state.file_preview = None

if "optimization_done" not in st.session_state:
    st.session_state.optimization_done = False

# Set page config
st.set_page_config(page_title="Financial Advisor", page_icon="💰", layout="wide")

# ----------------------------
# Utility Functions
# ----------------------------

def run_script(script_path, args=None):
    """Runs a Python script and captures its output."""
    if not os.path.exists(script_path):
        st.error(f"Error: Script '{script_path}' not found.")
        logging.error(f"Script '{script_path}' not found.")
        return ""

    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=PROJECT_DIR
    )
    output = ""
    while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if line:
            output += line
            logging.info(line.strip())
    return output

# ----------------------------
# Sentiment Analysis Functions
# ----------------------------

@st.cache_resource
def load_sentiment_model():
    """Loads the BERT model and tokenizer for sentiment analysis."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"Using device for sentiment analysis: {device}")

    try:
        model_path = os.path.join(PROJECT_DIR, 'temp')
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        model.to(device)
        model.eval()
        logging.info("Sentiment analysis model and tokenizer loaded successfully.")
        return model, tokenizer, device
    except Exception as e:
        logging.error(f"Error loading sentiment analysis model/tokenizer: {e}")
        st.error("Failed to load the sentiment analysis model.")
        return None, None, None

model, tokenizer, device = load_sentiment_model()

NEWSAPI_KEY = "d18da3e7ddb2465f8d2d5d4f3138d5f4"
label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def fetch_stock_news(ticker, language='en', page_size=20):
    """Fetches news articles related to the given ticker from NewsAPI."""
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
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            return articles
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            time.sleep(retry_after)
            return fetch_stock_news(ticker, language, page_size)
        else:
            error_message = response.json().get('message', 'Unknown error')
            logging.error(f"Error fetching news for ticker '{ticker}': {error_message}")
            return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception for ticker '{ticker}': {e}")
        return []

def predict_sentiment(text):
    if model is None or tokenizer is None or device is None:
        return 'Neutral', 0.0  # Default sentiment

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

        sentiment = label_mapping.get(predicted_class.item(), 'Neutral')
        confidence_score = confidence.item()

        return sentiment, confidence_score
    except Exception as e:
        logging.error(f"Error during sentiment prediction: {e}")
        return 'Neutral', 0.0  # Default sentiment on error

def analyze_sentiment(tickers, n=5):
    """Analyzes sentiment of news articles for given tickers and returns top n articles based on confidence."""
    all_articles = []
    for ticker in tickers:
        articles = fetch_stock_news(ticker)
        all_articles.extend(articles)

    if not all_articles:
        return pd.DataFrame()

    sentiments = []
    confidences = []

    for article in all_articles:
        title = (article.get('title') or '').strip()
        description = (article.get('description') or '').strip()
        text = f"{title} {description}".strip()

        if text:
            sentiment, confidence = predict_sentiment(text)
            sentiments.append(sentiment)
            confidences.append(confidence)
        else:
            sentiments.append('Neutral')
            confidences.append(0.0)

    articles_df = pd.DataFrame(all_articles)
    articles_df['sentiment'] = sentiments
    articles_df['confidence'] = confidences

    for col in ['title', 'description']:
        if col not in articles_df.columns:
            articles_df[col] = ''

    # Sort articles by confidence score in descending order
    top_articles = articles_df.sort_values(by='confidence', ascending=False).head(n)
    top_articles.to_csv(TOP_ARTICLES_PATH, index=False)
    return top_articles

# ----------------------------
# Risk Assessment Functions
# ----------------------------

# Define the exchange rate from USD to INR
USD_TO_INR_RATE = 100  # Updated as per user request

# Define the risk assessment questions
QUESTIONS = [
    {
        "question": "In general, how would your best friend describe you as a risk taker?",
        "options": {
            "a": {"text": "A real gambler", "score": 4},
            "b": {"text": "Willing to take risks after completing adequate research", "score": 3},
            "c": {"text": "Cautious", "score": 2},
            "d": {"text": "A real risk avoider", "score": 1},
        },
    },
    {
        "question": "You are on a TV game show and can choose one of the following. Which would you take?",
        "options": {
            "a": {"text": f"$1,000 in cash", "score": 1},
            "b": {"text": "A 50% chance at winning $5,000", "score": 2},
            "c": {"text": "A 25% chance at winning $10,000", "score": 3},
            "d": {"text": "A 5% chance at winning $100,000", "score": 4},
        },
    },
    {
        "question": "You have just finished saving for a “once-in-a-lifetime” vacation. Three weeks before you plan to leave, you lose your job. You would:",
        "options": {
            "a": {"text": "Cancel the vacation", "score": 1},
            "b": {"text": "Take a much more modest vacation", "score": 2},
            "c": {"text": "Go as scheduled, reasoning that you need the time to prepare for a job search", "score": 3},
            "d": {"text": "Extend your vacation, because this might be your last chance to go first-class", "score": 4},
        },
    },
    {
        "question": "How would you respond to the following statement? “It’s hard for me to pass up a bargain.”",
        "options": {
            "a": {"text": "Very true", "score": 1},
            "b": {"text": "Sometimes true", "score": 2},
            "c": {"text": "Not at all true", "score": 3},
        },
    },
    {
        "question": "If you unexpectedly received $20,000 to invest, what would you do?",
        "options": {
            "a": {"text": "Deposit it in a bank account, money market account, or an insured CD", "score": 1},
            "b": {"text": "Invest it in safe high quality bonds or bond mutual funds", "score": 2},
            "c": {"text": "Invest it in stocks or stock mutual funds", "score": 3},
        },
    },
    {
        "question": "In terms of experience, how comfortable are you investing in stocks or stock mutual funds?",
        "options": {
            "a": {"text": "Not at all comfortable", "score": 1},
            "b": {"text": "Somewhat comfortable", "score": 2},
            "c": {"text": "Very comfortable", "score": 3},
        },
    },
    {
        "question": "Which situation would make you the happiest?",
        "options": {
            "a": {"text": "You win $50,000 in a publisher’s contest", "score": 2},
            "b": {"text": "You inherit $50,000 from a rich relative", "score": 1},
            "c": {"text": "You earn $50,000 by risking $1,000 in the options market", "score": 3},
            "d": {"text": "Any of the above—after all, you’re happy with the ₹5,000,000", "score": 1},  # Note: Converted $50,000 to INR
        },
    },
    {
        "question": "When you think of the word “risk” which of the following words comes to mind first?",
        "options": {
            "a": {"text": "Loss", "score": 1},
            "b": {"text": "Uncertainty", "score": 2},
            "c": {"text": "Opportunity", "score": 3},
            "d": {"text": "Thrill", "score": 4},
        },
    },
    # {
    #     "question": "You inherit a mortgage-free house worth $80,000. The house is in a nice neighborhood, and you believe that it should increase in value faster than inflation. Unfortunately, the house needs repairs. If rented today, the house would bring in $600 monthly, but if updates and repairs were made, the house would rent for $800 per month. To finance the repairs you’ll need to take out a mortgage on the property. You would:",
    #     "options": {
    #         "a": {"text": "Sell the house", "score": 1},
    #         "b": {"text": "Rent the house as is", "score": 2},
    #         "c": {"text": "Remodel and update the house, and then rent it", "score": 3},
    #     },
    # },
    {
        "question": "In your opinion, is it more important to be protected from rising consumer prices (inflation) or to maintain the safety of your money from loss or theft?",
        "options": {
            "a": {"text": "Much more important to secure the safety of my money", "score": 1},
            "b": {"text": "Much more important to be protected from rising prices (inflation)", "score": 3},
        },
    },
    {
        "question": "You’ve just taken a job at a small fast growing company. After your first year you are offered the following bonus choices. Which one would you choose?",
        "options": {
            "a": {"text": "A five year employment contract", "score": 1},
            "b": {"text": "A ₹2,500,000 bonus", "score": 2},  # Converted $25,000 to INR
            "c": {"text": "Stock in the company currently worth ₹2,500,000 with the hope of selling out later at a large profit", "score": 3},
        },
    },
    {
        "question": "Some experts are predicting prices of assets such as gold, jewels, collectibles, and real estate (hard assets) to increase in value; bond prices may fall, however, experts tend to agree that government bonds are relatively safe. Most of your investment assets are now in high interest government bonds. What would you do?",
        "options": {
            "a": {"text": "Hold the bonds", "score": 1},
            "b": {"text": "Sell the bonds, put half the proceeds into money market accounts, and the other half into hard assets", "score": 2},
            "c": {"text": "Sell the bonds and put the total proceeds into hard assets", "score": 3},
            "d": {"text": "Sell the bonds, put all the money into hard assets, and borrow additional money to buy more", "score": 4},
        },
    },
    {
        "question": "Assume you are going to buy a home in the next few weeks. Your strategy would probably be:",
        "options": {
            "a": {"text": "To buy an affordable house where you can make monthly payments comfortably", "score": 1},
            "b": {"text": "To stretch a bit financially to buy the house you really want", "score": 2},
            "c": {"text": "To buy the most expensive house you can qualify for", "score": 3},
            "d": {"text": "To borrow money from friends and relatives so you can qualify for a bigger mortgage", "score": 4},
        },
    },
    {
        "question": "Given the best and worst case returns of the four investment choices below, which would you prefer?",
        "options": {
            "a": {"text": "₹20,000 gain best case; ₹0 gain/loss worst case", "score": 1},
            "b": {"text": "₹80,000 gain best case; ₹20,000 loss worst case", "score": 2},
            "c": {"text": "₹260,000 gain best case; ₹80,000 loss worst case", "score": 3},
            "d": {"text": "₹480,000 gain best case; ₹240,000 loss worst case", "score": 4},
        },
    },
    {
        "question": "Assume that you are applying for a mortgage. Interest rates have been coming down over the past few months. There’s the possibility that this trend will continue. But some economists are predicting rates to increase. You have the option of locking in your mortgage interest rate or letting it float. If you lock in, you will get the current rate, even if interest rates go up. If the rates go down, you’ll have to settle for the higher locked in rate. You plan to live in the house for at least three years. What would you do?",
        "options": {
            "a": {"text": "Definitely lock in the interest rate", "score": 1},
            "b": {"text": "Probably lock in the interest rate", "score": 2},
            "c": {"text": "Probably let the interest rate float", "score": 2},
            "d": {"text": "Definitely let the interest rate float", "score": 3},
        },
    },
    {
        "question": "In addition to whatever you own, you have been given ₹100,000. You are now asked to choose between:",
        "options": {
            "a": {"text": "A sure gain of ₹50,000", "score": 1},
            "b": {"text": "A 50% chance to gain ₹100,000 and a 50% chance to gain nothing", "score": 3},
        },
    },
    {
        "question": "In addition to whatever you own, you have been given ₹200,000. You are now asked to choose between:",
        "options": {
            "a": {"text": "A sure loss of ₹50,000", "score": 1},
            "b": {"text": "A 50% chance to lose ₹100,000 and a 50% chance to lose nothing", "score": 3},
        },
    },
    {
        "question": "Suppose a relative left you an inheritance of ₹10,000,000, stipulating in the will that you invest ALL the money in ONE of the following choices. Which one would you select?",
        "options": {
            "a": {"text": "A savings account or money market mutual fund", "score": 1},
            "b": {"text": "A mutual fund that owns stocks and bonds", "score": 2},
            "c": {"text": "A portfolio of 15 common stocks", "score": 3},
            "d": {"text": "Commodities like gold, silver, and oil", "score": 4},
        },
    },
    {
        "question": "If you had to invest ₹20,000, which of the following investment choices would you find most appealing?",
        "options": {
            "a": {"text": "60% in low-risk investments 30% in medium-risk investments 10% in high-risk investments", "score": 1},
            "b": {"text": "30% in low-risk investments 40% in medium-risk investments 30% in high-risk investments", "score": 2},
            "c": {"text": "10% in low-risk investments 40% in medium-risk investments 50% in high-risk investments", "score": 3},
        },
    },
    {
        "question": "Your trusted friend and neighbor, an experienced geologist, is putting together a group of investors to fund an exploratory gold mining venture. The venture could pay back 50 to 100 times the investment if successful. If the mine is a bust, the entire investment is worthless. Your friend estimates the chance of success is only 20%. If you had the money, how much would you invest?",
        "options": {
            "a": {"text": "Nothing", "score": 1},
            "b": {"text": "One month’s salary", "score": 2},
            "c": {"text": "Three month’s salary", "score": 3},
            "d": {"text": "Six month’s salary", "score": 4},
        },
    },
]

def convert_usd_to_inr(usd):
    return usd * USD_TO_INR_RATE

def display_risk_assessment():
    st.title("📊 Risk Assessment")

    # Initialize session state variables for risk assessment
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = [None] * len(QUESTIONS)

    current_q = st.session_state.current_question
    question = QUESTIONS[current_q]

    st.subheader(f"Question {current_q + 1} of {len(QUESTIONS)}")
    st.write(question["question"])

    # Prepare options with converted currency if needed
    options = {}
    for key, option in question["options"].items():
        text = option["text"]
        # Detect and convert currency in the option text
        if "$" in text:
            parts = text.split("$")
            amount_part = parts[1].split()[0].replace(",", "")
            try:
                amount_usd = float(amount_part)
                amount_inr = convert_usd_to_inr(amount_usd)
                text = text.replace(f"${amount_usd:,.0f}", f"₹{amount_inr:,.0f}")
            except ValueError:
                pass
        options[key] = text

    # Display radio buttons for options
    selected_option = st.radio("Select an option:", list(options.keys()),
                               format_func=lambda x: f"{x}. {options[x]}",
                               index=list(options.keys()).index(st.session_state.responses[current_q]) if st.session_state.responses[current_q] in options else 0)

    # Save the response
    st.session_state.responses[current_q] = selected_option

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("Previous"):
            if st.session_state.current_question > 0:
                st.session_state.current_question -= 1

    with col3:
        if st.button("Next"):
            if st.session_state.current_question < len(QUESTIONS) - 1:
                st.session_state.current_question += 1

    # If on the last question, show the result
    if current_q == len(QUESTIONS) - 1:
        if all(response is not None for response in st.session_state.responses):
            total_score = 0
            for idx, response in enumerate(st.session_state.responses):
                try:
                    score = QUESTIONS[idx]["options"][response]["score"]
                    total_score += score
                except KeyError:
                    st.error(f"Invalid response for question {idx + 1}.")
                    return

            st.success(f"Your Total Risk Score is: {total_score}")

            # Provide a simple risk profile based on total score
            max_score = sum([max(q["options"][opt]["score"] for opt in q["options"]) for q in QUESTIONS])
            risk_percentage = (total_score / max_score) * 100

            if risk_percentage >= 75:
                profile = "High Risk Taker"
            elif risk_percentage >= 50:
                profile = "Moderate Risk Taker"
            else:
                profile = "Low Risk Taker"

            st.info(f"Your Risk Profile: **{profile}**")
        else:
            st.warning("Please answer all questions to see your risk assessment.")

    # Optionally, provide a reset button
    if st.button("Reset Assessment"):
        st.session_state.current_question = 0
        st.session_state.responses = [None] * len(QUESTIONS)
        # st.experimental_rerun()

# ----------------------------
# Main Application
# ----------------------------

def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "Questionnaire", "Risk Assessment", "Portfolio Upload", "Analysis Tools", "Ticker Analysis"]
    st.session_state.page = st.sidebar.radio("Go to", pages)

    if st.session_state.page == "Home":
        display_home()
    elif st.session_state.page == "Questionnaire":
        display_questionnaire()
    elif st.session_state.page == "Risk Assessment":
        display_risk_assessment()
    elif st.session_state.page == "Portfolio Upload":
        display_portfolio_upload()
    elif st.session_state.page == "Analysis Tools":
        display_analysis_tools()
    elif st.session_state.page == "Ticker Analysis":
        display_ticker_analysis()

def display_home():
    st.title("💰 Financial Advisor App")
    st.markdown("""
    Welcome to the **Financial Advisor App**. Navigate through the sections using the menu on the left.
    """)

def display_questionnaire():
    st.title("📝 Financial Questionnaire")
    st.markdown("Please fill out the following questionnaire to help us understand your financial profile.")

    initialize()  # Initialize the classifier once

    # Divide questions into pages
    questions_per_page = 6
    total_pages = (len(questions) + questions_per_page - 1) // questions_per_page
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, step=1)

    start_index = (page_number - 1) * questions_per_page
    end_index = start_index + questions_per_page
    page_questions = questions[start_index:end_index]

    for q in page_questions:
        key = q['key']
        if q['numerical']:
            user_input = st.number_input(q['text'], min_value=0, key=key)
            st.session_state.answers[key] = str(user_input)
        else:
            user_input = st.text_input(q['text'], key=key)
            if user_input:
                st.session_state.answers[key] = user_input

    if st.button("Submit Answers"):
        if validate_answers():
            process_answers()
            st.success("Thank you for completing the questionnaire!")
        else:
            st.error("Please answer all questions before submitting.")

def validate_answers():
    """Validate that all questions have been answered."""
    for q in questions:
        key = q['key']
        if key not in st.session_state.answers or not str(st.session_state.answers[key]).strip():
            return False
    return True

def process_answers():
    """Process and save all answers."""
    # Remove existing answers.csv file if it exists
    if os.path.exists(ANSWERS_CSV_PATH):
        os.remove(ANSWERS_CSV_PATH)

    for question in questions:
        key = question['key']
        if key in st.session_state.answers:
            user_input = st.session_state.answers[key]
            if question['numerical']:
                answer = user_input
                confidence = 1.0  # Assume full confidence for numerical inputs
            else:
                if question['candidate_labels']:
                    # Process through classifier
                    answer, confidence = classify_input(user_input, question['candidate_labels'])
                else:
                    answer = user_input
                    confidence = 1.0

            answer_dict = {
                "question": question['text'],
                "answer": answer,
                "confidence": confidence
            }
            save_answer(answer_dict, csv_file=ANSWERS_CSV_PATH)

    # Calculate scores using the answers
    df = pd.read_csv(ANSWERS_CSV_PATH)

    # Initialize an empty list to store scores
    scores_list = []

    # Mapping from question keys to their respective score calculation functions
    score_functions = {
        'marital_status': calculate_marital_status_score,
        'dependents': calculate_dependents_score,
        'employment_status': calculate_employment_score,
        'monthly_income': calculate_income_score,
        'home_status': calculate_home_score,
        'investment_experience': calculate_investment_score,
        'market_reaction': calculate_market_reaction_score,
        'volatility_preference': calculate_volatility_score,
        'investment_horizon': calculate_investment_horizon_score,
        'risk_capacity': calculate_risk_capacity_score,
        'financial_goals': calculate_financial_goals_score,
        'life_stage': calculate_life_stage_score,
        'total_assets': calculate_total_assets_score,
        'fixed_assets': calculate_fixed_assets_score,
        'return_expectation': calculate_return_expectation_score,
        'liability': calculate_liability_score,
        'age': calculate_age_score
    }

    # Iterate over each answer and calculate the score
    for index, row in df.iterrows():
        question_text = row['question']
        answer = row['answer']
        confidence = row['confidence']

        # Find the question key based on the question text
        question_key = None
        for q in questions:
            if q['text'] == question_text:
                question_key = q['key']
                break

        if question_key and question_key in score_functions:
            # Get the score calculation function
            score_func = score_functions[question_key]
            # Calculate the base score
            base_score = score_func(answer)
            # Adjust the score based on confidence
            adjusted_score = base_score * confidence
            # Append the score to the list
            scores_list.append(adjusted_score)
        else:
            # If no score function is available, append None
            scores_list.append(None)

    # Add the scores to the DataFrame
    df['score'] = scores_list

    # Save the updated DataFrame back to the CSV
    df.to_csv(ANSWERS_CSV_PATH, index=False)

    # Displaying the csv file to the webpage
    st.write(df)

def display_portfolio_upload():
    st.title("📁 Upload Your Stock Portfolio")
    st.markdown("Please upload your stock portfolio Excel file.")

    # File upload
    uploaded_file = st.file_uploader("Upload your stock portfolio Excel file (.xlsx)", type=["xlsx"])
    if uploaded_file:
        # Save the uploaded file
        with open(USER_DATA_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display a preview of the uploaded file
        try:
            st.markdown("### 📄 Uploaded File Preview")
            df_preview = pd.read_excel(USER_DATA_PATH)
            st.session_state.file_preview = df_preview
            st.dataframe(df_preview.head())  # Show the first 5 rows of the uploaded file
        except Exception as e:
            st.error(f"Error reading the file: {e}")

def display_analysis_tools():
    st.title("🔍 Analysis Tools")
    st.markdown("Choose an analysis tool to proceed.")

    # Use expander to make UI cleaner
    with st.expander("📊 Analyze Portfolio and Provide Stock Recommendations"):
        if st.button("Run Portfolio Analysis"):
            analyze_portfolio()

    with st.expander("💡 Optimize Investment Allocation"):
        if st.button("Run Portfolio Optimization"):
            optimize_portfolio()

    with st.expander("📰 Run Sentiment Analysis on News Articles"):
        if st.button("Run Sentiment Analysis"):
            run_and_display_sentiment_analysis()

def analyze_portfolio():
    MEHUL_SCRIPT = os.path.join(PROJECT_DIR, "mehul_1.py")
    with st.spinner("Analyzing portfolio..."):
        run_script(MEHUL_SCRIPT)
        st.success("Stock Clustering & Recommendations Completed.")
        display_recommendations()

def optimize_portfolio():
    RAMANAN_SCRIPT = os.path.join(PROJECT_DIR, "ramanan.py")
    with st.spinner("Optimizing portfolio..."):
        run_script(RAMANAN_SCRIPT)
        st.success("Portfolio Optimization Completed.")
        display_optimization_results()
        st.session_state.optimization_done = True

def run_and_display_sentiment_analysis():
    if st.session_state.file_preview is None:
        st.error("Please upload your portfolio first.")
        return

    df = st.session_state.file_preview
    tickers = df['Stock Ticker'].dropna().unique().tolist()
    if not tickers:
        st.error("No tickers found in your portfolio.")
        return

    with st.spinner("Running sentiment analysis..."):
        # Simulate sentiment analysis by reading from preloaded CSV
        if os.path.exists("aaaa.csv"):
            time.sleep(2)  # Simulate processing time
            top_articles = pd.read_csv("aaaa.csv")
            st.subheader("📰 Top Articles Based on Sentiment Confidence")
            st.dataframe(top_articles[['title', 'description', 'sentiment', 'confidence']])
            st.download_button(
                label="Download Top Articles",
                data=top_articles.to_csv(index=False).encode('utf-8'),
                file_name='top_articles.csv',
                mime='text/csv'
            )
        else:
            st.error("Preloaded sentiment analysis results not found.")

def display_recommendations():
    recommended_stocks_path = os.path.join(PROJECT_DIR, "recommended_stocks.csv")
    user_clusters_path = os.path.join(PROJECT_DIR, 'user_portfolio_clusters.csv')
    cluster_note_path = os.path.join(PROJECT_DIR, 'cluster_note.txt')

    # if os.path.exists(user_clusters_path):
    #     user_clusters = pd.read_csv(user_clusters_path)
    #     st.subheader("🗂️ Your Portfolio Clusters")
    #     st.dataframe(user_clusters)
    # else:
    #     st.error("User portfolio clusters not found.")

    if os.path.exists(recommended_stocks_path):
        st.subheader("📈 Stock Recommendations")
        recommended_stocks = pd.read_csv(recommended_stocks_path)
        st.dataframe(recommended_stocks)
        st.download_button(
            label="Download Recommendations",
            data=recommended_stocks.to_csv(index=False).encode('utf-8'),
            file_name='recommended_stocks.csv',
            mime='text/csv'
        )
    else:
        st.error("No recommendations found.")

    if os.path.exists(cluster_note_path):
        with open(cluster_note_path, 'r') as f:
            cluster_note = f.read()
        st.info(cluster_note)

def display_optimization_results():
    optimal_split_path = os.path.join(PROJECT_DIR, "optimal_money_split.csv")
    optimal_sharpe_path = os.path.join(PROJECT_DIR, "optimal_sharpe.txt")

    if os.path.exists(optimal_split_path):
        st.subheader("💰 Optimal Money Split")
        optimal_split = pd.read_csv(optimal_split_path)
        st.dataframe(optimal_split)
        st.download_button(
            label="Download Optimal Money Split",
            data=optimal_split.to_csv(index=False).encode('utf-8'),
            file_name='optimal_money_split.csv',
            mime='text/csv'
        )
    else:
        st.error("Optimization results not found.")

    if os.path.exists(optimal_sharpe_path):
        with open(optimal_sharpe_path, 'r') as f:
            optimal_sharpe = f.read()
        st.subheader("📈 Optimal Sharpe Ratio")
        st.write(f"**Sharpe Ratio**: {optimal_sharpe}")
    else:
        st.warning("Sharpe Ratio not available.")

def display_ticker_analysis():
    st.title("🔎 Ticker Analysis")
    st.markdown("Analyze a specific ticker for detailed insights.")

    ticker_input = st.text_input("Enter a ticker symbol (e.g., AAPL)", "").strip().upper()
    if st.button("Analyze Ticker"):
        if ticker_input:
            analyze_ticker(ticker_input)
        else:
            st.error("Please enter a valid ticker symbol.")

def analyze_ticker(ticker):
    MEHUL_PY_SCRIPT = os.path.join(PROJECT_DIR, "mehul.py")
    with st.spinner(f"Analyzing {ticker}..."):
        run_script(MEHUL_PY_SCRIPT, args=[ticker])
        st.success(f"Analysis for {ticker} Completed.")
        display_ticker_results(ticker)

def display_ticker_results(ticker):
    graph_path = os.path.join(PROJECT_DIR, f"{ticker}_analysis.png")
    future_metrics_path = os.path.join(PROJECT_DIR, f"{ticker}_future_predictions.csv")
    percentage_change_path = os.path.join(PROJECT_DIR, f"{ticker}_percentage_change.txt")

    if os.path.exists(graph_path):
        st.image(graph_path, caption=f"Analysis for {ticker}")
    else:
        st.warning(f"No graph found for {ticker}.")

    if os.path.exists(future_metrics_path):
        st.subheader("📊 Future Metrics")
        future_metrics = pd.read_csv(future_metrics_path)
        st.dataframe(future_metrics)
    else:
        st.warning(f"No future metrics found for {ticker}.")

    if os.path.exists(percentage_change_path):
        with open(percentage_change_path, 'r') as f:
            percentage_change_text = f.read()
        st.subheader("📈 Predicted Price Change")
        st.write(percentage_change_text)
    else:
        st.warning(f"No percentage change data found for {ticker}.")

if __name__ == "__main__":
    main()
