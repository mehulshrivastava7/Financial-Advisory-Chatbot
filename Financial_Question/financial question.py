"""This code is by nitin alone"""
import os
import csv
from transformers import pipeline
from huggingface_hub import login

# Replace with your actual Hugging Face API key
HUGGINGFACE_TOKEN = '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'

# Global variable for the classifier
classifier = None

def login_huggingface():
    """Login to Hugging Face using your API key."""
    login(token=HUGGINGFACE_TOKEN)

def init_classifier():
    """Initialize the zero-shot classification pipeline."""
    device = -1  # Use -1 for CPU or 0 for GPU
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )

def initialize():
    """Initialize the classifier if not already initialized."""
    global classifier
    if classifier is None:
        login_huggingface()
        classifier = init_classifier()

def classify_input(user_input, candidate_labels):
    """Classify the user input using zero-shot classification."""
    initialize()
    classification = classifier(
        sequences=user_input,
        candidate_labels=candidate_labels,
        multi_label=False
    )
    top_label = classification['labels'][0]
    score = classification['scores'][0]
    return top_label, score

def save_answer(answer, csv_file='answers.csv'):
    """Save the answer to a CSV file."""
    fieldnames = ["question", "answer", "confidence"]
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(answer)

# Define the list of questions
questions = [
    {
        "text": "What is your marital status?",
        "key": "marital_status",
        "candidate_labels": ["Single", "Common law", "Married", "Separated", "Divorced", "Widowed"],
        "numerical": False
    },
    {
        "text": "Are you currently employed?",
        "key": "employment_status",
        "candidate_labels": ["Yes", "No"],
        "numerical": False
    },
    {
        "text": "What are your primary financial goals?",
        "key": "financial_goals",
        "candidate_labels": ["Retirement", "Home purchase", "Education", "Emergency fund", "Wealth accumulation"],
        "numerical": False
    },
    {
        "text": "How would you describe your current life stage?",
        "key": "life_stage",
        "candidate_labels": ["Starting out", "Career building", "Peak earning years", "Pre-retirement", "Retirement"],
        "numerical": False
    },
    {
        "text": "What is your home ownership status?",
        "key": "home_status",
        "candidate_labels": ["I don't own a home", "I'm paying a mortgage", "My mortgage is paid off"],
        "numerical": False
    },
    {
        "text": "What is your investment experience?",
        "key": "investment_experience",
        "candidate_labels": [
            "Mostly Cash Savings and GICs",
            "Bonds, Income funds, GICs",
            "Mutual Funds and ETFs",
            "Self-Directed Investor: Stocks, Equities, Cryptocurrencies"
        ],
        "numerical": False
    },
    {
        "text": "What would you do if your investment lost 20 percent in a year?",
        "key": "market_reaction",
        "candidate_labels": ["Sell all investments", "Sell some", "Hold steady", "Buy more", "Buy a lot more"],
        "numerical": False
    },
    {
        "text": "What level of volatility would you be most comfortable with?",
        "key": "volatility_preference",
        "candidate_labels": ["Low Volatility", "Balanced", "High Volatility"],
        "numerical": False
    },
    {
        "text": "How long do you plan to hold your investments?",
        "key": "investment_horizon",
        "candidate_labels": ["0-3 years", "3-5 years", "5+ years"],
        "numerical": False
    },
    {
        "text": "What's your risk capacity (ability to take risks)?",
        "key": "risk_capacity",
        "candidate_labels": ["Very low", "Low", "Medium", "High", "Very high"],
        "numerical": False
    },
    {
        "text": "How old are you?",
        "key": "age",
        "candidate_labels": None,
        "numerical": True
    },
    {
        "text": "How many dependents do you have?",
        "key": "dependents",
        "candidate_labels": None,
        "numerical": True
    },
    {
        "text": "What is your monthly income?",
        "key": "monthly_income",
        "candidate_labels": [
            "0-25,000",
            "25,000-50,000",
            "50,000-75,000",
            "75,000-1,00,000",
            "1,00,000-1,25,000",
            "1,25,000-1,50,000",
            "1,50,000-1,75,000",
            "1,75,000-2,00,000"
        ],
        "numerical": False
    },
    {
        "text": "How much liability do you have?",
        "key": "liability",
        "candidate_labels": [
            "0-25,000",
            "25,000-50,000",
            "50,000-75,000",
            "75,000-1,00,000",
            "1,00,000-1,25,000",
            "1,25,000-1,50,000",
            "1,50,000-1,75,000",
            "1,75,000-2,00,000"
        ],
        "numerical": False
    },
    {
        "text": "What is the estimated value of all your assets?",
        "key": "total_assets",
        "candidate_labels": [
            "0-10,00,000",
            "10,00,000-20,00,000",
            "20,00,000-30,00,000",
            "30,00,000-40,00,000",
            "40,00,000-50,00,000",
            "50,00,000-70,00,000",
            "70,00,000-90,00,000"
        ],
        "numerical": False
    },
    {
        "text": "What is the estimated value of your fixed assets?",
        "key": "fixed_assets",
        "candidate_labels": [
            "0-5,00,000",
            "5,00,000-10,00,000",
            "10,00,000-15,00,000",
            "15,00,000-20,00,000",
            "20,00,000-30,00,000",
            "30,00,000-40,00,000",
            "40,00,000-50,00,000"
        ],
        "numerical": False
    },
    {
        "text": "What percentage of the investment do you expect as monthly return?",
        "key": "return_expectation",
        "candidate_labels": ["0-2", "2-4", "4-6", "6-8", "8-10"],
        "numerical": False
    },
]

# Helper functions
def get_answer(question):
    """Retrieve the answer and confidence from the CSV file."""
    with open('answers.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['question'] == question:
                return row['answer'], float(row['confidence'])
    return None, None

def get_question_info(key):
    """Return question info based on the key."""
    for question in questions:
        if question['key'] == key:
            return question
    return None

def get_score(question, scores_dict, max_score=1):
    """Calculate the score for a given question."""
    answer, confidence = get_answer(question)
    if answer and confidence is not None:
        answer_score = scores_dict.get(answer, 0)
        remaining_scores = [score for key, score in scores_dict.items() if key != answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        total_score = ((answer_score * confidence) + ((1 - confidence) * average_remaining_score)) / max_score
        return round(total_score, 2)
    else:
        return 0.0

# Score calculation functions
def get_score_from_value(value, scores_dict, max_score=1):
    """Calculate the score for a given input value."""
    answer_score = scores_dict.get(value, 0)
    return round(answer_score / max_score, 2)

# Rewritten score calculation functions
def calculate_marital_status_score(value):
    """Calculate score for marital status."""
    scores = {
        "Single": 8,
        "Common law": 6,
        "Married": 4,
        "Separated": 3,
        "Divorced": 2,
        "Widowed": 1
    }
    return get_score_from_value(value, scores, max_score=8)

def calculate_dependents_score(value):
    """Calculate score for number of dependents."""
    try:
        dependents = int(value)
        score = max(0, 20 - dependents * 2) / 20
        return round(score, 2)
    except ValueError:
        return 0.0

def calculate_employment_score(value):
    """Calculate score for employment status."""
    scores = {"Yes": 1, "No": -1}
    return get_score_from_value(value, scores)

def calculate_income_score(value):
    """Calculate score for monthly income."""
    scores = {
        "0-25,000": 1,
        "25,000-50,000": 2,
        "50,000-75,000": 3,
        "75,000-1,00,000": 4,
        "1,00,000-1,25,000": 5,
        "1,25,000-1,50,000": 6,
        "1,50,000-1,75,000": 7,
        "1,75,000-2,00,000": 8
    }
    return get_score_from_value(value, scores, max_score=8)

def calculate_home_score(value):
    """Calculate score for home ownership status."""
    scores = {
        "I don't own a home": 0,
        "I'm paying a mortgage": 4,
        "My mortgage is paid off": 8
    }
    return get_score_from_value(value, scores, max_score=8)

def calculate_investment_score(value):
    """Calculate score for investment experience."""
    scores = {
        "Mostly Cash Savings and GICs": 0,
        "Bonds, Income funds, GICs": 3,
        "Mutual Funds and ETFs": 6,
        "Self-Directed Investor: Stocks, Equities, Cryptocurrencies": 10
    }
    return get_score_from_value(value, scores, max_score=10)

def calculate_market_reaction_score(value):
    """Calculate score for market reaction."""
    scores = {
        "Sell all investments": 0,
        "Sell some": 3,
        "Hold steady": 6,
        "Buy more": 8,
        "Buy a lot more": 10
    }
    return get_score_from_value(value, scores, max_score=10)

def calculate_volatility_score(value):
    """Calculate score for volatility preference."""
    scores = {
        "Low Volatility": 0,
        "Balanced": 5,
        "High Volatility": 10
    }
    return get_score_from_value(value, scores, max_score=10)

def calculate_investment_horizon_score(value):
    """Calculate score for investment horizon."""
    scores = {
        "0-3 years": 0,
        "3-5 years": 5,
        "5+ years": 10
    }
    return get_score_from_value(value, scores, max_score=10)

def calculate_risk_capacity_score(value):
    """Calculate score for risk capacity."""
    scores = {
        "Very low": 0,
        "Low": 3,
        "Medium": 6,
        "High": 8,
        "Very high": 10
    }
    return get_score_from_value(value, scores, max_score=10)

def calculate_financial_goals_score(value):
    """Calculate score for financial goals."""
    scores = {
        "Retirement": 1,
        "Home purchase": 3,
        "Education": 4,
        "Emergency fund": 5,
        "Wealth accumulation": 2
    }
    return get_score_from_value(value, scores, max_score=5)

def calculate_life_stage_score(value):
    """Calculate score for current life stage."""
    scores = {
        "Starting out": 5,
        "Career building": 4,
        "Peak earning years": 3,
        "Pre-retirement": 2,
        "Retirement": 1
    }
    return get_score_from_value(value, scores, max_score=5)

def calculate_total_assets_score(value):
    """Calculate score for total assets."""
    scores = {
        "0-10,00,000": 1,
        "10,00,000-20,00,000": 2,
        "20,00,000-30,00,000": 3,
        "30,00,000-40,00,000": 4,
        "40,00,000-50,00,000": 5,
        "50,00,000-70,00,000": 6,
        "70,00,000-90,00,000": 7
    }
    return get_score_from_value(value, scores, max_score=7)

def calculate_fixed_assets_score(value):
    """Calculate score for fixed assets."""
    scores = {
        "0-5,00,000": 1,
        "5,00,000-10,00,000": 2,
        "10,00,000-15,00,000": 3,
        "15,00,000-20,00,000": 4,
        "20,00,000-30,00,000": 5,
        "30,00,000-40,00,000": 6,
        "40,00,000-50,00,000": 7
    }
    return get_score_from_value(value, scores, max_score=7)

def calculate_return_expectation_score(value):
    """Calculate score for return expectation."""
    scores = {
        "0-2": 1,
        "2-4": 2,
        "4-6": 3,
        "6-8": 4,
        "8-10": 5
    }
    return get_score_from_value(value, scores, max_score=5)

def calculate_liability_score(value):
    """Calculate score for liabilities."""
    scores = {
        "0-25,000": 8,
        "25,000-50,000": 7,
        "50,000-75,000": 6,
        "75,000-1,00,000": 5,
        "1,00,000-1,25,000": 4,
        "1,25,000-1,50,000": 3,
        "1,50,000-1,75,000": 2,
        "1,75,000-2,00,000": 1
    }
    return get_score_from_value(value, scores, max_score=8)

def calculate_age_score(value):
    """Calculate score based on age."""
    try:
        age = int(value)
        score = max(0, min(10, (65 - age) / 4)) / 10
        return round(score, 2)
    except ValueError:
        return 0.0


# Optional: Main function for testing
def main():
    # Example usage
    # Remove existing answers.csv file
    if os.path.exists('answers.csv'):
        os.remove('answers.csv')

    # Simulate answers for testing
    test_answers = [
        {"question": "What is your marital status?", "answer": "Married", "confidence": 0.9},
        {"question": "How many dependents do you have?", "answer": "2", "confidence": 1.0},
        {"question": "Are you currently employed?", "answer": "Yes", "confidence": 1.0},
        {"question": "What is your monthly income?", "answer": "50,000-75,000", "confidence": 0.8},
        {"question": "What is your home ownership status?", "answer": "I'm paying a mortgage", "confidence": 0.85},
        {"question": "What is your investment experience?", "answer": "Mutual Funds and ETFs", "confidence": 0.95},
        {"question": "What would you do if your investment lost 20 percent in a year?", "answer": "Hold steady", "confidence": 0.9},
        {"question": "What level of volatility would you be most comfortable with?", "answer": "Balanced", "confidence": 0.8},
        {"question": "How long do you plan to hold your investments?", "answer": "5+ years", "confidence": 0.95},
        {"question": "What's your risk capacity (ability to take risks)?", "answer": "Medium", "confidence": 0.9},
        {"question": "What are your primary financial goals?", "answer": "Retirement", "confidence": 0.85},
        {"question": "How would you describe your current life stage?", "answer": "Career building", "confidence": 0.9},
        {"question": "What is the estimated value of all your assets?", "answer": "30,00,000-40,00,000", "confidence": 0.8},
        {"question": "What is the estimated value of your fixed assets?", "answer": "15,00,000-20,00,000", "confidence": 0.85},
        {"question": "What percentage of the investment do you expect as monthly return?", "answer": "4-6", "confidence": 0.9},
        {"question": "How much liability do you have?", "answer": "25,000-50,000", "confidence": 0.8},
        {"question": "How old are you?", "answer": "35", "confidence": 1.0}
    ]

    for ans in test_answers:
        save_answer(ans)

    # Calculate and display scores
    scores = {
        "Marital Status Score": calculate_marital_status_score(),
        "Dependents Score": calculate_dependents_score(),
        "Employment Score": calculate_employment_score(),
        "Income Score": calculate_income_score(),
        "Home Ownership Score": calculate_home_score(),
        "Investment Experience Score": calculate_investment_score(),
        "Market Reaction Score": calculate_market_reaction_score(),
        "Volatility Preference Score": calculate_volatility_score(),
        "Investment Horizon Score": calculate_investment_horizon_score(),
        "Risk Capacity Score": calculate_risk_capacity_score(),
        "Financial Goals Score": calculate_financial_goals_score(),
        "Life Stage Score": calculate_life_stage_score(),
        "Total Assets Score": calculate_total_assets_score(),
        "Fixed Assets Score": calculate_fixed_assets_score(),
        "Return Expectation Score": calculate_return_expectation_score(),
        "Liability Score": calculate_liability_score(),
        "Age Score": calculate_age_score()
    }

    print("\nYour Scores:")
    for key, value in scores.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
