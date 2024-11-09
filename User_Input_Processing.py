from transformers import pipeline
from huggingface_hub import login
import csv
import os
 
# Global model loading - done once during app startup
login(token='YOUR_HF_API_KEY')  # Replace with your actual Hugging Face API key
device = -1  # Use -1 for CPU
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)
 
# Define the questions and candidate labels
questions = [
    {
        "question": "What is your marital status?",
        "candidate_labels": ["Single", "Common law", "Married", "Separated", "Divorced", "Widowed"]
    },
    {
        "question": "What is your employment status?",
        "candidate_labels": ["Yes", "No"]
    },
    {
        "question": "What are your primary financial goals?",
        "candidate_labels": ["Retirement", "Home purchase", "Education", "Emergency fund", "Wealth accumulation"]
    }
    {
        "question": "How would you describe your current life stage?",
        "candidate_labels": ["Starting out", "Career building", "Peak earning years", "Pre-retirement", "Retirement"]
    }
    {
        "question": "What is your home ownership status?",
        "candidate_labels": ["I don't own a home", "I'm paying a mortgage", "My mortgage is paid off"]
    }
    {
        "question": "What is your investment experience?",
        "candidate_labels": ["Mostly Cash Savings and GICs", "Bonds, Income funds, GICs", "Mutual Funds and Exchange Traded Funds (ETFs)", "Self-Directed Investor: Stocks, Equities, Cryptocurrencies"]
    }
    {
        "question": "What would you do if your investment lost 20 percent in a year?",
        "candidate_labels": ["Sell all investments", "Sell some", "Hold steady", "Buy more", "Buy a lot more"]
    }
    {
        "question": "What level of volatility would you be the most comfortable with?",
        "candidate_labels": ["Low Volatility", "Balanced", "High Volatility"]
    }
    {
        "question": "How long do you plan to hold your investments?",
        "candidate_labels": ["0-3 years", "3-5 years", "5+ years"]
    }
    {
        "question": "What's your risk capacity (ability to take risks)?",
        "candidate_labels": ["Very low", "Low", "Medium", "High", "Very high"]
    }
    {
        "question": "How old are you?",
        "candidate_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98"]
    }
    {
        "question": "How many dependents do you have?",
        "candidate_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    }
    {
        "question": "What is your monthly income?",
        "candidate_labels": ["0-25,000", "25,000-50,000", "50,000-75,000", "75,000-1,00,000", "1,00,000-1,25,000", "1,25,000-1,50,000", "1,50,000-1,75,000", "1,75,000-2,00,000"]
    }
    {
        "question": "How much liability do you have?",
        "candidate_labels": ["0-25,000", "25,000-50,000", "50,000-75,000", "75,000-1,00,000", "1,00,000-1,25,000", "1,25,000-1,50,000", "1,50,000-1,75,000", "1,75,000-2,00,000"]
    }
    {
        "question": "What is the estimated value of all your assets?",
        "candidate_labels": ["0-10,00,000", "10,00,000-20,00,000", "50,00,000-70,00,000", "70,00,000-90,00,000", "20,00,000-30,00,000", "30,00,000-40,00,000", "40,00,000-50,00,000",]
    }
    {
        "question": "What is the estimated value of your fixed assets?",
        "candidate_labels": ["0-10,00,000", "10,00,000-20,00,000", "50,00,000-70,00,000", "70,00,000-90,00,000", "20,00,000-30,00,000", "30,00,000-40,00,000", "40,00,000-50,00,000",]
    }
    {
        "question": "What percentage of the investment do you expect as monthly return?",
        "candidate_labels": ["0-2", "2-4", "4-6", "6-8", "8-10"]
    }
]
 
# Function to classify user input
def classify_input(user_input, candidate_labels):
    classification = classifier(
        sequences=user_input,
        candidate_labels=candidate_labels,
        multi_label=False
    )
    top_label = classification['labels'][0]
    score = classification['scores'][0]
    return top_label, score
 
# Function to save answer to CSV
def save_answer_to_csv(answer_data, csv_file):
    fieldnames = ["question", "answer", "confidence"]
    file_exists = os.path.isfile(csv_file)
 
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(answer_data)
 
# Main function to process all questions
def process_questions(user_responses):
    folder_path = '/home/arnavbhatt/project'
    os.makedirs(folder_path, exist_ok=True)
    csv_file = os.path.join(folder_path, 'answers.csv')
 
    for idx, q in enumerate(questions):
        user_input = user_responses[idx]  # Get user response from the provided list
        label, confidence = classify_input(user_input, q["candidate_labels"])
        answer_data = {
            "question": q["question"],
            "answer": label,
            "confidence": confidence
        }
        save_answer_to_csv(answer_data, csv_file)
