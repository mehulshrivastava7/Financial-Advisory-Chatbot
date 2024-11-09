import csv
import os

def calculate_marital_status_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    marital_status_answer = None
    marital_status_confidence = None
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What is your marital status?":
                marital_status_answer = answer
                marital_status_confidence = confidence
                break

    # Dictionary to map marital status answers to their corresponding scores
    marital_status_scores = {
        "Single": 8,
        "Common law": 6,
        "Married": 4,
        "Separated": 3,
        "Divorced": 2,
        "Widowed": 1
    }

    # Compute the marital_status_score if an answer was found
    if marital_status_answer:
        # Get the score for the provided marital status
        answer_score = marital_status_scores.get(marital_status_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in marital_status_scores.items() if status != marital_status_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        marital_status_score = (answer_score * marital_status_confidence) + \
                               ((1 - marital_status_confidence) * average_remaining_score)

        return round(marital_status_score, 2)
    else:
        raise ValueError("The question 'What is your marital status?' was not found in the CSV.")
    
def calculate_dependents_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    dependents_answer = None
    dependents_confidence = None
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                dependents_confidence = float(confidence)
            except ValueError:
                dependents_confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "How many dependents do you have?"
            if question == "How many dependents do you have?":
                try:
                    # Convert the answer to an integer (number of dependents)
                    dependents_answer = int(answer)
                except ValueError:
                    dependents_answer = None  # Handle invalid number of dependents
                break

    # Compute the dependents_score if an answer was found
    if dependents_answer is not None and dependents_confidence is not None:
        # Apply the formula for dependents_score
        dependents_score = max(0, 20 - dependents_answer * 2 * dependents_confidence - (20 - dependents_answer) / 10 * 2 * (1 - dependents_confidence))/40
        return round(dependents_score, 2)
    else:
        raise ValueError("The question 'How many dependents do you have?' was not found or is invalid in the CSV.")

def calculate_employment_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    employment_answer = None
    employment_confidence = None

    employment_status_scores = {
        "Yes": 1,
        "No": -1
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What is your employment status?":
                employment_answer = answer
                employment_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if employment_answer is not None and employment_confidence is not None:
        # Apply the formula for dependents_score
        answer_score = employment_status_scores.get(employment_answer, 0)
        employment_score = answer_score * employment_confidence - answer_score * (1 - employment_confidence)
        return round(employment_score, 2)
    else:
        raise ValueError("The question 'What is your employment status?' was not found or is invalid in the CSV.")

def calculate_income_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    income_answer = None
    income_confidence = None

    income_status_scores = {
        "0-25,000": 1,
        "25,000-50,000": 2,
        "50,000-75,000": 3,
        "75,000-1,00,000": 4,
        "1,00,000-1,25,000": 5,
        "1,25,000-1,50,000": 6,
        "1,50,000-1,75,000": 7,
        "1,75,000-2,00,000": 8
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What is your monthly income?":
                income_status_answer = answer
                income_status_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if income_status_answer:
        # Get the score for the provided marital status
        answer_score = income_status_scores.get(income_status_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in income_status_scores.items() if status != income_status_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        marital_status_score = ((answer_score * income_status_confidence) + \
                               ((1 - income_status_confidence) * average_remaining_score)) / 8

        return round(marital_status_score, 2)
    else:
        raise ValueError("The question 'What is your monthly income?' was not found or is invalid in the CSV.")

def calculate_home_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    home_answer = None
    home_confidence = None

    home_status_scores = {
        "I don't own a home": 0, 
        "I'm paying a mortgage": 4, 
        "My mortgage is paid off": 8
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What is your home ownership status?":
                home_answer = answer
                home_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if home_answer:
        # Get the score for the provided marital status
        answer_score = home_status_scores.get(home_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in home_status_scores.items() if status != home_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        marital_status_score = ((answer_score * home_confidence) + \
                               ((1 - home_confidence) * average_remaining_score)) / 8

        return round(marital_status_score, 2)
    else:
        raise ValueError("The question 'What is your home ownership status?' was not found or is invalid in the CSV.")

def calculate_investment_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    investment_answer = None
    investment_confidence = None

    investment_status_scores = {
        "Mostly Cash Savings and GICs": 0,
        "Bonds, Income funds, GICs": 3,
        "Mutual Funds and Exchange Traded Funds (ETFs)": 6,
        "Self-Directed Investor: Stocks, Equities, Cryptocurrencies": 10
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What is your investment experience?":
                investment_answer = answer
                investment_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if investment_answer:
        # Get the score for the provided marital status
        answer_score = investment_status_scores.get(investment_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in investment_status_scores.items() if status != investment_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        investment_status_score = ((answer_score * investment_confidence) + \
                               ((1 - investment_confidence) * average_remaining_score)) / 10

        return round(investment_status_score, 2)
    else:
        raise ValueError("The question 'What is your investment experience?' was not found or is invalid in the CSV.")

def calculate_reaction_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    reaction_answer = None
    reaction_confidence = None

    reaction_status_scores = {
        "Sell all investments": 0, 
        "Sell some": 3, 
        "Hold steady": 6, 
        "Buy more": 8, 
        "Buy a lot more": 10
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What would you do if your investment lost 20 percent in a year?":
                reaction_answer = answer
                reaction_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if reaction_answer:
        # Get the score for the provided marital status
        answer_score = reaction_status_scores.get(reaction_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in reaction_status_scores.items() if status != reaction_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        reaction_status_score = ((answer_score * reaction_confidence) + \
                               ((1 - reaction_confidence) * average_remaining_score)) / 10

        return round(reaction_status_score, 2)
    else:
        raise ValueError("The question 'What would you do if your investment lost 20 percent in a year?' was not found or is invalid in the CSV.")

def calculate_volatility_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    volatility_answer = None
    volatility_confidence = None

    volatility_status_scores = {
        "Low Volatility": 0, 
        "Balanced": 5, 
        "High Volatility": 10
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What level of volatility would you be the most comfortable with?":
                volatility_answer = answer
                volatility_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if volatility_answer:
        # Get the score for the provided marital status
        answer_score = volatility_status_scores.get(volatility_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in volatility_status_scores.items() if status != volatility_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        volatility_status_score = ((answer_score * volatility_confidence) + \
                               ((1 - volatility_confidence) * average_remaining_score)) / 10

        return round(volatility_status_score, 2)
    else:
        raise ValueError("The question 'What level of volatility would you be the most comfortable with?' was not found or is invalid in the CSV.")

def calculate_horizon_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    horizon_answer = None
    horizon_confidence = None

    horizon_status_scores = {
        "0-3 years": 0, 
        "3-5 years": 5, 
        "5+ years": 10
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "How long do you plan to hold your investments?":
                horizon_answer = answer
                horizon_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if horizon_answer:
        # Get the score for the provided marital status
        answer_score = horizon_status_scores.get(horizon_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in horizon_status_scores.items() if status != horizon_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        horizon_status_score = ((answer_score * horizon_confidence) + \
                               ((1 - horizon_confidence) * average_remaining_score)) / 10

        return round(horizon_status_score, 2)
    else:
        raise ValueError("The question 'How long do you plan to hold your investments?' was not found or is invalid in the CSV.")

def calculate_capacity_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    capacity_answer = None
    capacity_confidence = None

    capacity_status_scores = {
        "Very low": 0, 
        "Low": 3, 
        "Medium": 6, 
        "High": 8, 
        "Very high": 10
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What's your risk capacity (ability to take risks)?":
                capacity_answer = answer
                capacity_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if capacity_answer:
        # Get the score for the provided marital status
        answer_score = capacity_status_scores.get(capacity_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in capacity_status_scores.items() if status != capacity_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        capacity_status_score = ((answer_score * capacity_confidence) + \
                               ((1 - capacity_confidence) * average_remaining_score)) / 10

        return round(capacity_status_score, 2)
    else:
        raise ValueError("The question 'What's your risk capacity (ability to take risks)?' was not found or is invalid in the CSV.")

def calculate_goal_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    goal_answer = None
    goal_confidence = None

    goal_status_scores = {
        "Retirement": 1, 
        "Home purchase": 3, 
        "Education": 4, 
        "Emergency fund": 5, 
        "Wealth accumulation": 2
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What are your primary financial goals?":
                goal_answer = answer
                goal_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if goal_answer:
        # Get the score for the provided marital status
        answer_score = goal_status_scores.get(goal_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in goal_status_scores.items() if status != goal_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        goal_status_score = ((answer_score * goal_confidence) + \
                               ((1 - goal_confidence) * average_remaining_score)) / 5

        return round(goal_status_score, 2)
    else:
        raise ValueError("The question 'What are your primary financial goals?' was not found or is invalid in the CSV.")

def calculate_career_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    career_answer = None
    career_confidence = None

    career_status_scores = {
        "Starting out": 5, 
        "Career building": 4, 
        "Peak earning years": 3, 
        "Pre-retirement": 2, 
        "Retirement": 1
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "How would you describe your current life stage?":
                career_answer = answer
                career_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if career_answer:
        # Get the score for the provided marital status
        answer_score = career_status_scores.get(career_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in career_status_scores.items() if status != career_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        career_status_score = ((answer_score * career_confidence) + \
                               ((1 - career_confidence) * average_remaining_score)) / 5

        return round(career_status_score, 2)
    else:
        raise ValueError("The question 'How would you describe your current life stage?' was not found or is invalid in the CSV.")

def calculate_property_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    property_answer = None
    property_confidence = None

    property_status_scores = {
        "0-10,00,000": 1,
        "10,00,000-20,00,000": 2,
        "50,00,000-70,00,000": 3,
        "70,00,000-90,00,000": 4,
        "20,00,000-30,00,000": 5,
        "30,00,000-40,00,000": 6,
        "40,00,000-50,00,000": 7,
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What is the estimated value of all your assets?":
                property_answer = answer
                property_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if property_answer:
        # Get the score for the provided marital status
        answer_score = property_status_scores.get(property_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in property_status_scores.items() if status != property_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        property_status_score = ((answer_score * property_confidence) + \
                               ((1 - property_confidence) * average_remaining_score)) / 7

        return round(property_status_score, 2)
    else:
        raise ValueError("The question 'What is the estimated value of all your assets' was not found or is invalid in the CSV.")

def calculate_fixed_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    fixed_answer = None
    fixed_confidence = None

    fixed_status_scores = {
        "0-10,00,000": 1,
        "10,00,000-20,00,000": 2,
        "50,00,000-70,00,000": 3,
        "70,00,000-90,00,000": 4,
        "20,00,000-30,00,000": 5,
        "30,00,000-40,00,000": 6,
        "40,00,000-50,00,000": 7,
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What is the estimated value of your fixed assets?":
                fixed_answer = answer
                fixed_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if fixed_answer:
        # Get the score for the provided marital status
        answer_score = fixed_status_scores.get(fixed_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in fixed_status_scores.items() if status != fixed_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        fixed_status_score = ((answer_score * fixed_confidence) + \
                               ((1 - fixed_confidence) * average_remaining_score)) / 7

        return round(fixed_status_score, 2)
    else:
        raise ValueError("The question 'What is the estimated value of your fixed assets?' was not found or is invalid in the CSV.")

def calculate_return_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    return_answer = None
    return_confidence = None

    return_status_scores = {
        "0-2": 1,
        "2-4": 2,
        "4-6": 3,
        "6-8": 4,
        "8-10": 5
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "What percentage of the investment do you expect as monthly return?":
                return_answer = answer
                return_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if return_answer:
        # Get the score for the provided marital status
        answer_score = return_status_scores.get(return_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in return_status_scores.items() if status != return_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        return_status_score = ((answer_score * return_confidence) + \
                               ((1 - return_confidence) * average_remaining_score)) / 5

        return round(return_status_score, 2)
    else:
        raise ValueError("The question 'What percentage of the investment do you expect as monthly return?' was not found or is invalid in the CSV.")

def calculate_liability_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    liability_answer = None
    liability_confidence = None

    liability_status_scores = {
        "0-25,000": 8,
        "25,000-50,000": 7,
        "50,000-75,000": 6,
        "75,000-1,00,000": 5,
        "1,00,000-1,25,000": 4,
        "1,25,000-1,50,000": 3,
        "1,50,000-1,75,000": 2,
        "1,75,000-2,00,000": 1
    }
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "How much liability do you have?":
                liability_answer = answer
                liability_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if liability_answer:
        # Get the score for the provided marital status
        answer_score = liability_status_scores.get(liability_answer, 0)
        
        # Calculate the average score of the remaining fields
        remaining_scores = [score for status, score in liability_status_scores.items() if status != liability_answer]
        average_remaining_score = sum(remaining_scores) / len(remaining_scores)
        
        # Calculate the marital_status_score
        liability_status_score = ((answer_score * liability_confidence) + \
                               ((1 - liability_confidence) * average_remaining_score)) / 8

        return round(liability_status_score, 2)
    else:
        raise ValueError("The question 'How much liability do you have?' was not found or is invalid in the CSV.")

def calculate_age_score(csv_file_path):
    # Initialize variable to store the answer and confidence
    age_answer = None
    age_confidence = None
    
    # Open and read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            question, answer, confidence = row
            
            # Ensure the confidence is a float
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = None  # Handle invalid confidence values
                
            # Check if the question matches "What is your marital status?"
            if question == "How old are you?":
                age_answer = answer
                age_confidence = confidence
                break

    # Compute the dependents_score if an answer was found
    if age_answer:
        age_status_score = max(0, min(10, (65 - int(age_answer)) / 4)) / 10
        return round(age_status_score, 2)
    else:
        raise ValueError("The question 'How old are you?' was not found or is invalid in the CSV.")

def calculate_user_tolerance_score(csv_file_path):
    score = (calculate_age_score(csv_file_path) 
         + calculate_liability_score(csv_file_path)
         + calculate_marital_status_score(csv_file_path) 
         + calculate_dependents_score(csv_file_path) 
         + calculate_employment_score(csv_file_path) 
         + calculate_income_score(csv_file_path) 
         + calculate_home_score(csv_file_path) 
         + calculate_investment_score(csv_file_path) 
         + calculate_reaction_score(csv_file_path) 
         + calculate_volatility_score(csv_file_path) 
         + calculate_horizon_score(csv_file_path) 
         + calculate_capacity_score(csv_file_path) 
         + calculate_goal_score(csv_file_path) 
         + calculate_career_score(csv_file_path) 
         + calculate_property_score(csv_file_path) 
         + calculate_fixed_score(csv_file_path) 
         + calculate_return_score(csv_file_path)) / 17
    return round(score, 2)

csv_file_path = '/home/arnavbhatt/project/answers.csv'
score = calculate_user_tolerance_score(csv_file_path)
