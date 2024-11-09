#Code for User Input Processing
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
        reaction_status_score = (answer_score * reaction_confidence) + \
                               ((1 - reaction_confidence) * average_remaining_score)

        return round(reaction_status_score, 2)
    else:
        raise ValueError("The question 'What would you do if your investment lost 20 percent in a year?' was not found or is invalid in the CSV.")

# Example usage:
csv_file_path = 'answers.csv'
score = calculate_reaction_score(csv_file_path)
print(f"Market Reaction Score: {score}")
