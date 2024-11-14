"""This code is by Nitin"""

def categorize_user_volatility(volatility_score):
    """
    Categorize the user's volatility score.
    
    Parameters:
        volatility_score (float): A score between 0 and 1.
        
    Returns:
        str: The category name ('Low Volatile', 'Medium Volatile', 'High Volatile').
    """
    if 0 <= volatility_score < 0.33:
        return 'Low Volatile'
    elif 0.33 <= volatility_score < 0.66:
        return 'Medium Volatile'
    else:
        return 'High Volatile'

def update_volatility_scores(csv_file_path, updated_csv_file_path, answers_csv_file_path):
    """
    Update the 'Score' column in the CSV based on user volatility from answers.csv.
    
    Parameters:
        csv_file_path (str): Path to the input CSV file (recommended stocks).
        updated_csv_file_path (str): Path to save the updated CSV file.
        answers_csv_file_path (str): Path to the answers CSV file containing user responses.
    """
    # Read the main CSV file into a pandas DataFrame with error handling
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The main CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The main CSV file is malformed.")
        return

    # Convert 'Volatility' to absolute values to ensure they are positive
    if 'Volatility' not in df.columns:
        print("Error: 'Volatility' column not found in the main CSV.")
        return
    df['Volatility'] = df['Volatility'].abs()

    # Initialize the dictionary to store categorized tickers
    volatility_dict = {
        'Low Volatile': [],
        'Medium Volatile': [],
        'High Volatile': []
    }

    # Calculate thresholds for categorization
    max_volatility = df['Volatility'].max()
    min_volatility = df['Volatility'].min()
    range_volatility = max_volatility - min_volatility

    if range_volatility == 0:
        print("All 'Volatility' values are the same. No update needed.")
        return

    one_third_volatility = min_volatility + range_volatility / 3
    two_third_volatility = min_volatility + 2 * range_volatility / 3

    # Iterate over each row in the DataFrame to categorize tickers
    for index, row in df.iterrows():
        vol = row['Volatility']
        ticker = row['Ticker']
        
        if min_volatility <= vol < one_third_volatility:
            volatility_dict['Low Volatile'].append(ticker)
        elif one_third_volatility <= vol < two_third_volatility:
            volatility_dict['Medium Volatile'].append(ticker)
        elif two_third_volatility <= vol <= max_volatility:
            volatility_dict['High Volatile'].append(ticker)

    # Ensure the 'Score' column exists; if not, create it initialized to 0
    if 'Score' not in df.columns:
        df['Score'] = 0

    # Ensure the 'Score' column is of integer type
    try:
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0).astype(int)
    except ValueError:
        print("Error: 'Score' column contains non-numeric values that cannot be converted.")
        return

    # Read the answers CSV file to extract the volatility score
    try:
        answers_df = pd.read_csv(answers_csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{answers_csv_file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The answers CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The answers CSV file is malformed.")
        return

    # Verify required columns exist
    required_columns = {'question', 'answer', 'confidence', 'score'}
    if not required_columns.issubset(answers_df.columns):
        print(f"Error: The answers CSV file must contain the columns: {required_columns}")
        return

    # Extract the volatility score for the specific question
    question = "What level of volatility would you be most comfortable with?"
    matched_scores = answers_df[answers_df['question'] == question]['score']

    if matched_scores.empty:
        print(f"Error: The question '{question}' was not found in the answers CSV.")
        return

    volatility_score = matched_scores.values[0]

    # Validate the volatility_score
    if not isinstance(volatility_score, (int, float)):
        print(f"Error: The volatility score '{volatility_score}' is not a valid number.")
        return
    if not (0 <= volatility_score <= 1):
        print(f"Error: The volatility score '{volatility_score}' is out of bounds (0 <= score <= 1).")
        return

    # Categorize the user based on the volatility score
    try:
        category = categorize_user_volatility(volatility_score)
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    # Get the list of tickers for the determined category
    tickers_to_update = volatility_dict.get(category, [])

    if not tickers_to_update:
        print(f"No tickers found for category '{category}'. No scores updated.")
    else:
        # Check which tickers exist in the DataFrame
        existing_tickers_mask = df['Ticker'].isin(tickers_to_update)
        existing_tickers = df.loc[existing_tickers_mask, 'Ticker']
        missing_tickers = set(tickers_to_update) - set(existing_tickers)

        if missing_tickers:
            print(f"Warning: The following tickers were not found in the CSV and will be skipped: {missing_tickers}")

        # Increment the 'Score' by 1 for the matching tickers
        df.loc[existing_tickers_mask, 'Score'] += 1
        # print(f"Incremented 'Score' by 1 for {existing_tickers_mask.sum()} tickers in category '{category}'.")

    # Save the updated DataFrame to a new CSV file
    try:
        df.to_csv(updated_csv_file_path, index=False)
        # print(f"Updated CSV saved to '{updated_csv_file_path}'.")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")

def categorize_user_risk(risk_score):
    """
    Categorize the user's risk score.
    
    Parameters:
        risk_score (float): A score between 0 and 1.
        
    Returns:
        str: The category name ('Conservative', 'Moderately Conservative', 'Balanced', 'Moderately Aggressive', 'Aggressive').
    """
    if 0 <= risk_score < 0.30:
        return 'Conservative'
    elif 0.30 <= risk_score < 0.50:
        return 'Moderately Conservative'
    elif 0.50 <= risk_score < 0.70:
        return 'Balanced'
    elif 0.70 <= risk_score < 0.90:
        return 'Moderately Aggressive'
    elif 0.90 <= risk_score <= 1.0:
        return 'Aggressive'
    else:
        raise ValueError(f"Risk score {risk_score} is out of bounds (0 <= score <= 1).")

def update_risk_scores(csv_file_path, updated_csv_file_path, answers_csv_file_path):
    """
    Update the 'Score' column in the CSV based on user risk from answers.csv.
    
    Parameters:
        csv_file_path (str): Path to the input CSV file (recommended stocks).
        updated_csv_file_path (str): Path to save the updated CSV file.
        answers_csv_file_path (str): Path to the answers CSV file containing user responses.
    """
    # Read the main CSV file into a pandas DataFrame with error handling
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The main CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The main CSV file is malformed.")
        return

    # Convert 'Risk Score' to absolute values to ensure they are positive
    if 'Risk_Score' not in df.columns:
        print("Error: 'Risk Score' column not found in the main CSV.")
        return

    # Initialize the dictionary to store categorized tickers
    risk_dict = {
        'Conservative': [],
        'Moderately Conservative': [],
        'Balanced': [],
        'Moderately Aggressive': [],
        'Aggressive': []
    }

    min_risk = df['Risk_Score'].min()
    max_risk = df['Risk_Score'].max()

    # Calculate thresholds for categorization based on fixed ranges
    # Alternatively, you can calculate dynamic thresholds based on min and max
    # Here, we'll use fixed thresholds as per categorization function
    # Iterate over each row in the DataFrame to categorize tickers
    for index, row in df.iterrows():
        risk = (row['Risk_Score'] - min_risk) / (max_risk - min_risk)
        ticker = row['Ticker']
        
        if 0 <= risk < 0.30:
            risk_dict['Conservative'].append(ticker)
        elif 0.30 <= risk < 0.50:   
            risk_dict['Moderately Conservative'].append(ticker)
        elif 0.50 <= risk < 0.70:
            risk_dict['Balanced'].append(ticker)
        elif 0.70 <= risk < 0.90:
            risk_dict['Moderately Aggressive'].append(ticker)
        elif 0.90 <= risk <= 1.0:
            risk_dict['Aggressive'].append(ticker)
        else:
            print(f"Warning: Ticker '{ticker}' has an out-of-bounds Risk Score: {risk}")

    # Ensure the 'Score' column exists; if not, create it initialized to 0
    if 'Score' not in df.columns:
        df['Score'] = 0

    # Ensure the 'Score' column is of integer type
    try:
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0).astype(int)
    except ValueError:
        print("Error: 'Score' column contains non-numeric values that cannot be converted.")
        return

    # Read the answers CSV file to extract the risk score
    try:
        answers_df = pd.read_csv(answers_csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{answers_csv_file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The answers CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The answers CSV file is malformed.")
        return

    # Verify required columns exist
    required_columns = {'question', 'answer', 'confidence', 'score'}
    if not required_columns.issubset(answers_df.columns):
        print(f"Error: The answers CSV file must contain the columns: {required_columns}")
        return

    # Extract the risk score by summing all 'score' values and dividing by 17
    total_score = answers_df['score'].sum()
    risk_score = total_score / 17

    # Validate the risk_score
    if not isinstance(risk_score, (int, float)):
        print(f"Error: The calculated risk score '{risk_score}' is not a valid number.")
        return
    if not (0 <= risk_score <= 1):
        print(f"Error: The calculated risk score '{risk_score}' is out of bounds (0 <= score <= 1).")
        return

    # Categorize the user based on the risk score
    try:
        category = categorize_user_risk(risk_score)
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    # Get the list of tickers for the determined category
    tickers_to_update = risk_dict.get(category, [])

    if not tickers_to_update:
        print(f"No tickers found for category '{category}'. No scores updated.")
    else:
        # Check which tickers exist in the DataFrame
        existing_tickers_mask = df['Ticker'].isin(tickers_to_update)
        existing_tickers = df.loc[existing_tickers_mask, 'Ticker']
        missing_tickers = set(tickers_to_update) - set(existing_tickers)

        if missing_tickers:
            print(f"Warning: The following tickers were not found in the CSV and will be skipped: {missing_tickers}")

        # Increment the 'Score' by 1 for the matching tickers
        df.loc[existing_tickers_mask, 'Score'] += 1
        # print(f"Incremented 'Score' by 1 for {existing_tickers_mask.sum()} tickers in category '{category}'.")

    # Save the updated DataFrame to a new CSV file
    try:
        df.to_csv(updated_csv_file_path, index=False)
        # print(f"Updated CSV saved to '{updated_csv_file_path}'.")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")

def update_return_scores(csv_file_path, updated_csv_file_path):
    """
    Update the 'Score' column in the CSV based on user volatility.
    
    Parameters:
        csv_file_path (str): Path to the input CSV file.
        updated_csv_file_path (str): Path to save the updated CSV file.
    """
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The CSV file is malformed.")
        return

    # Check if the 'Volatility' column exists in the DataFrame
    if 'Average Return' in df.columns:

        # Find the maximum and minimum values of the filtered volatility column
        max_return = df['Average Return'].max()
        min_return = df['Average Return'].min()
        one_third_return = min_return + (max_return - min_return)/3
        two_third_return = min_return + 2*(max_return - min_return)/3

    # Iterate over each row in the DataFrame to categorize tickers
    for index, row in df.iterrows():
        ret = row['Average Return']
        
        if min_return <= ret < one_third_return:
            df.at[index, 'Score'] += 0  # Low Return
        elif one_third_return <= ret < two_third_return:
            df.at[index, 'Score'] += 1  # Medium Return
        elif two_third_return <= ret <= max_return:
            df.at[index, 'Score'] += 2  # High Return

    # Save the updated DataFrame to a new CSV file
    try:
        df.to_csv(updated_csv_file_path, index=False)
        # print(f"Updated CSV saved to '{updated_csv_file_path}'.")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")

def update_cap_scores(csv_file_path, updated_csv_file_path):
    """
    Update the 'Score' column in the CSV based on user volatility.
    
    Parameters:
        csv_file_path (str): Path to the input CSV file.
        updated_csv_file_path (str): Path to save the updated CSV file.
    """
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The CSV file is malformed.")
        return

    # Check if the 'Volatility' column exists in the DataFrame
    if 'Market Cap' in df.columns:

        # Find the maximum and minimum values of the filtered volatility column
        max_cap = df['Market Cap'].max()
        min_cap = df['Market Cap'].min()
        one_third_cap = min_cap + (max_cap - min_cap)/3
        two_third_cap = min_cap + 2*(max_cap - min_cap)/3

    # Iterate over each row in the DataFrame to categorize tickers
    for index, row in df.iterrows():
        cap = row['Market Cap']
        
        if min_cap <= cap < one_third_cap:
            df.at[index, 'Score'] += 0  # Low Return
        elif one_third_cap <= cap < two_third_cap:
            df.at[index, 'Score'] += 1  # Medium Return
        elif two_third_cap <= cap <= max_cap:
            df.at[index, 'Score'] += 2  # High Return

    # Save the updated DataFrame to a new CSV file
    try:
        df.to_csv(updated_csv_file_path, index=False)
        # print(f"Updated CSV saved to '{updated_csv_file_path}'.")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")

def update_ratio_scores(csv_file_path, updated_csv_file_path):
    """
    Update the 'Score' column in the CSV based on user volatility.
    
    Parameters:
        csv_file_path (str): Path to the input CSV file.
        updated_csv_file_path (str): Path to save the updated CSV file.
    """
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: The CSV file is malformed.")
        return

    # Check if the 'Volatility' column exists in the DataFrame
    if 'P/E Ratio' in df.columns:

        # Find the maximum and minimum values of the filtered volatility column
        max_ratio = df['P/E Ratio'].max()
        min_ratio = df['P/E Ratio'].min()
        one_third_ratio = min_ratio + (max_ratio - min_ratio)/3
        two_third_ratio = min_ratio + 2*(max_ratio - min_ratio)/3

    # Iterate over each row in the DataFrame to categorize tickers
    for index, row in df.iterrows():
        ratio = row['P/E Ratio']
        
        if min_ratio <= ratio < one_third_ratio:
            df.at[index, 'Score'] += 0  # Low Return
        elif one_third_ratio <= ratio < two_third_ratio:
            df.at[index, 'Score'] += 1  # Medium Return
        elif two_third_ratio <= ratio <= max_ratio:
            df.at[index, 'Score'] += 2  # High Return

    # Save the updated DataFrame to a new CSV file
    try:
        df.to_csv(updated_csv_file_path, index=False)
        # print(f"Updated CSV saved to '{updated_csv_file_path}'.")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")

def stock_prediction(questionnaire_score_csv_file_path, recommended_stocks_csv_file_path):
    df = pd.read_csv(recommended_stocks_csv_file_path)
    df['Score'] = 0
    df.to_csv(recommended_stocks_csv_file_path, index=False)  
    update_risk_scores(recommended_stocks_csv_file_path, recommended_stocks_csv_file_path, questionnaire_score_csv_file_path)
    update_volatility_scores(recommended_stocks_csv_file_path, recommended_stocks_csv_file_path, questionnaire_score_csv_file_path)
    update_ratio_scores(recommended_stocks_csv_file_path, recommended_stocks_csv_file_path)
    update_cap_scores(recommended_stocks_csv_file_path, recommended_stocks_csv_file_path)
    update_return_scores(recommended_stocks_csv_file_path, recommended_stocks_csv_file_path)
    data = pd.read_csv(recommended_stocks_csv_file_path)
    top_rows = data.sort_values(by='Score', ascending=False).head(6)
    top_companies = top_rows[['Company_Name', 'Ticker', 'Score']]
    print(top_companies.to_string(index=True))

