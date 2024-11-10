import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('updated_sample_indian_stocks_data 2.csv')  # Replace 'your_file.csv' with your CSV file path

# Extract distinct values from the 'sector' column
distinct_sectors = df['Sector'].dropna().unique()

# Convert to a list if needed, and print the distinct sectors
distinct_sectors_list = distinct_sectors.tolist()
print(distinct_sectors_list)

# Load the CSV file into a DataFrame
file_path = 'updated_sample_indian_stocks_data 2.csv'  # replace with your file path
df = pd.read_csv(file_path)

# Check if the 'Volatility' column exists in the DataFrame
if 'Volatility' in df.columns:
    # Filter out rows where volatility is 0 or less (if needed)

    df_filtered = df[df['Volatility'] > 0]

    # Find the maximum and minimum values of the filtered volatility column
    max_volatility = df_filtered['Volatility'].max()
    min_volatility = df_filtered['Volatility'].min()

    print(f"Maximum volatility (above 0): {max_volatility}")
    print(f"Minimum volatility (above 0): {min_volatility}")
else:
    print("The 'Volatility' column does not exist in the CSV file.")
  
# Check if the 'Average Return' column exists in the DataFrame
if 'Average Return' in df.columns:
    # Filter out rows where Average Return is 0 or less (if needed)

    df_filtered = df[df['Average Return'] > 0]

    # Find the maximum and minimum values of the filtered Average Return column
    max_Average = df_filtered['Average Return'].max()
    min_Average = df_filtered['Average Return'].min()

    print(f"Maximum Average Return (above 0): {max_Average }")
    print(f"Minimum Average Return (above 0): {min_Average }")
else:
    print("The 'Average Return' column does not exist in the CSV file.")

# Check if the 'P/E Ratio' column exists in the DataFrame
if 'P/E Ratio' in df.columns:
    # Clean the 'P/E Ratio' column:
    # 1. Replace infinite values (inf and -inf) with NaN
    # 2. Drop rows where 'P/E Ratio' is NaN
    df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['P/E Ratio'])

    # Find the maximum and minimum values of the cleaned 'P/E Ratio' column
    max_pe_ratio = df_cleaned['P/E Ratio'].max()
    min_pe_ratio = df_cleaned['P/E Ratio'].min()

    print(f"Maximum P/E Ratio: {max_pe_ratio}")
    print(f"Minimum P/E Ratio: {min_pe_ratio}")
else:
    print("The 'P/E Ratio' column does not exist in the CSV file.")
