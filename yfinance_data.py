# Extracting information from Yahoo finance dataset
import yfinance as yf  # Import the yfinance library for accessing financial data
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations

# USE THE STOCK SYMBOL
stock = yf.Ticker("AAPL")  # Initialize a Ticker object for Apple Inc. (AAPL)

# Get the number of elements in the stock's info dictionary
print(len(stock.info))  # Prints the total number of keys in the info dictionary (132)

# Print all the values in the stock's info dictionary
print(stock.info.values())  # Prints the values in the info dictionary

# Print all the keys in the stock's info dictionary
print(stock.info.keys())  # Prints the keys in the info dictionary

# Print the entire stock's info dictionary
print(stock.info)  # Prints all the key-value pairs in the info dictionary

#print some historical data
hist = stock.history(period="5d")
print(hist)

# IT WILL GIVE:
#                                  Open        High         Low       Close  \
# Date                                                                        
# 2024-10-28 00:00:00-04:00  233.320007  234.729996  232.550003  233.399994   
# 2024-10-29 00:00:00-04:00  233.100006  234.330002  232.320007  233.669998   
# 2024-10-30 00:00:00-04:00  232.610001  233.470001  229.550003  230.100006   
# 2024-10-31 00:00:00-04:00  229.339996  229.830002  225.369995  225.910004   
# 2024-11-01 00:00:00-04:00  220.970001  225.350006  220.270004  222.910004   

#                              Volume  Dividends  Stock Splits  
# Date                                                          
# 2024-10-28 00:00:00-04:00  36087100        0.0           0.0  
# 2024-10-29 00:00:00-04:00  35417200        0.0           0.0  
# 2024-10-30 00:00:00-04:00  47070900        0.0           0.0  
# 2024-10-31 00:00:00-04:00  64370100        0.0           0.0  
# 2024-11-01 00:00:00-04:00  65242200        0.0           0.0  

# FOR FINE TUNING MODELS LIKE PROPHET MODELS WE WILL BE USING THE ABOVE DATA. 
