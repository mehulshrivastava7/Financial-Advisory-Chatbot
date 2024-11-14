"""This code is written by Nitin Vetcha"""

Description:
------------
This script is designed to recommend stocks based on user profile. It takes into consideration the stocks provided by the diversification module as well as user's personal preferences provided at the time of filling the questionnaire.  

Motivation:
-----------
User's often tend to overlook the diversity of their portfolio, and might end up investing heavily in companies of a particular sector only. This might cause an issue if the sector crashes down, for example, during COVID-19, the IT sector crashed and Health Service, Financial Services sectors bloomed. Hence, diversification plays a key role to ensure stability against market conditions even in unpredictable scenarious ensuring the loss and gains are not too skewed. However, its also important to take the personal preferences of the user into account as well while recommending stcoks. For example, an ardent risk taker, perhaps with an appetite for gambling would, for the sake of high returns in short duration, prefer to invest in highly volatile stocks while a long-term investor might as well plan accordingly and invest in low volatile stocks with moderate returns, increasing the importance for personalized recommendtaions

Input:
------------
The stocks suggested by diversification module and the scores obtained by performing the finnacial questionaire would be provided as csv files to the module.

Data Processing:
----------------
Now, each stock would be assigned a score, based on which it would be recommended. Our variables of interest would be risk, market cap, average returns, volatility, P/E ratio. Firstly, for each variable of interest, all stocks would be classified into three (five) categories - (very low : 0) , low : 1, balanced : 2 , high : 3 , (very high : 4), with their value indicated as well. Similarly, the user would also be classified into these five categories based on his overall risk and volatility scores. Now, corresponding to the category of the user, the corresponding set of stocks would recieve additional 3 points as well, and it would decrease by 1 point as it spreads accross. To explain, if the user is classifed as low risk, then the stocks which are classified as low risk receive 3 additional points, stocks which are classified as very low & balanced risk receive 2 additional points, stocks which are classified as high risk receive 1 additional points and stocks which are classified as very high risk receive 0 additional points. Finally, top k (=6) stocks with highest scores would be recommended to the user and provided as a csv file. 

Overview of the Code:
---------------------
The script is structured into the following main components:

1. **Data Processing:** implements the facebook/bart-large-mnli model to compute confidence values for each class
2. **Scoring:** computes the score using the predefined weights for each class, taking into account the confidence level of each class as well

How to Use:
-----------
1. Provide the suggested stocks and financial scores in the form of csv files as input.
2. Run the script to obtain your personalized recommendations.


