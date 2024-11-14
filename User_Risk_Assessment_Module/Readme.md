
"""This code is written by Nitin Vetcha"""

Description:
------------
This script is designed to assess the risk tolerance level of the user based on his financial knowledge & risk aptitude (with no regards to personal details as was in the case of Financial Questionnaire). The resulting score takes into account primarily three factors of the user - Investment Risk, Risk Comfort & Experience, Speculative Risk.

Motivation:
-----------
The primary motivation to implement this module stems for the notable highly cited paper - Financial risk tolerance revisited : the development of a risk assessment instrument - published in the prestigious Financial Review Service journal in 1999. Based on the study perfomered, it was noted that there exists a positive correlation between risk  tolerance and income, investment knowledge and positive stock market expectations. This prompted me to assess the individual's risk tolerance value with the help of a personal questionnaire

Input:
------------
User would be permitted to  mark a single correct answer for each of the 20 questions. Each option has a predetermined weight indicating its contribution to the score, which would then be summed up.

Overview of the Code:
---------------------
The script is structured into the following main components:

1. **Questionnaire:** contains questions with the pre-determined class values
2. **Scoring:** computes the score using the predefined weights for each option followed by finally summing up all weights

How to Use:
-----------
1. Answer the questionnaire while selecting a single correct option for each question
2. Run the script to compute your risk tolerance score.

References:
-----------
1.https://static.arnaudsylvain.fr/2017/03/Grable-Lyton-1999-Financial-Risk-revisited.pdf
