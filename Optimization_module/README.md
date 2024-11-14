
'''The code in this module was written by me(Ramanan)

# Explanation of optimization.py file
The optimization.py file is a function that takes tickers as input and changes the weights given to each ticker in such a way that the sharpe ratio is maximised. It uses a stochastic optimization algorithm where the initial weights are initialised randomly 10 times the number of assets times to separate iterations of changing the weights and optimizing the sharpe ratio.
This will search throughout all the weights possible and because the initial weights are being initialised randomly a lot of times , the optimization won't be stuck
in a local minima and would be searching the whole weights domain.

# Explanation of returns_optimization.py file
The returns_optimization.py file is a function that takes tickers and risk_tolerance from the User_Risk_Assessment module (This part has been integrated into the UI by me(Ramanan) as can be seen in app.py)
to maximise the returns for a particular risk_tolerance level. The idea of the optimization is that the risk value which is calculated by weights.T.cov.weights is kept less than
the risk_tolerance and a stochastic optimization is performed to maximise returns in a similar way sharpe ratio was maximised in the optimization.py function.

# Explanation of how risk from User_Risk_Assessment was compared
I(Ramanan) took the risk from User_Risk_Assessment module as a percentage and made it into a numerical value between 0 and 1 and got that as input to the returns_optimization.py as can be seen in the app.py file. The risk value of a stock in the returns_optimization.py file i.e weights.T.cov.weights is a value that is not necessarily between 0 and 1, so to make it be in between 0 and 1, I took the correlation matrix instead of the covariance matrix added 1 to each entry and divided by 2, so that each entry of the correlation matrix is between 0 and 1 and then i applied softmax activation to each row of the matrix, so that the sum of entries in a row is 1 and as the weights add up to 1, by cauchy schwartz inequality the dot product of each row and the weights will be less than 1. But the sum of entries of this vector won't be 1 , so we apply softmax on this again and make the sum equal to 1 and dot it again with the weights vector to get a final value less between 0 and 1, which we compare with the risk_tolerance ratio gotten from the User_Risk_Assessment module
and get the maximum returns which has risk less than or equal to the risk_tolerance.
'''
