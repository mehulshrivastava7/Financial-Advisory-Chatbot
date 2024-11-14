'''The code in this module was written by me, Ramanan
the optimization.py file is a function that takes tickers as input and changes the weights given to each ticker in such a way that the sharpe ratio is maximised. It uses 
a stochastic optimization algorithm where the initial weights are initialised randomly 10 times the number of assets to separate iterations of changing the weights and optimizing the sharpe ratio.
This will search throughout all the weights possible and because the initial weights are being generated randomly , the optimization won't be stuck
in a local minima and would be searching the whole weights domain.

The returns_optimization.py file is a function that takes tickers and risk_tolerance from the User_Risk_Assessment module (This part has been integrated into the UI too as can be seen in app.py by me)
to maximise the returns for a particular risk_tolerance level. The optimization is done in a way that the risk value which is calculated by weights.T.cov.weights is kept less than
the risk_tolerance and a stochastic optimization is performed to maximise returns in a similar way sharpe ratio was maximised from the previous function.
