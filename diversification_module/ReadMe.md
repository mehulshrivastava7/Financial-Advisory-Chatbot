"""This code is by Mehul Shrivastava"""

"""


This idea is to create a diversified stock portfolio by performing clustering analysis.


Motivation:

The goal of this module is to enhance portfolio diversification by identifying clusters of stocks based on various financial and market features. 

The output of this is used in the recommendation module. 

The module helps us identify the clusters and then recommend the stocks from other clusters which are performing good. All those stocks are taken as input in the recommendation module and then it chooses the stocks from them 
to give the final output.

References:


I am not using anything directly from these papers, but these are what I came across initially which helped me choose the features and discuss importance of diversifications in detail. 

1. "Creating Diversified Portfolios Using Cluster Analysis" by Karina Marvin:

   - Discusses the importance of diversification and how clustering can be used to achieve a balanced portfolio.
   
2. https://medium.com/@facujallia/stock-classification-using-k-means-clustering-8441f75363de

   - I am not sure if a article like this is a source which I can use, if not then below I described a rationale of using these features.

Feature Selection Rationale:

- **Sector:** Categorical variable indicating the industry sector of the stock.
- **Market Cap:** Represents the size of the company; a key indicator of the stock's risk and potential growth.
- **Volatility:** A measure of the stock's risk and stability.

Methodology:
------------
The script follows these main steps:
1. **Data Collection:** Fetches historical stock data using yFinance in batches to handle API rate limits.
2. **Data Preprocessing:** Handles missing values, encodes categorical features, and scales numerical features.
3. **Optimal Clustering:** Determines the optimal number of clusters using the Elbow Method and Silhouette Score.
4. **Clustering Analysis:** Performs K-Means clustering on the preprocessed stock data.
5. **Portfolio Analysis:** Analyzes the user's current stock portfolio and identifies underrepresented clusters.
6. **Stock Recommendation:** Recommends top-performing stocks from unrepresented clusters to enhance diversification.

Output:
-------
1. A CSV file (`indian_stocks_clusters.csv`) containing the clustering results for each stock.

                                                                              """

