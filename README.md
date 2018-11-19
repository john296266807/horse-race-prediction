# horse-race-prediction

# 0. Background
   This is a project for a machine learning course of The Chinese University of Hong Kong (CSCI3320 Fundamentals of Machine learning).
   
# 1. Data folder
   In the folder, there are two csv files (data source: https://www.kaggle.com/lantanacamara/hong-kong-horse-racing).
   The dataset has data of 2367 races in Hong Kong in 2014-2017, covering 2162 horses, 106 jockeys and 95 trainers.
   race-result-race.csv describes details the races while each entry of race-result-horse.csv corresponds to one horse in a race.

# 2. Preprocessing
   preprocess.py has three tasks: a) remove data without final racing rank, b) calculate average rank of the past six races for the jockeys and horses, and c) split the dataset into training data and testing data.
   
   Training data is used to train the models and testing data is used to test the models.
   
# 3. Classification
   classification.py determines if a particular horse in a race can win, get into top 3, and/or get into top 50%.
   Four types of models are used: a) Logistic regression, b) Naive Bayes (Apart from the one in the sklearn.naive_bayes library, I also try to implement it myself in naive_bayes.py), c) Support Vector Machine (SVM), and d) Random forest.
   
# 4. Regression
   regression.py predicts the final finishing time of each horse in each race.
   Two models are used: a) Support Vector Regression (SVR) and b) Gradient Boosting Regression Tree Model (GBRT).
   
   I also try to standardize the data before applying them to the models.
   
# 5. Betting
   I try to drive strategies by combining the models and setting constraints on win odds (betting.py).
   
# 6. Visualization
   I create five plots with the scripts in the 'plot' folder.
   The results are shown in the document "plot_result.pdf".
   
   
