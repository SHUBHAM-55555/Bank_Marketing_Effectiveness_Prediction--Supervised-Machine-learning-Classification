# Bank_Marketing_Effectiveness_Prediction--Supervised-Machine-learning-Classification

![image](https://github.com/SHUBHAM-55555/Bank_Marketing_Effectiveness_Prediction--Supervised-Machine-learning-Classification/assets/135215329/5ea36799-968f-41cf-a99e-d9d616ee9dd6)

Marketing is the most common method which many companies are using to sell their products, services and reach out to the potential customers to increase their sales. Telemarketing is one of them and most useful way of doing marketing for increasing business and build good relationship with customers to get business for a company.

It’s also important to select and follow up with those customers who are most likely to subscribe product or service. There are many classification models, such as Logistic Regression, Decision Trees, Random Forest, KNN, ANN and Support Vector Machines (SVM) that can be used for classification prediction.

# Problem Statement

The given dataset is of a direct marketing campaign (Phone Calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (Target variable y).

We were provided with following dataset:

Bank Client data: age (numeric) job : type of job
marital : marital status
education
default: has credit in default?
housing: has housing loan?
loan: has personal loan? Related with the last contact of the current campaign: contact: contact communication type
month: last contact month of year
day of week: last contact day of the week
duration: last contact duration, in seconds (numeric).
Other attributes: campaign: number of contacts performed during this campaign
pdays: number of days that passed by after the client was last contacted from a previous campaign
previous: number of contacts performed before this campaign
poutcome: outcome of the previous marketing campaign

Output variable (desired target):

y - has the client subscribed a term deposit? (binary: 'yes','no')

![image](https://github.com/SHUBHAM-55555/Bank_Marketing_Effectiveness_Prediction--Supervised-Machine-learning-Classification/assets/135215329/dad9a76c-eeae-4440-a8f5-497eab98ab24)


# Introduction
m 3

Marketing is the most common method which many companies are using to sell their products, services and reach out to the potential customers to increase their sales. Telemarketing is one of them and most useful way of doing marketing for increasing business and build good relationship with customers to get business for a company.

It’s also important to select and follow up with those customers who are most likely to subscribe product or service. There are many classification models, such as Logistic Regression, Decision Trees, Random Forest, KNN, ANN and Support Vector Machines (SVM) that can be used for classification prediction.

# Classification Approach

After understanding the problem statement, we loaded the dataset for following operations:

Data Exploration

Exploratory Data Analysis

Feature Engineering

Feature selection

Balancing Target Feature

Building Model

Hyperparameter Tuning

Dataset Exploration

The given dataset was initially loaded for a quick overview.

It was observed that our dataset contains 45211 records and 17 features. Datatypes of features was then checked and it was found that there are 7 numerical (int) and 10 Categorical (object) datatypes among which no null values and duplicated records were found in our dataset.

# Analysis

we can conclude that when the client age categories are 'stable' ,'old age' and 'about to retire' then their is very high possibilty that those category person subscribe for a term deposit. when clients category is struggling and counting last breathe then there is very less possibility that a customer subscribe for term deposit

Bank has not contacted most of the clients ,the clients which bank not contacted before have high posibility that they suscribe for term deposite than a client which bank contacted before.

Most of the clients in our dataset was not credit defaulter so that when the client has credit is not in default then there is high possibility that customer suscribe for term deposite.

when the client is credit default there is very less possibility that a customer suscribe for term deposite.

we can roughly conclude that when that balance was from 500-35000 (in the middle range) then those customer subscribed for the term deposit so we can say that high balance or low balance will not be predict that client will subscribed for term deposit or not

we can see that the pdays have most of the values are 0 and less than 0 so we have to drop that column for better prediction of our mode

we can conclude that when contact communication type is cellular then there is high possibility that the client subscribe a term deposit hence the bank should contact the customer by cellular type mostly.

when the contact communication type is telephone then there was very less possibility that the client subscribe a term deposit.

we can conclude that when the customer education is tertiary and secondary then there is a high possibility that client subscribe a term deposit hence bank should approach mostly to the tertiary and secondary class education client to subscribe for term deposit.

When the education of the customer is unknown and primary those client have very low possibility to subscribe for term deposit.

Most of the clients who are married and single had subscribed for the term deposit therefore , When marital status of client is 'Single' and 'married' then there are high possibility that those clients subscribe a term deposit .Bank should target 'single' and 'married' client both to subscribe for term deposit.

when clients marital status was devorced those clients did not subscribe for the term deposit much that’s why , When the client marital status is divorced then there is very less chance that these clients agrees to subscribe for term deposit.

Most of clients are from the job called as 'blue collar, management, technician and admin, when the client jobs are Management , technician, blue_ collar, admin services the there is high chance that those customers subscribe for term deposit so that bank should prefer salaried persons most to approach for term deposit.

when the client is retired person we can see high probability to subscribe term deposit hence retired client has high possibility that they subscribe for term deposit bank should communicate mostly to retired person to subscribe for term deposit.

when a clients are self employed and entrepreneur we can see less probability for subscribe to term deposit as well as when a clients have a category house maid , unemployed and student and unknows there are least possibility that those customers agree to subscribe for term deposit.

# Feature Engineering

![image](https://github.com/SHUBHAM-55555/Bank_Marketing_Effectiveness_Prediction--Supervised-Machine-learning-Classification/assets/135215329/34c3a0a1-6199-400f-91f1-189c67af77d1)


Feature engineering is one of the important steps in model building and thus we focused more into it. We performed the following in feature engineering

# Dealing with outliers

![image](https://github.com/SHUBHAM-55555/Bank_Marketing_Effectiveness_Prediction--Supervised-Machine-learning-Classification/assets/135215329/680eb03e-7817-4ab0-bfe1-5c59ba5e1122)

After looking at the plots above we removed the outliers

In duration we removed those observation with no output and duration> 2000s

In campaign we removed campaigns> 20

In previous we removed observations for previous contacts> 11

# SMOTE Oversampling

![image](https://github.com/SHUBHAM-55555/Bank_Marketing_Effectiveness_Prediction--Supervised-Machine-learning-Classification/assets/135215329/a61ca790-44c9-4a21-8b4e-0d783fa3fd03)

To start with building first we dealt with highly imbalanced data using SMOTE and then feature standardization.

The target variable contains highly imbalanced labeled data in the 88:12 ratio. Using SMOTE which is basically used to create synthetic class samples of minority class to balance the distribution of target variable. The target variable balanced for modeling.

Original Dataset length 45211 Dataset length after SMOTE Oversampling 79844

# There are several classification models available for prediction/classification.

In this project we used following models for classification Algorithm’s

KNN

Random Forest

XGBOOST

XGBOOST with Hyperparameter tunning

#K-Nearest neighbors (KNN)

K-Nearest Neighbor is a non-parametric supervised learning algorithm both for classification and regression. The principle is to find the predefined number of training samples closest to the new point and predict the correct label from these training sample.

It’s a simple and robust algorithm and effective in large training datasets.

Following are steps involved in KNN.

Select the K value.

Calculate the Euclidean distance between new point and training point.

According to the similarity in training data points, distance and K value the new data point gets assigned to the majority class.

Cross_validation score [0.76655256 0.7773232 0.77723971 0.78239813 0.7752171 ] KNN Test accuracy Score 0.7889885276288763 precision recall f1-score support

       0       0.84      0.71      0.77      9981
       1       0.75      0.87      0.80      998

# Random Forest

Random forest is a Decision Tree based algorithm. It’s a supervised learning algorithm. This algorithm can solve both type of problems i.e. classification and regression. Decision Trees are flexible and it often gets overfitted.

To overcome this challenge Random Forest helps to make classifications more efficiently.

It creates a number of decision trees from a randomly selected subset of the training set and averages the-final outcome. Its accuracy is generally high. Random forest has ability to handle large number of input variables.

Cross_validation score [0.89997495 0.90030893 0.89922351 0.8995491 0.90405812] RandomForest Test accuracy Score 0.9070186864385552 precision recall f1-score support

       0       0.87      0.96      0.91      9981
       1       0.95      0.86      0.90      9980

# XGBOOST
XGboost is the most widely used algorithm in machine learning, whether the problem is a classification or a regression problem.

It is known for its good performance as compared to all other machine learning algorithms.

Cross_validation score [0.93186942 0.93270435 0.93170243 0.9261022 0.93470274] xgb Test accuracy Score 0.9356244677120384 precision recall f1-score support

       0       0.90      0.98      0.94      9981
       1       0.97      0.90      0.93      9980


# Hyperparameter tuning of XG Boost Classifier
Cross_validation score [0.93137419 0.93454667 0.93270997 0.93737475 0.93269873 0.93102872 0.92334669 0.93036072 0.93670675 0.93219773] xgb_hypertuned Test accuracy Score 0.9358749561645208 precision recall f1-score support

       0       0.91      0.97      0.94      9981
       1       0.97      0.90      0.93      9980


# Model Evaluation
For classification problems we have different metrics to measure and analyze the model’s performance.

In highly imbalanced target feature accuracy metrics doesn’t represents true reality of model.


# Conclusion-
It was a great learning experience working on a Bank dataset.

From the above model explanatory tool we have seen that poutcome Unknown is the most important feature while predicting our target variable also from the table we can see that when the poutcome is 0 then it contribute in the negative way and increases the probability of predicting 0.

Marital married is the second most important feature for predicting target when the marital married then it will affect positively and increases the probability of predicting 1.

Also age cat stable variable affect positively on the target variable when the age of clients is stable then it will increases the probability of predicting 1 that means it higher the probability that client will subscribe for term deposit.

Also education secondary affects positively on the target variable when the client education is secondary then it increases the probability that client will agree to subscribe for term deposit.

From the above project we can conclude that XG boost classifier is the best fit classification model for predicting weather the client agree to subscribe for personal loan or not.

When we Hypertuned these XG Boost classifier the accuracy of the model increases by 1 % So it predicts 94% prediction correctly. There are some important feature for predicting our target variable we use Shapash model explanatory to explore that features.

We visualize 20 feature which are most important while predicting target variable. From that feature we conclude that clients age , education ,job and and marital status and outcome of previous campaign are the most important feature for predicting that weather client agree to subscribe for term deposit or not that’s why bank prefer these information to start for new campaign and to target customer.

# Future Scope -
Our main objective is to get good precision score for without 'duration' models and good recall score for 'duration' included model.

So, we can initially formulate the required time to converge a lead using 'duration' included models and then sort out precise leads for 'duration' excluded models using this formulated time.

Here, the idea is to find out responses for any particular record with varying assumed predefined duration range.

In this way we can help marketing team to get precise leads along with time required to converge that lead and also, those leads that have least probability to converge (if we get no positive response for any assumed duration). Thus, an effective marketing campaign can be executed with maximum leads converging to term deposit.
