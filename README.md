# credit-risk-classification
Module 20 Challenge: Supervised Learning


## Overview of the Analysis

The dataset contains information about historical lending activity from a peer-to-peer lending services loans. It includes the following features:
1) The size of the loan <br>
2) The interest rate <br>
3) The borrower's income <br>
4) The ratio of debt to the borrower's income <br>
5) The number of accounts the borrower has with the lending service <br>
6) The number of derogatory marks for the borrower (negative information that appear on a borrower's credit report, indicating a history of delinquent payments or other defaults on a loan or credit account.)	<br>
7) The borrower's total debt <br>
The dataset also includes the loan status - whether the loan was 'Healthy' or 'High-Risk.' <br>
<br>
The purpose of the analysis was to establish a robust Supervised Machine Learning model that can be used on additional data to predict/classify whether a loan is at risk or healthy, based on the above given features.
<br>
We used the dataset to test two logistic regression models. <br>


## Machine Learning Model 1
**Based on the Original Imbalanced Target Labels.** <br>
The first model used the original data. We noted that the original data has imbalanced categories of the target variable (loan status). Many more loans were classified as healty (n = 75,036) compared to high-risk (n = 2,500). We first split the dataset to the features (the seven listed above, noted as X), and to the target (the loan status, noted as y). We then splitted the data themselves to train data (used to find the model) and test data (used to validate the model). Once the data were split, we instantiated the logistic regression model with 
the solver parameter set as 'lbfgs' (an optimization algorithm that is appropriate for small to medium-sized datasets). We then fitted model using the train data (X_train, and y_train). Finally, we validated the model using the test data - we made predictions based on the test features (X_test) and then assessed the predictions against the actual test target (y_test). We used several measures to assess the model.

## Machine Learning Model 2
**Based on Resampled Data using Balanced Target Labels.** <br>
In the second model we used the RandomOverSampler module from the imbalanced-learn library to resample the data and create a balanced dataset in terms of the target categories. After instiating the random oversampler model we fitted the original training data to this model to create a balanced sample (X_train_resampled, y_train_resampled). We found that the new sample is balanced with regards to the target variable (loan status). 56,277 loans were classified as healty, and 56,277 loans were classified as high-risk. We then instantiated the logistic regression model with the solver parameter set as 'lbfgs' and fitted model using the resampled data (X_train_resampled, y_train_resampled). Finally, we validated the model using the test data - we made new predictions (balanced_predictions) based on the test features (X_test) and then assessed the predictions against the actual test target (y_test). We used the same measures to assess this model and compared it to the first model.
The results of both models are provided below.
## Results  
* **Machine Learning Model 1:**
  * Accuracy = (TP + TN) / (TP + TN + FP + FN) = 0.99
  * Healthy Loans: <br>
        
        * Precision = TP(Healthy Loans) / (TP(Healthy Loans) + FP(Healthy Loans)) = 1
        * Recall  = TP(Healthy Loans) / (TP(Healthy Loans) + FN(Healthy Loans)) = 1 
        * f1-score = 2 * (Precision * Recall) / (Precision + Recall) = 1
  * High-Risk Loans: <br>
        
        * Precision = TP(High-Risk Loans) / (TP(High-Risk Loans) + FP(Healthy Loans)) = 0.87
        * Recall  = TP(High-Risk Loans) / (TP(High-Risk Loans) + FN(High-Risk Loans)) = 0.89 <br>
        * f1-score = 2 * (Precision * Recall) / (Precision + Recall) = 0.88

* **Machine Learning Model 2:**
  * Accuracy = (TP + TN) / (TP + TN + FP + FN) = 0.995
  * Healthy Loans:<br>
        
        * Precision = TP(Healthy Loans) / (TP(Healthy Loans) + FP(Healthy Loans)) = 1 <br>
        * Recall  = TP(Healthy Loans) / (TP(Healthy Loans) + FN(Healthy Loans)) = 1 <br>
        * f1-score = 2 * (Precision * Recall) / (Precision + Recall) = 1
  * High-Risk Loans:<br>
        
        * Precision = TP(High-Risk Loans) / (TP(High-Risk Loans) + FP(Healthy Loans)) = 0.87
        * Recall  = TP(High-Risk Loans) / (TP(High-Risk Loans) + FN(High-Risk Loans)) = 1
        * f1-score = 2 * (Precision * Recall) / (Precision + Recall) = 0.93

## Summary
The first model (the imbalanced model) accuracy  - the ratio of correctly classifing the loan instances to the total number of instances -  is high (99%). <br> For predicting healthy loans, all measures are at 100% - precision (ratio of true positives to the total number of instances predicted as positive), recall (ratio of true positives to the total number of positive instances) and f1-score (the harmonic mean of precision and recall). <br> However, the presicion, recall and f1-score for the high-risk loans are considerably lower at 87%, 89%, and 88% respectively. <br>
Compared to the initial model, the new model that is based on balanced sampling of the test data is more accurate (99.5% for the new model vs 99% for the initial one). The precision, recall, and f-1 score for the healthy loan classification are the same as the original model.
The big difference between the models is showing for the high-risk loans. Here we see a big improvement in the recall and f1-score (precision stays the same). Thus, the new model is better in classifing both health and high-risk loans and should be used for future data classifications.

For the purpose of classifying loans it is much more important to make sure that the classification of the high-risk loans are classified correctly. and the second model do just that - it provides a robust classification model not just for classifing healthy loans, but also high-risk ones. Thus, I would recommend using the second model (based on balanced target data).
