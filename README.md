a. Problem Statement
There are total of 25 features. The dataset is for Credit Card Default Payment Prediction. Each row represents separate data point for each customer. 
Based on the nominal, ordinal, ratio and interval we can perform approriate feature scaling and one hot encoding. 
Post approriate feature scaling and one hot encoding, we have to detect whether the given customer will default the payment for next month or otherwise.
It is very crucial for any investment bank, to understand how much amount will be defaulted for the next month. 
This helps bank in taking informed decision, to manage the their balance sheet and reserve and financial planning for that month. 

b. Dataset description
There are 25 columns, first columns represent the ID, and the last column represents the class to which that row belongs.
Following are the features : 
LIMIT_BAL	Ratio
SEX		Nominal
EDUCATION	Ordinal
MARRIAGE	Nominal
AGE		Ratio
PAY_0		Nominal
PAY_2		Nominal
PAY_3		Nominal
PAY_4		Nominal
PAY_5		Nominal
PAY_6		Nominal
BILL_AMT1	Ratio
BILL_AMT2	Ratio
BILL_AMT3	Ratio
BILL_AMT4	Ratio
BILL_AMT5	Ratio
BILL_AMT6	Ratio
PAY_AMT1	Ratio
PAY_AMT2	Ratio
PAY_AMT3	Ratio
PAY_AMT4	Ratio
PAY_AMT5	Ratio
PAY_AMT6	Ratio
DEFAULT		class 0 
		or 1

One can perform Scaling on Ratio and OneHotEncoding for Nominal.



c. Models Used
Following are the Models Used :
LogisticRegression
DecisionTreeClassifier
KNeighborsClassifier
GaussianNB
RandomForestClassifier
XGBClassifier

ML Model Name :

Logistic Regression
Accuracy : 81.0833
AUC : 0.7252
Precision : 0.7233
Recall : 0.2344
F1 : 0.3540
MCC : 0.3361

Decision Tree 
Accuracy : 72.5667
AUC : 0.6082
Precision : 0.3838
Recall : 0.3971
F1 : 0.3904
MCC : 0.2135

kNN
Accuracy : 77.1833
AUC : 0.6763
Precision : 0.4790
Recall : 0.3617 
F1 : 0.4122
MCC : 0.2782

Naive Bayes
Accuracy : 22.1167
AUC : 0.6672
Precision : 0.2212
Recall : 1.0000
F1 : 0.3622
MCC : 0.0000

Random Forest ( Ensemble )
Accuracy : 80.9833
AUC : 0.7710
Precision : 0.6802
Recall : 0.2645
F1 : 0.3809
MCC : 0.3393

XGBoost ( Ensemble )
Accuracy : 81.6667
AUC : 0.7815
Precision : 0.6553
Recall : 0.3610
F1 : 0.4655
MCC : 0.3896

Logistic Regression :
For Logistic Regression, while training and splitting the data, the data was equally split into positive and negative class. 
So, accuracy can be considered as good estimator. But for credit card default prediction, False Negative is of utmost importance. Hence, Recall is important in this case.
The recall is 0.2344. This particular recall is considered poor.

Decision Trees :
For Decision Trees, The recall is 0.3971. This particular recall is better than the logistic regression.

kNN :
For kNN, This has recall of 0.3617. This is lower than decision trees but lesser than decision trees. 

Naive Bayes :
For Naive Bayes, This has recall of 1.0. This is the best recall. Here Gaussian Naive Bayes is used. 
The model is complete with diffferent feature engineering for other models and here the different feature engineering is used. 
Therefore, this recall of 1.0 can be wrong.

Random Forest (Ensemble) :
For Random Forest, This has recall of 0.2645. This has 2nd best recall.

XGBoost (Ensemble) :
For XGBoost, This has recall of 0.3610. This has the 3rd Largest recall.