#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 22:33:56 2020

@author: apple
"""
# 0,import package
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

        
'''
Part 1 Load the data
'''
data1 = pd.read_csv("train.csv")
data2 = pd.read_csv("test.csv")
data3 = pd.read_csv("gender_submission.csv")
data4=pd.merge(data3,data2)
data=pd.concat([data1,data4],axis=0)
data=data.reset_index()
print(data.head())


'''
Part 2 Pre-process the data (aka data wrangling)
'''
# 1, Data cleanning
# drop the unrelated columns
data.drop(['PassengerId','Cabin','Ticket'],axis=1,inplace=True)

#2, Identification and treatment of missing values and outliers.
## find the missing value
print(data.isnull().sum())
#Find the null value in Fare catergory and fill with mean value
data[data['Embarked'].isnull()]

# Miss. Amelie and Mrs. George Nelson was embarked with 'S',since the search from 
#https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html
data['Embarked'] = data['Embarked'].fillna('S')
# find the any relationship from other column and Fare
print(data.corr())
# find the missing value in fare column
print(data[data['Fare'].isnull()])
#fill NA value within "Fare" column based on the relationship with 'Pclass'and 'Parch'
data['Fare'] = data['Fare'].fillna(data.groupby(['Pclass'])['Parch'].mean()[3])

# Since ['age'] has the large empty value,
#so fill the age with mean value
data['Age'].fillna(data['Age'].mean(), inplace = True)

# Check each numerical,compare the mean,max,min
print(data.describe())
# for the descibe table, the fare have the outlier more 400
sns.boxplot(x="Survived", y="Fare", data=data)
plt.show()
# Remove the outlier of Fare with more than 400
data.drop(data[data.Fare > 400].index, inplace=True)

#3, Feature engineering
# encoding the sex (categorical variable)
table1=pd.get_dummies(data['Sex'])
data=pd.concat([data, table1], axis=1)
# encoding the embarked (categorical variable)
table2=pd.get_dummies(data['Embarked'])
data=pd.concat([data, table2], axis=1)


'''
Part 3. Exploratory data analysis
'''
#1, At least two plots describing different aspects of the data set 
# heatmap for correlations
table3=data.drop(['Name','Sex','Embarked'],axis=1)
plt.figure(figsize=(8,8))
sns.heatmap(table3.astype(float).corr(), mask=np.triu(table3.astype(float).corr()), cmap = sns.diverging_palette(230, 20, as_cmap=True), annot=True, fmt='.1g', square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
#the relationship between survival and categorical datad catergircal data(sex,Embarked)
sns.pointplot(x="Embarked", y="Survived", hue="Sex", kind="box", data=data,palette="Set3")
plt.show()

#2, Print a basic data description 
print(data.info())

#3, Print (or include in the plots) descriptive statistics
print(data.describe())

'''
Part 4. Partition data into train, validation and test sets
'''

'''
From Lecture06.slide:
training set: 60% of total data set   1305*0.6 = 783
Validation set: 20% of total data set 1305*0.2 = 261
Testing setzz: 20% of total data set  1305*0.2 = 261
'''
train_data=data[:783]
valid_data=data[783:1044]
test_data=data[1044:]

'''
Part5. Fit models on the training set and select the best based on validation set performance.
'''
#1，building the machine learning model for both test and valid data
def build_x(df):
    return StandardScaler().fit_transform(df.drop(columns=['Name','Sex','Embarked','index','Survived']))

train_x=build_x(train_data)
valid_x=build_x(valid_data)
test_x=build_x(test_data)

train_y = train_data['Survived'].values
valid_y = valid_data['Survived'].values
test_y  =  test_data['Survived'].values

'''2, runing into different model'''
#Decision Tree Classifier
parameters={'criterion':('gini','entropy'),
            'splitter':('random','best'),'max_depth':range(1,5)}
clf=tree.DecisionTreeClassifier(random_state=30)
clf_gs=GridSearchCV(clf,parameters)
clf_gs=clf_gs.fit(train_x,train_y)
clf_score=clf_gs.score(valid_x,valid_y)

#Random Forest Classifier
parameters={'criterion':('gini','entropy'),
            'max_features':('auto','int','float'),'max_depth':range(1,5)}
random_forest=RandomForestClassifier()
random_forest_rs=RandomizedSearchCV(random_forest,parameters)
random_forest_rs=random_forest_rs.fit(train_x,train_y)
random_forest_score=random_forest_rs.score(valid_x,valid_y)

#Gradient Boosting Classifier
Gradient_Boosting=GradientBoostingClassifier().fit(train_x,train_y)
Gradient_Boosting_score=Gradient_Boosting.score(valid_x,valid_y)

#Logistic Regression
parameters={'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
            'penalty':('l1', 'l2', 'elasticnet','none')}
logis_R=LogisticRegression()
logis_R_gs=GridSearchCV(logis_R,parameters)
logis_R_gs=logis_R_gs.fit(train_x,train_y)
logis_R_score=logis_R_gs.score(valid_x,valid_y)

#Gaussian Naive Bayes(GNB)
GNB=GaussianNB().fit(train_x,train_y)
GNB.score=GNB.score(valid_x,valid_y)

#Stochastic Gradient Descent (SGD)
parameters={'loss':('deviance','exponential'),'learning_rate':[0.01,0.05,0.1,0.2],'n_estimators':[50,100,150]}
SGD=GradientBoostingClassifier()
SGD_gs=GridSearchCV(SGD,parameters)
SGD_gs=SGD_gs.fit(train_x,train_y)
SGD_score=SGD_gs.score(valid_x,valid_y)

#xgboost
Xgboost=XGBClassifier().fit(train_x,train_y)
Xgboost_score=Xgboost.score(valid_x,valid_y)

#3, select the table from best performance of validation
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest Classifier','Gradient Boosting',
              'Logistic Regression','Gaussian Naive Bayes','Stochastic Gradient Decent', 
              'xgboost'],
    'Score': [clf_score,random_forest_score,Gradient_Boosting_score,
              logis_R_score,GNB.score,SGD_score,Xgboost_score]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df)

'''
Part6. Print the results of the final model on the test set. 
'''
#find the predicted value from test_data
#i run the result table in many times, the table always show that
# the Random Forest get the highest score among different model
Y_prediction = random_forest_rs.predict(test_x)
#Accuracy
accuracy=accuracy_score(test_y, Y_prediction)
# F1-score
f1_score=f1_score(test_y, Y_prediction)
# AUC score
y_scores = random_forest_rs.predict_proba(test_x)[:,1]
r_a_score = roc_auc_score(test_y, y_scores)
Final_result = pd.DataFrame({
    'Indicator': ['Accuracy','F1 score','AUC Score'],
    'Score': [accuracy,f1_score,r_a_score]})
print(Final_result)

'''
For Kaggle submission
'''
# fix the data in to normal data for prediction 
data2['Age'].fillna(data2['Age'].mean(), inplace = True)
data2['Fare'] = data2['Fare'].fillna(data2.groupby(['Pclass'])['Fare'].mean()[3])
table1=pd.get_dummies(data2['Sex'])
data2=pd.concat([data2, table1], axis=1)
table2=pd.get_dummies(data2['Embarked'])
data2=pd.concat([data2, table2], axis=1)
data5=data2.drop(columns=['Name','Sex','Embarked','PassengerId','Cabin','Ticket'])
submission_x=StandardScaler().fit_transform(data5)
#make the prediction for kaggle
Y_prediction=random_forest_rs.predict(submission_x)
submission = pd.DataFrame({
        "PassengerId": data2["PassengerId"],
        "Survived": Y_prediction})

submission.to_csv('submission_1.csv', index=False)