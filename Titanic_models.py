import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix,precision_score,recall_score,auc,roc_curve,PrecisionRecallDisplay,RocCurveDisplay,ConfusionMatrixDisplay,classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv('C:/Users/ravit/Documents/ML/Ensemble Learning/titanic/train.csv')
test = pd.read_csv('C:/Users/ravit/Documents/ML/Ensemble Learning/titanic/test.csv')
answer = pd.read_csv('C:/Users/ravit/Documents/ML/Ensemble Learning/titanic/gender_submission.csv')
y_test = answer['Survived']
#Label Encoding of categorial variables
Labelencoder = LabelEncoder()
train.describe
# Imputing missing values
print(train.isnull().sum())
print(test.isnull().sum())
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
train['Cabin'] = train['Cabin'].fillna(train['Cabin'].mode()[0])
test['Cabin'] = test['Cabin'].fillna(test['Cabin'].mode()[0])
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

#*******************************************************************************#
# Label encoding the categorial variables using label encoder                   #
#*******************************************************************************#

train_1 = train
train_1.columns
test_1 = test
train_1['Sex'] = Labelencoder.fit_transform(train_1['Sex'])
train_1['Cabin'] = Labelencoder.fit_transform(train_1['Cabin'])
train_1['Embarked'] = Labelencoder.fit_transform(train_1['Embarked'])
test_1['Sex'] = Labelencoder.fit_transform(test_1['Sex'])
test_1['Cabin'] = Labelencoder.fit_transform(test_1['Cabin'])
test_1['Embarked'] = Labelencoder.fit_transform(test_1['Embarked'])

#*******************************************************************************#
# Since name is unique id and ticket number is given to identify the ticket 
#given to each induvidual they don't have any impact on the 
#*******************************************************************************#

train_1.drop(['Name','Ticket'],axis=1,inplace=True)
test.drop(['Name','Ticket'],axis=1,inplace=True)
#test_1.drop(['Name','Ticket'],axis=1,inplace=True)
#*******************************************************************************#
# Developing logistic regression a base model without scaling values
#*******************************************************************************#
y_train = train_1['Survived']
X_train = train_1.drop(['Survived'],axis=1)
logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)
y_pred = logistic_model.predict(test_1)
print('Accuracy of Logitic regression classifier on train set:{:.2f}'.format(logistic_model.score(X_train,y_train)))
print('Accuracy of Logitic regression classifier on train set:{:.2f}'.format(logistic_model.score(test_1,y_test)))
print('Accuracy of Logitic regression classifier on train set:{:.2f}'.format(precision_score(y_test,y_pred)))

#*******************************************************************************#
# Hyperparameter tuning of the model:
# Different hyper parameters in logistic regression are
# 1. Solver
# 2. Penalty
# 3. Max_iter
# 4. C (regularization strength)
# 5. tol
# 6. Fit_intercept
# 7. intercept scaling
# 8. class_weight(Not effective)
# 9. random_state(Not effective)
# 10. multi_class(Not effective)
# 11. verbose (Not effective)
# 12. warm_start (Not effective)
# 13. l1_ratio

#*******************************************************************************#
# After Scaling the data since we have got cabin and fare values much higher and different from other values
# In this model the default parameters (penalty = 12 and solver='lbfgs')
#*******************************************************************************#

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.values),columns = X_train.columns,index=X_train.index)
X_test = pd.DataFrame(scaler.transform(test_1),columns = test_1.columns,index=test_1.index)
print(type(X_train))
logistic_model.fit(X_train,y_train)
y_pred = logistic_model.predict(X_test)
print('Accuracy of Logitic regression classifier on train set:{:.2f}'.format(logistic_model.score(X_train,y_train)))
print('Accuracy of Logitic regression classifier on test set:{:.2f}'.format(logistic_model.score(X_test,y_test)))
print('Precision of Logitic regression classifier on test set:{:.2f}'.format(precision_score(y_test,y_pred)))

#*******************************************************************************#
# Changing penalty = None and keeping defaul solver (lbfgs)
#*******************************************************************************#
logistic_model = LogisticRegression(penalty=None)
logistic_model.fit(X_train,y_train)
y_pred = logistic_model.predict(X_test)
print('Accuracy of Logitic regression classifier on train set:{:.2f}'.format(logistic_model.score(X_train,y_train)))
print('Accuracy of Logitic regression classifier on test set:{:.2f}'.format(logistic_model.score(X_test,y_test)))
print('Precision of Logitic regression classifier on test set:{:.2f}'.format(precision_score(y_test,y_pred)))

#*************************************************************************************************#
# identifying the scores using ***penalty = None*** with all other solvers
#*************************************************************************************************#
classifier = [
    LogisticRegression(solver='newton-cg',penalty=None),
    LogisticRegression(solver='sag',penalty=None),
    LogisticRegression(solver='saga',penalty=None),
    LogisticRegression(solver='newton-cholesky',penalty=None)
]
classifier_columns = []
classifier_compare = pd.DataFrame(columns=classifier_columns)
row_index=0
for c in classifier:
    predicted = c.fit(X_train,y_train).predict(X_test)
    fp,tp,th = roc_curve(y_test,predicted)
    #classifier_name = c.__class__.__name__
    classifier_compare.loc[row_index,'Train Accuracy'] = round(c.score(X_train,y_train),5)
    classifier_compare.loc[row_index,'Test Accuracy'] = round(c.score(X_test,y_test),5)
    classifier_compare.loc[row_index,'Precision'] = round(precision_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'Recall'] = round(recall_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'AUC'] = round(auc(fp,tp),5)

    row_index +=1
classifier_compare.sort_values(by=['Test Accuracy'],ascending=False,inplace=True)
classifier_compare

#*************************************************************************************************#
# identifying the scores using ***penalty = l2*** with all other solvers
#*************************************************************************************************#
classifier = [
    LogisticRegression(solver='newton-cg',penalty='l2'),
    LogisticRegression(solver='sag',penalty='l2'),
    LogisticRegression(solver='saga',penalty='l2'),
    LogisticRegression(solver='newton-cholesky',penalty='l2')
]
classifier_columns = []
classifier_compare = pd.DataFrame(columns=classifier_columns)
row_index=0
for c in classifier:
    predicted = c.fit(X_train,y_train).predict(X_test)
    fp,tp,th = roc_curve(y_test,predicted)
    #classifier_name = c.__class__.__name__
    classifier_compare.loc[row_index,'Train Accuracy'] = round(c.score(X_train,y_train),5)
    classifier_compare.loc[row_index,'Test Accuracy'] = round(c.score(X_test,y_test),5)
    classifier_compare.loc[row_index,'Precision'] = round(precision_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'Recall'] = round(recall_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'AUC'] = round(auc(fp,tp),5)

    row_index +=1
classifier_compare.sort_values(by=['Test Accuracy'],ascending=False,inplace=True)
classifier_compare

#*************************************************************************************************#
# identifying the scores using ***penalty = l1/l2*** with solver = 'liblinear'
#*************************************************************************************************#
classifier = [
    LogisticRegression(solver='liblinear',penalty='l1'),
    LogisticRegression(solver='liblinear',penalty='l2')
]
classifier_columns = []
classifier_compare = pd.DataFrame(columns=classifier_columns)
row_index=0
for c in classifier:
    predicted = c.fit(X_train,y_train).predict(X_test)
    fp,tp,th = roc_curve(y_test,predicted)
    #classifier_name = c.__class__.__name__
    classifier_compare.loc[row_index,'Train Accuracy'] = round(c.score(X_train,y_train),5)
    classifier_compare.loc[row_index,'Test Accuracy'] = round(c.score(X_test,y_test),5)
    classifier_compare.loc[row_index,'Precision'] = round(precision_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'Recall'] = round(recall_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'AUC'] = round(auc(fp,tp),5)

    row_index +=1
classifier_compare.sort_values(by=['Test Accuracy'],ascending=False,inplace=True)
classifier_compare

#************************************************************************************************************#
# identifying the scores using ***penalty = None*** with all solvers (except liblinear) and changing max_iter
#************************************************************************************************************#
classifier = [
    LogisticRegression(solver='newton-cg',penalty=None,max_iter=100),
    LogisticRegression(solver='sag',penalty=None,max_iter=250),
    LogisticRegression(solver='saga',penalty=None,max_iter=500),
    LogisticRegression(solver='newton-cholesky',penalty=None,max_iter=750),
    #LogisticRegression(solver='liblinear',penalty='l1',max_iter=1000)
]
classifier_columns = []
classifier_compare = pd.DataFrame(columns=classifier_columns)
row_index=0
for c in classifier:
    predicted = c.fit(X_train,y_train).predict(X_test)
    fp,tp,th = roc_curve(y_test,predicted)
    classifier_name = c.__class__.__name__
    classifier_compare.loc[row_index,'Train Accuracy'] = round(c.score(X_train,y_train),5)
    classifier_compare.loc[row_index,'Test Accuracy'] = round(c.score(X_test,y_test),5)
    classifier_compare.loc[row_index,'Precision'] = round(precision_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'Recall'] = round(recall_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'AUC'] = round(auc(fp,tp),5)

    row_index +=1
classifier_compare.sort_values(by=['Test Accuracy'],ascending=False,inplace=True)
classifier_compare

#************************************************************************************************************#
# identifying the scores using ***penalty = l2*** with all solvers (except liblinear) and changing C(Regularization)
#************************************************************************************************************#
classifier = [
    LogisticRegression(solver='newton-cg',penalty='l2',C=1),
    LogisticRegression(solver='sag',penalty='l2',C=2.5),
    LogisticRegression(solver='saga',penalty='l2',C=5),
    LogisticRegression(solver='newton-cholesky',penalty='l2',C=10),
    #LogisticRegression(solver='liblinear',penalty='l1',max_iter=1000)
]
classifier_columns = []
classifier_compare = pd.DataFrame(columns=classifier_columns)
row_index=0
for c in classifier:
    predicted = c.fit(X_train,y_train).predict(X_test)
    fp,tp,th = roc_curve(y_test,predicted)
    classifier_name = c.__class__.__name__
    classifier_compare.loc[row_index,'Train Accuracy'] = round(c.score(X_train,y_train),5)
    classifier_compare.loc[row_index,'Test Accuracy'] = round(c.score(X_test,y_test),5)
    classifier_compare.loc[row_index,'Precision'] = round(precision_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'Recall'] = round(recall_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'AUC'] = round(auc(fp,tp),5)

    row_index +=1
classifier_compare.sort_values(by=['Test Accuracy'],ascending=False,inplace=True)
classifier_compare

#************************************************************************************************************#
# identifying the scores using ***penalty = l2*** with all solvers (except liblinear) and changing tol (tolerance)
#************************************************************************************************************#
classifier = [
    LogisticRegression(solver='newton-cg',penalty='l2',tol=0.0001),
    LogisticRegression(solver='sag',penalty='l2',tol=0.01),
    LogisticRegression(solver='saga',penalty='l2',tol=0.1),
    LogisticRegression(solver='newton-cholesky',penalty='l2',tol=10),
    #LogisticRegression(solver='liblinear',penalty='l1',max_iter=1000)
]
classifier_columns = []
classifier_compare = pd.DataFrame(columns=classifier_columns)
row_index=0
for c in classifier:
    predicted = c.fit(X_train,y_train).predict(X_test)
    fp,tp,th = roc_curve(y_test,predicted)
    classifier_name = c.__class__.__name__
    classifier_compare.loc[row_index,'Train Accuracy'] = round(c.score(X_train,y_train),5)
    classifier_compare.loc[row_index,'Test Accuracy'] = round(c.score(X_test,y_test),5)
    classifier_compare.loc[row_index,'Precision'] = round(precision_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'Recall'] = round(recall_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'AUC'] = round(auc(fp,tp),5)

    row_index +=1
classifier_compare.sort_values(by=['Test Accuracy'],ascending=False,inplace=True)
classifier_compare

#************************************************************************************************************#
# identifying the scores using ***penalty = l1*** with all solver(liblinear) and changing fit_intercept
#************************************************************************************************************#
classifier = [
    #LogisticRegression(solver='newton-cg',penalty='l2',tol=0.0001),
    #LogisticRegression(solver='sag',penalty='l2',tol=0.01),
    #LogisticRegression(solver='saga',penalty='l2',tol=0.1),
    #LogisticRegression(solver='newton-cholesky',penalty='l2',tol=10),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=True),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=False)
]
classifier_columns = []
classifier_compare = pd.DataFrame(columns=classifier_columns)
row_index=0
for c in classifier:
    predicted = c.fit(X_train,y_train).predict(X_test)
    fp,tp,th = roc_curve(y_test,predicted)
    classifier_name = c.__class__.__name__
    classifier_compare.loc[row_index,'Train Accuracy'] = round(c.score(X_train,y_train),5)
    classifier_compare.loc[row_index,'Test Accuracy'] = round(c.score(X_test,y_test),5)
    classifier_compare.loc[row_index,'Precision'] = round(precision_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'Recall'] = round(recall_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'AUC'] = round(auc(fp,tp),5)

    row_index +=1
classifier_compare.sort_values(by=['Test Accuracy'],ascending=False,inplace=True)
classifier_compare

#************************************************************************************************************#
# identifying the scores using ***penalty = l1*** and solver = 'liblinear' and changing intercept_scaling
#************************************************************************************************************#
classifier = [
    #LogisticRegression(solver='newton-cg',penalty='l2',tol=0.0001),
    #LogisticRegression(solver='sag',penalty='l2',tol=0.01),
    #LogisticRegression(solver='saga',penalty='l2',tol=0.1),
    #LogisticRegression(solver='newton-cholesky',penalty='l2',tol=10),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=True,intercept_scaling=0.01),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=True,intercept_scaling=0.1),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=True,intercept_scaling=0.2),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=True,intercept_scaling=0.5),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=True,intercept_scaling=1),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=True,intercept_scaling=2),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=True,intercept_scaling=5),
    LogisticRegression(solver='liblinear',penalty='l1',fit_intercept=True,intercept_scaling=10)
]
classifier_columns = []
classifier_compare = pd.DataFrame(columns=classifier_columns)
row_index=0
for c in classifier:
    predicted = c.fit(X_train,y_train).predict(X_test)
    fp,tp,th = roc_curve(y_test,predicted)
    classifier_name = c.__class__.__name__
    classifier_compare.loc[row_index,'Train Accuracy'] = round(c.score(X_train,y_train),5)
    classifier_compare.loc[row_index,'Test Accuracy'] = round(c.score(X_test,y_test),5)
    classifier_compare.loc[row_index,'Precision'] = round(precision_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'Recall'] = round(recall_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'AUC'] = round(auc(fp,tp),5)

    row_index +=1
classifier_compare.sort_values(by=['Test Accuracy'],ascending=False,inplace=True)
classifier_compare

#************************************************************************************************************#
# identifying the scores using penalty=elasticnet and solver(saga) and changing parameter as l1_ratio
#************************************************************************************************************#
classifier = [
    LogisticRegression(solver='saga',penalty='elasticnet',fit_intercept=True,intercept_scaling=0.01,l1_ratio=0),
    LogisticRegression(solver='saga',penalty='elasticnet',fit_intercept=True,intercept_scaling=0.01,l1_ratio=0.2),
    LogisticRegression(solver='saga',penalty='elasticnet',fit_intercept=True,intercept_scaling=0.01,l1_ratio=0.5),
    LogisticRegression(solver='saga',penalty='elasticnet',fit_intercept=True,intercept_scaling=0.01,l1_ratio=1)
]
classifier_columns = []
classifier_compare = pd.DataFrame(columns=classifier_columns)
row_index=0
for c in classifier:
    predicted = c.fit(X_train,y_train).predict(X_test)
    fp,tp,th = roc_curve(y_test,predicted)
    #classifier_name = c.__class__.__name__
    classifier_compare.loc[row_index,'Train Accuracy'] = round(c.score(X_train,y_train),5)
    classifier_compare.loc[row_index,'Test Accuracy'] = round(c.score(X_test,y_test),5)
    classifier_compare.loc[row_index,'Precision'] = round(precision_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'Recall'] = round(recall_score(y_test,predicted),5)
    classifier_compare.loc[row_index,'AUC'] = round(auc(fp,tp),5)

    row_index +=1
classifier_compare.sort_values(by=['Test Accuracy'],ascending=False,inplace=True)
classifier_compare
