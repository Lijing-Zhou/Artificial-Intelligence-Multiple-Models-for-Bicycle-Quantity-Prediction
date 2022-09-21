# 21768704 
#
'''
Careful !!!!!!!!!!!: 
ery Important !!!!!!!!: 
I change the Functioning Day to Functioning_Day. 
I add a _ to instead the space, 
PLS change it in the coursework_other.csv 
if you want to run the python file!!!
{Functioning Day    ---  Functioning_Day })
'''
'''
KFold this part should be runned finialy;
if u want to run this part ,remove the big Comments (''' ''') below
'''

# clear varibles
for key in list(globals().keys()):
 if (not key.startswith("_")) and (key !="key"):
     globals().pop(key)
del(key)    

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

# import the csv file : Careful !!!!!!!!!!!:  {Functioning Day    ---  Functioning_Day })
path = '/Users/Lijing/Desktop/AIG/coursework_other.csv'
data=pd.read_csv('/Users/Lijing/Desktop/AIG/coursework_other.csv',encoding='latin1')

#  The Pretreatment of the data

# extract the day and month
data['month'] = pd.DatetimeIndex(data.Date).month
data['day'] = pd.DatetimeIndex(data.Date).dayofweek

#transfer the Seasons Holiday Functioning_Day data
data.loc[data.Seasons=='Winter','Seasons'] = '1'
data.loc[data.Seasons=='Spring','Seasons'] = '2'
data.loc[data.Seasons=='Summer','Seasons'] = '3'
data.loc[data.Seasons=='Autumn','Seasons'] = '4'
data.loc[data.Holiday=='Holiday','Holiday'] = '1'
data.loc[data.Holiday=='No Holiday','Holiday'] = '0'
data.loc[data.Functioning_Day=='Yes','Functioning_Day'] = '1'
data.loc[data.Functioning_Day=='No','Functioning_Day'] = '0'

df1 = pd.DataFrame(np.random.randn(8760, 1))
df2 = pd.DataFrame(np.random.randn(8760, 12))
df3 = pd.DataFrame(np.random.randn(8760, 13))
df1 = data.iloc[0:8760,[1]]
df2 = data.iloc[0:8760,[2,3,4,5,6,7,8,9,10,11,12,13,14,15]]


X = df2 #data
Y = df1 #target

#  firs_time create the train and traget data
from sklearn.model_selection import train_test_split
Xtr, Xtest, Ytr, Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 10)



from sklearn import linear_model
from sklearn import model_selection
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# baseline model
svc0 = DummyRegressor()
svc0.fit(Xtr, Ytr)
svc0tr=svc0.score(Xtr, Ytr)
svc0test=svc0.score(Xtest, Ytest)

#1. Random forests model
svc1 = RandomForestRegressor(n_estimators = 100)
svc1.fit(Xtr, Ytr)
svc1tr=svc1.score(Xtr, Ytr)
svc1test=svc1.score(Xtest, Ytest)

Ytrpred1=svc1.predict(Xtr)
Ytestpred1=svc1.predict(Xtest)
# scores, Mean Squared Error (MSE) and  R^2.
meantr1= mean_squared_error(Ytrpred1, Ytr)
R2tr1=r2_score(Ytrpred1, Ytr)
meantest2= mean_squared_error(Ytestpred1, Ytest)
R2test1=r2_score(Ytestpred1, Ytest)






# KFold this part should be runned finialy;
# if u want to run this part ,remove the big Comments below
# It will take more than 10 mins to run the code!!
# beacuse it rechange the the the train and test data 
# in the forms of array. 

'''
# It will take more than 10 mins to run the code!!

# recreate the train and test data in the forms of array. 
X = np.array(df2)  #data
Y = np.array(df1) #target
Xtr, Xtest, Ytr, Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 10)
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, random_state=63, shuffle=True)
# from 100 200 ... 1000
max_n = 10
r1 = [[] for _ in range(max_n)]
r2 = [[] for _ in range(max_n)]
n=0
for n in range(max_n):
    
    scr = RandomForestRegressor(n_estimators = 100*n+100)
    
    for train_index, val_index in kf.split(Xtr):
        Xtrain, Xval = Xtr[train_index], Xtr[val_index]
        Ytrain, Yval = Ytr[train_index], Ytr[val_index]
        scr.fit(Xtrain, Ytrain)

        

    
        scr.fit(Xtrain, Ytrain)
        Ypredtrain=scr.predict(Xtrain)
        Ypredval=scr.predict(Xval)

      
        r1[n].append(scr.score(Xtrain,Ytrain))
        r2[n].append(scr.score(Xval,Yval))

train_accuracy_mean = np.mean(r1, axis=1)
train_accuracy_stdev = np.std(r1, axis=1)
val_accuracy_mean = np.mean(r2, axis=1)
val_accuracy_stdev = np.std(r2, axis=1)

assert(np.shape(train_accuracy_mean)==(max_n,))
assert(np.shape(train_accuracy_stdev)==(max_n,))
assert(np.shape(val_accuracy_mean)==(max_n,))
assert(np.shape(val_accuracy_stdev)==(max_n,))

fig1=plt.figure()
x = list(range(1,max_n+1))
plt.plot(x,train_accuracy_mean, label = 'Training Accuracy')
plt.plot(x,val_accuracy_mean, label = 'Validation Accuracy')
plt.xlabel('n_estimators')
plt.xticks(x)
plt.legend()

fig2=plt.figure()
x = list(range(1,max_n+1))
plt.plot(x,r1, label = 'Training Scores')
plt.plot(x,r2, label = 'Test Scores')
plt.xlabel('n_estimators')
plt.xticks(x)
plt.legend()
'''





#  Hyperparameters part of n_estimators and max_depth 
# if u want to run this part ,remove the big Comments below

# use Grid Search to test the different values of n_estimators 0,100,500 
tuned_parameters = [{'n_estimators':[10,100,500]}]   
scores = ['r2']

for score in scores:
    
    print(score)
    
    clf = model_selection.GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(Xtr, Ytr)

   
    #best_estimator_ returns the best estimator chosen by the search
    print(clf.best_estimator_)
 
    cv_results = clf.cv_results_
    params = cv_results['params']
    mean_test_score = cv_results['mean_test_score']
    std_test_score = cv_results['std_test_score']
  
    print("Scores: ")
    print("")
    for i in range(len(params)):
        print("%0.3f (+/-%0.03f) for %r" % (mean_test_score[i], std_test_score[i] / 2, params[i]))
    print("")
    # print(cv_results)


# use Grid Search to test the different values of max_depth 0,100,500 

tuned_parameters = [{'max_depth':[2,4,6,8]}] 
scores = ['r2']

for score in scores:
    
    print(score)
    
    clf = model_selection.GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(Xtr, Ytr)

   
    #best_estimator_ returns the best estimator chosen by the search
    print(clf.best_estimator_)
 
    cv_results = clf.cv_results_
    params = cv_results['params']
    mean_test_score = cv_results['mean_test_score']
    std_test_score = cv_results['std_test_score']
  
    print("Scores: ")
    print("")
    for i in range(len(params)):
        print("%0.3f (+/-%0.03f) for %r" % (mean_test_score[i], std_test_score[i] / 2, params[i]))
    print("")
    # print(cv_results)


# 2. Linear Regression
svc2 = linear_model.LinearRegression()
svc2.fit(Xtr, Ytr)
svc2tr=svc2.score(Xtr, Ytr)
svc2test=svc2.score(Xtest, Ytest)

Ytrpred2=svc2.predict(Xtr)
Ytestpred2=svc2.predict(Xtest)

# scores, Mean Squared Error (MSE) and  R^2.
meantr2= mean_squared_error(Ytrpred2, Ytr)
R2tr2=r2_score(Ytrpred2, Ytr)
meantest2= mean_squared_error(Ytestpred2, Ytest)
R2test2=r2_score(Ytestpred2, Ytest)

# 3. Ridge Regression
svc3 = linear_model.Ridge()
svc3.fit(Xtr, Ytr)
svc3tr=svc3.score(Xtr, Ytr)
svc3test=svc3.score(Xtest, Ytest)

# 4. Epsilon-Support Vector Regression
svc4 = svm.SVR(kernel ='rbf', C = 10, gamma = .001)
svc4.fit(Xtr, Ytr)
svc4tr=svc4.score(Xtr, Ytr)
svc4test=svc4.score(Xtest, Ytest)

# 5. KNN
svc5 = KNeighborsClassifier(n_neighbors=3)
svc5.fit(Xtr, Ytr)
svc5tr=svc5.score(Xtr, Ytr)
svc5test=svc5.score(Xtest, Ytest)


