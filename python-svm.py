# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:30:44 2015

@author: mah228
"""

import numpy as np
import pandas as pd
##python script for building energy
data = pd.read_csv("H:/building-energy-paper/elsarticle/elsarticle/sjmr11.csv")
data['diff.1hr']=data['Temp']-data['temp.1hr']
data['diff.2hr']=data['Temp']-data['temp.2hr']
data['RH.Temp']=data['RH']*data['Temp']

skc = data['SKC']
#skc=get_dummies(skc)
skc=pd.get_dummies(skc)
#data1 = merge(data,skc,how='outer')
#data1 = pd.merge(data,skc,how='outer')
#data1 = pd.concat(data,skc,axis=1)
data1 = pd.concat([data,skc],axis=1)
#day.night = data['Night.Day']
day = data['Night.Day']
day = pd.get_dummies(day)
data1 = pd.concat([data1,day],axis=1)
data['Time'].dtypes
#time=as.str(data['Time'])
#time=str(data['Time'])
#time.dtypes
#time=pd.get_dummies(time)
time=data['Time']
time=data['Time'].apply(str)
#time.dtype
time=pd.get_dummies(time)
#time['nan'].sum()
time['0.0'].sum()
time['0.25'].sum()
data1 = pd.concat([data1,time],axis=1)
month=data['Month']
month = data['Month'].apply(str)
month = pd.get_dummies(month)
data1 = pd.concat([data1,month],axis=1)
day1=data['Day.of.Week']
day1 = pd.get_dummies(day1)
data1 = pd.concat([data1,day1],axis=1)
## python svm with all variables
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import ParameterGrid
from sklearn.svm import SVR
from sklearn import svm

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
                    
scores = ['precision', 'recall']     

grid = [ {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
           
#######################
############
data2 = pd.read_csv("H:/building-energy-paper/elsarticle/elsarticle/sjmr-test.csv")
data2['diff.1hr']=data['Temp']-data['temp.1hr']
data2['diff.2hr']=data['Temp']-data['temp.2hr']
data2['RH.Temp']=data['RH']*data['Temp']

skc = data2['SKC']
#skc=get_dummies(skc)
skc=pd.get_dummies(skc)
#data1 = merge(data,skc,how='outer')
#data1 = pd.merge(data,skc,how='outer')
#data1 = pd.concat(data,skc,axis=1)
data3 = pd.concat([data2,skc],axis=1)
#day.night = data['Night.Day']
day = data2['Night.Day']
day = pd.get_dummies(day)
data3 = pd.concat([data3,day],axis=1)
#data['Time'].dtypes
#time=as.str(data['Time'])
#time=str(data['Time'])
#time.dtypes
#time=pd.get_dummies(time)
time=data2['Time']
time=data2['Time'].apply(str)
#time.dtype
time=pd.get_dummies(time)
#time['nan'].sum()
time['0.0'].sum()
time['0.25'].sum()
data3 = pd.concat([data3,time],axis=1)
month=data2['Month']
month = data2['Month'].apply(str)
month = pd.get_dummies(month)
data3 = pd.concat([data3,month],axis=1)
    
day2=data3['Day.of.Week']
day2 = pd.get_dummies(day2)
data3 = pd.concat([data3,day2],axis=1) 
#############           
#variables = temp + BKN+CLR+OBS+OVC+SCT+RH+T.MAX+T.MIN.PREV+DewP+RH
#data['diff.1hr']=data['Temp']-data['temp.1hr']
#data['diff.2hr']=data['Temp']-data['temp.2hr']
train= data1[['Temp','Dewp','RH','T.MAX','T.MIN','T.MEAN','T.MAX.PREV','RH.MEAN','RH.MEAN.PREV',
'T.MIN.PREV','T.MEAN.PREV','diff.1hr','diff.2hr','temp.24hr','temp.48hr','RH.Temp',
'temp.1hr','temp.15hr','temp.30hr','temp.45hr','temp.2hr','temp.115hr','temp.130hr','temp.145hr','O.M']]

train = pd.concat([train,data1.ix[:,76:184]],axis=1)
train_np=train.as_matrix()
test= data1[['Temp','Dewp','RH','T.MAX','T.MIN','T.MEAN','T.MAX.PREV','RH.MEAN','RH.MEAN.PREV',
'T.MIN.PREV','T.MEAN.PREV','diff.1hr','diff.2hr','temp.24hr','temp.48hr','RH.Temp',
'temp.1hr','temp.15hr','temp.30hr','temp.45hr','temp.2hr','temp.115hr','temp.130hr','temp.145hr','O.M']]
test = pd.concat([test,data1.ix[:,76:198]],axis=1)
test['Mon']=0
test_np=test.as_matrix()
energy = data1['Energy'].as_matrix()

energy_test=data3['Energy'].as_matrix()

#######################
svr=SVR(epsilon=0)
grid = [ {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
clf1 = GridSearchCV(svr, grid,cv=3)
clf1.fit(train_np,energy)


clf1=SVR(kernel='rbf',C=9,gamma=7e-03,epsilon=0)
clf1.fit(train,energy)


pred1 = clf1.predict(test)
w = clf.coef_[0]
print(w)

from sklearn.metrics import explained_variance_score


from sklearn.metrics import mean_squared_error


from sklearn.metrics import r2_score
print explained_variance_score(energy_test,pred1)
print mean_squared_error(energy_test,pred1)
r2_score(energy_test,pred1)
######################
n_samples, n_features = 100, 5
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
z=np.random.randn(20,5)
z1=np.random.randn(20)
clf.fit(X, y)   
clf.predict(z)
#########################
from sklearn.ensemble.forest import RandomForestRegressor
regressor = RandomForestRegressor()
parameters = [{"n_estimators": [250, 500, 1000,2000]}]

# Returns the best configuration for a model using crosvalidation
# and grid search

import time


regressor = RandomForestRegressor(n_estimators=300, min_samples_split=1,max_features=67)

regressor.fit(train_np,energy)
pred=regressor.predict(test_np)

print explained_variance_score(energy_test,pred)
print mean_squared_error(energy_test,pred)
r2_score(energy_test,pred)




##prediction comparison
comp = pd.read_csv("H:/bee-efficiency/cisco presentation/pred.csv")




    