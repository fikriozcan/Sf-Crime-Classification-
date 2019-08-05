# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:18:18 2019

@author: ASUSNB
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv')

df['PdDistrict'].value_counts()

df['Time'] = pd.to_datetime(df['Time'])
df['Hour'] = df.Time.dt.hour
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.weekday

df.info()


## removing outliers, 143 outliers removed 
df = df[df.X < -122.3549]
cal = 2215024 - 2214881 

###########dropping duplicates, 488076 row removed 

df = df.drop_duplicates('IncidntNum').sort_index()                 

cal2 = 2214881 - 1746805

#### data transition, recovered vehicle and secondatry cohesion removed from categories
df['Category'].value_counts()
fd = df[~df['Category'].isin(['SECONDARY CODES', 'RECOVERED VEHICLE'])


## train and test splitting/ stratify 

train = df.loc[:,['Day','Hour','Year','Month','PdDistrict','X','Y']]

target = df.loc[:,['Category']]

train.info()
from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
target['Category'] = le1.fit_transform(target['Category'])

train.loc[:, "PdDistrict"] = pd.factorize(train.PdDistrict)[0]# suna bak

train['PdDistrict'].value_counts()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import statsmodels.formula.api as smf 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


X_train, X_test,y_train, y_test = train_test_split(train,target,test_size = 0.25,random_state = 508,stratify = target)

from sklearn.linear_model import LogisticRegression


logreg = LogisticRegression()
logreg_fit = logreg.fit(X_train, y_train)
logreg_pred = logreg_fit.predict(X_test)

# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))




from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true = y_test,y_pred = logreg_pred))


#######################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors' : range(1,18,2),
              'weights' : ['uniform','distance'],
              'metric' : ['minkowski','manhattan','euclidean','chebyshev']}

grid = GridSearchCV(KNeighborsClassifier(),parameters)

model = grid.fit(X_train,y_train)
model

print(model.best_params_,model.best_estimator_,model.best_score_)



### data visualization


sns.lineplot(x='Year', y='crimeCount', data = df2.reset_index())

fig, ax = plt.subplots(figsize=(20,20))
sns.barplot(x='Category', y='index', data = df['Category'].value_counts().reset_index())


fig, ax = plt.subplots(2,2, figsize=(25,25))
sns.barplot(x='index', y='PdDistrict', data = df['PdDistrict'].value_counts().reset_index(), ax=ax[0,1])
ax[0,1].set_title('Number of Incidents in each PdDistrict')
ax[0,1].set_ylabel('Number of Incident ')
ax[0,1].set_xlabel(' PdDistrict')


fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(x='Resolution', y='index', data = df['Resolution'].value_counts().reset_index())


fig, ax = plt.subplots(2,2, figsize=(15,15))
sns.barplot(x='index', y='DayOfWeek', data = df['DayOfWeek'].value_counts().reset_index(), ax=ax[0,1])
ax[0,1].set_title('Number of instances in each Day')
ax[0,1].set_ylabel('Day of Week')
ax[0,1].set_xlabel('Number of Instances')




fig, ax = plt.subplots(figsize=(60,20))
ax = sns.boxplot(x="Category", y="Hour", data=df)


sns.lineplot(x='Month', y='crimeCount', data = df3.reset_index())



fig, ax = plt.subplots(figsize=(60,20))
ax = sns.boxplot(x="Category", y="Month", data=df)


## suclar hangı distritlerde oluyor. ? 

others = df[df.Category == 'OTHER OFFENSES']
assault= df[df.Category == 'ASSAULT']
vehicle_theft= df[df.Category == 'VEHICLE THEFT']
drug = df[df.Category == 'DRUG/NARCOTIC']
vandalism = df[df.Category == 'VANDALISM']


fig, ax = plt.subplots(3,2, figsize=(23,15))
sns.countplot(x="PdDistrict", data=theft,palette="Blues_d", ax=ax[0,0])
ax[0,0].set_ylabel('Total Larcency/Theft')

sns.countplot(x='PdDistrict',  data = others, palette="Blues_d", ax=ax[0,1])
ax[0,1].set_ylabel('Total Other crimes')
 
sns.countplot(x='PdDistrict', data = assault,palette="Blues_d", ax=ax[1,0])
ax[1,0].set_ylabel('Total assault')

sns.countplot(x='PdDistrict', data =vehicle_theft,palette="Blues_d", ax=ax[1,1])
ax[1,1].set_ylabel('Total vehicle_theft')

sns.countplot(x='PdDistrict', data = drug,palette="Blues_d", ax=ax[2,0])
ax[2,0].set_ylabel('Total amount of Drug')

sns.countplot(x='PdDistrict', data =vandalism,palette="Blues_d", ax=ax[2,1])
ax[2,1].set_ylabel('Total amount  of Vandalism')


#############################################################

arrested = df.loc[
df['Resolution'] == 'LOCATED'].groupby(['Category',"DayOfWeek"]).count()['IncidntNum'] / df.groupby('Category').count()['IncidntNum']


by_offer_type_df8 = df.loc[
df['Resolution'] == 'ARREST, BOOKED'].groupby(['Category']).count()['IncidntNum'] / df.groupby('Category').count()['IncidntNum']

ax = (by_offer_type_df8).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Percentage of Arrest')
plt.show()

by_offer_type_df9 = df.loc[
df['Resolution'] == 'ARREST, BOOKED'].groupby(['PdDistrict']).count()['IncidntNum'] / df.groupby('PdDistrict').count()['IncidntNum']

ax = (by_offer_type_df9).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Percentage of Arrest')
plt.show()









