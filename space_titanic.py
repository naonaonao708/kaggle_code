import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
submission = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')


sns.countplot(x='Transported',data=train)

display(train['Transported'].value_counts())


sns.countplot(x='VIP', hue='Transported',data=train)
display(pd.crosstab(train['VIP'],train['Transported']))


sns.distplot(train['Age'].dropna(),kde=False,bins=30,label='all')
sns.distplot(train[train['Transported']==1].Age.dropna(),kde=False,bins=30,label='transport')
sns.distplot(train[train['Transported']==0].Age.dropna(),kde=False,bins=30,label='non-transport')


int(train['Transported'])


train = pd.get_dummies(train,columns=['CryoSleep','VIP'])
test = pd.get_dummies(test,columns=['CryoSleep','VIP'])

train.drop(['PassengerId','HomePlanet','Cabin','Destination','Name','Transported'],axis=1,inplace=True)
test.drop(['PassengerId','HomePlanet','Cabin','Destination','Name'],axis=1,inplace=True)

X_train = train