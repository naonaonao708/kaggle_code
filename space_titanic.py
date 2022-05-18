# Light GBM　ホールドアウト法

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


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


train['Transported'] = train['Transported'].astype(int)


train = pd.get_dummies(train,columns=['CryoSleep','VIP'])
test = pd.get_dummies(test,columns=['CryoSleep','VIP'])

train.drop(['PassengerId','HomePlanet','Cabin','Destination','Name'],axis=1,inplace=True)
test.drop(['PassengerId','HomePlanet','Cabin','Destination','Name'],axis=1,inplace=True)

X_train = train.drop(['Transported'],axis=1)
y_train = train['Transported']

train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.33, random_state=0)

lgb_train = lgb.Dataset(train_x, train_y)
lgb_eval = lgb.Dataset(valid_x, valid_y)

lgbm_params = {'objective':'binary'}

evals_result = {}
gbm = lgb.train(params=lgbm_params,
               train_set=lgb_train,
               valid_sets=[lgb_train,lgb_eval],
               early_stopping_rounds=20,
               evals_result=evals_result,
               verbose_eval=10);

oof = (gbm.predict(valid_x) > 0.5).astype(int)
print('score', round(accuracy_score(valid_y,oof) * 100,2))


plt.plot(evals_result['training']['binary_logloss'],label='train_loss')
plt.plot(evals_result['valid_1']['binary_logloss'],label='valid_loss')
plt.legend()

submission['Transported'] = (gbm.predict(test) > 0.5).astype(bool)
submission.to_csv('submission.csv',index=False)