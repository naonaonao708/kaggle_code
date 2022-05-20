import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
submission = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')


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

kf = KFold(n_splits=3, shuffle=True)
score_list = []
models = []


for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
    print(f'fold{fold_ + 1}start')
    train_x = X_train.iloc[train_index]
    valid_x = X_train.iloc[valid_index]
    train_y = y_train[train_index]
    valid_y = y_train[valid_index]
    
    lgb_train = lgb.Dataset(train_x,train_y)
    lgb_valid = lgb.Dataset(valid_x,valid_y)
    
    lgbm_params = {'objective':'binary'}
    
    gbm = lgb.train(params=lgbm_params,
               train_set=lgb_train,
               valid_sets=[lgb_train,lgb_valid],
               early_stopping_rounds=20,
               verbose_eval=-1)
    
    oof = (gbm.predict(valid_x) > 0.5).astype(int)
    score_list.append(round(accuracy_score(valid_y,oof) * 100,2))
    models.append(gbm)
    print(f'fold{fold_+1}end\n')
print(score_list,'平均score',round(np.mean(score_list),2))


test_pred = np.zeros((len(test),3))

for fold_, gbm in enumerate(models):
    pred_ = gbm.predict(test)
    test_pred[:,fold_] = pred_


submission['Transported'] = (np.mean(test_pred, axis=1)>0.5).astype(bool)
submission.to_csv('submission_3_cross.csv',index=False)