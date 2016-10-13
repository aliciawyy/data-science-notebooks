
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn import preprocessing
from scipy.stats import skew, boxcox
from IPython.core.pylabtools import figsize
import xgboost as xgb
get_ipython().magic(u'matplotlib inline')

from os import path
to_filename = lambda name: path.join("..", "data", "allstate", name +".csv")

import seaborn as sns
sns.set_style("whitegrid")


# In[2]:

train = pd.read_csv(to_filename("train"), index_col=0)
test = pd.read_csv(to_filename("test"), index_col=0)
print("shape: train {}, test {}".format(train.shape, test.shape))
print(train.head(2))


# In[3]:

response = np.log(train.loss)

mean_resp = np.mean(response)
std_resp = np.std(response)
response = (response - mean_resp) / std_resp


def restore_pred1(y):
    return np.exp(y)

def restore_pred(y):
    return np.exp(y * std_resp + mean_resp)


# In[4]:

cat_features = [col for col in train.columns if col.startswith("cat")]
print("Categorical columns:", cat_features)


# In[ ]:

# Categorical features preprocessing
# Method 1: Encoding categorical features into int
"""
for col in cat_features:
    encd = preprocessing.LabelEncoder()
    encd.fit(train[col].value_counts().index.union(test[col].value_counts().index))
    train[col] = encd.transform(train[col])
    test[col] = encd.transform(test[col])
"""

# In[5]:

# Method 2: Using cardinal features for categorical features
col = cat_features[0]
test_col = train[col][:10].copy()
for col in cat_features:
    key_map = response.groupby(train[col]).mean().to_dict()
    train[col] = train[col].replace(key_map)
    for k in set(test[col].value_counts().index).difference(key_map.keys()):
        key_map[k] = np.NAN
    test[col] = test[col].replace(key_map)


# In[6]:

# preprocess numerical features
num_features = [col for col in train.columns if col.startswith("cont")]
print("Numerical columns:", num_features)


# In[7]:

# Method 1: Standard Scaler
for col in num_features:
    sc = preprocessing.StandardScaler()
    sc.fit(pd.concat([train[[col]], test[[col]]]))
    train[col] = sc.transform(train[[col]])
    test[col] = sc.transform(test[[col]])


# In[ ]:
"""
# study the skewness in the numerical features
skewed_feats = pd.concat([train[num_features], test[num_features]]).apply(lambda x: skew(x.dropna()))
print("Skew in numeric features:", skewed_feats)


# In[ ]:

# Method 2: Box-Cox transformation when numerical feature skewness > .25
for feat in skewed_feats[skewed_feats > 0.25].index:
    train[feat], lam = boxcox(train[feat] + 1.)
    test[feat], lam = boxcox(test[feat] + 1.)
"""

# In[8]:

dtrain = xgb.DMatrix(train.drop("loss", 1), response)
dtest = xgb.DMatrix(test)


# In[9]:

params = {'objective':"reg:linear", 'silent': True, 'max_depth': 7, 'min_child_weight': 1,
          'colsample_bytree': .7, "subsample": 1., 'eta': 0.1, 'eval_metric':'mae',# "n_estimators": 20,
          "gamma": 0.25, "lambda": 0.8, "silent": 1}


# In[ ]:
"""
cvresult = xgb.cv(params, dtrain, nfold=3, num_boost_round=50)
print(cvresult)


# In[ ]:

cvresult[["test-mae-mean", "train-mae-mean"]].plot()


# In[ ]:

cvresult[["test-mae-mean", "train-mae-mean"]].plot()
"""

# In[10]:

folds = 10

pred_test = 0.
pred_train = 0.
restored_pred_train = 0.
restored_pred_test = 0.

kf = KFold(n_splits=folds)
kf.split(train)
for i, (train_index, test_index) in enumerate(kf.split(train)):
    train_pd_ind = train.index[train_index]
    test_pd_ind = train.index[test_index]
    train_part, test_part = train.ix[train_pd_ind], train.ix[test_pd_ind]
    
    dtrain_part = xgb.DMatrix(train_part.drop("loss", 1), response[train_pd_ind])
    dtest_part = xgb.DMatrix(test_part.drop("loss", 1), response[test_pd_ind])
    params['seed'] = i * 5 + 100
    clf = xgb.train(params, dtrain_part, num_boost_round=10000,
                    evals=[(dtrain_part, "train"), (dtest_part, "eval")])
    
    print("best ntree limit", i, clf.best_ntree_limit)
    this_pred_train = clf.predict(dtrain, ntree_limit=clf.best_ntree_limit)
    print("mae for part train",i, mean_absolute_error(
            train_part.loss, restore_pred(clf.predict(dtrain_part, ntree_limit=clf.best_ntree_limit))))
    print("mae for part test",i, mean_absolute_error(
            test_part.loss, restore_pred(clf.predict(dtest_part, ntree_limit=clf.best_ntree_limit))))
    print("mae for all train",i, mean_absolute_error(train.loss, restore_pred(this_pred_train)))
    
    pred_train += this_pred_train
    restored_pred_train += restore_pred(this_pred_train)
    
    this_pred_test = clf.predict(dtest, ntree_limit=clf.best_ntree_limit)
    pred_test += this_pred_test
    restored_pred_test += restore_pred(this_pred_test)
    


# In[11]:

print("mae final restore after", mean_absolute_error(train.loss, restore_pred(pred_train / folds)))
print("mae final restore before", mean_absolute_error(train.loss, restored_pred_train / folds))


# In[12]:

import datetime
result = pd.DataFrame({"id": test.index, "loss": restored_pred_test / folds})
result.to_csv("result_restored_before{:%Y%m%d%H%-M}.csv".format(datetime.datetime.now()), index=None)

"""
# # Using XGBRegressor and important features

# In[ ]:

from sklearn.grid_search import GridSearchCV
from xgboost import XGBRegressor


# In[ ]:

params_reg = dict(params)
params_reg.pop("eta")
params_reg.pop('eval_metric')
params_reg.pop('lambda')


# In[ ]:

reg = XGBRegressor(**params_reg)
reg.fit(train.drop("loss", 1), train.loss)


# In[ ]:

train_predprob = reg.score()


# In[ ]:

reg_booster = reg.booster()


# In[ ]:

figsize(18, 5)
feat_imp = pd.Series(reg.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')


# In[ ]:

important_features = list(feat_imp[feat_imp > 4].index)
print("important features:", important_features)


# In[ ]:

dtrain_imp = xgb.DMatrix(train[important_features], train.loss)
cvresult = xgb.cv(params, dtrain_imp, nfold=4, num_boost_round=50)
print(cvresult)


# In[ ]:

params2 = {'base_score': 0.1, 'colsample_bytree': 0.9,
 'eta': 0.3,
 'eval_metric': 'mae',
 'max_depth': 7,
 'min_child_weight': 3,
 'n_estimators': 10,
 'objective': 'reg:linear',
 'seed': 1,
 'silent': True}
regb = xgb.train(params2, dtrain_imp, num_boost_round=50, evals=[(dtrain_imp, "train")])
"""

# In[ ]:

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator=reg, 
 param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=4, iid=False, cv=5)


# In[ ]:

gsearch1.fit(train.drop("loss", 1), train.loss)


# In[ ]:



