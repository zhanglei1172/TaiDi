#!/usr/bin/env python
# coding=utf-8
import lightgbm as lgb
import numpy as np
from scipy.io import loadmat
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, auc, roc_curve, classification_report, confusion_matrix


feature = loadmat(
    '/home/zhang/.deepinwine/Deepin-TIM/drive_c/users/zhang/My Documents/Tencent Files/340000184/FileRecv/train.mat')
feature = feature['feature']

train_X, test_X, train_y, test_y = train_test_split(feature[:, :-1], feature[:, -1], test_size=0.2, shuffle=True,
                                                    stratify=feature[:, -1], random_state=2019)
skf = StratifiedKFold( shuffle=True, random_state=2019)



# lr = LogisticRegressionCV(multi_class="ovr", solver='liblinear',fit_intercept=True,Cs=np.logspace(-2,2,20),cv=10,penalty="l1",tol=0.01)
# re = lr.fit(train_X, train_y)
# re.score(train_X,train_y)

# skf.get_n_splits
# def f1_score_vail(labels, pred):
#     # pred > 0.5
#     return 'f1_score', f1_score(labels, pred>0.5), True

lgb_test_result = np.zeros(test_y.shape[0])
counter = 0
for train_index, test_index in skf.split(train_X, train_y):
    print('Fold {}\n'.format(counter + 1))

    # print("Train Index:", train_index, ",Test Index:", test_index)
    X_train, X_val = train_X[train_index], train_X[test_index]
    y_train, y_val = train_y[train_index], train_y[test_index]

    lgb_model = lgb.LGBMClassifier(max_depth=-1,
                                       n_estimators=30000,
                                       learning_rate=0.05,
                                       num_leaves=2**12-1,
                                       colsample_bytree=0.28,
                                       objective='binary',
                                       n_jobs=-1)

    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], eval_metric='auc',
                  verbose=100, early_stopping_rounds=100)
    del X_train, X_val, y_train, y_val, train_index, test_index
    gc.collect()

    lgb_test_result += lgb_model.predict_proba(test_X)[:, 1]
    counter += 1
print(lgb_test_result / counter)
accuracy_score(test_y, lgb_test_result / counter>0.5)
f1_score(test_y, lgb_test_result / counter>0.5)
f1_max = 0
threshold = None
for i in np.arange(0.1, 0.9, 0.05):
    tmp = f1_score(test_y, lgb_test_result / counter>i)
    if tmp > f1_max:
        f1_max  = tmp
        threshold = i
print(classification_report(test_y==1, (lgb_test_result / counter>threshold), labels=[True, False], target_names=['阳性', '阴性'], sample_weight=None, digits=2))
pass
cm = confusion_matrix(test_y==1, (lgb_test_result / counter>threshold))
plt.matshow(cm,cmap=plt.cm.Greens)
plt.colorbar()
for i, x in enumerate(cm.ravel()):
    print(x)
    plt.annotate(str(x), [i%2-0.08, i//2+0.08], size=15, color='#666666')

fpr, tpr, thresholds = roc_curve(test_y, lgb_test_result / counter)
plt.plot(fpr,tpr,marker = 'o')
