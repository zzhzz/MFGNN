import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import json
import math
import random
import numpy as np
lens_map = {}
with open('temp.txt', 'r') as rfh:
    for line in rfh:
        line = line.replace('\n', '')
        _, proj_name, sz = line.split(' ')
        sz = int(sz)
        lens_map[proj_name] = sz 

proj = sys.argv[1]
train_ver = sys.argv[2]
test_ver = sys.argv[3]
token_size = 44213
lens = max(lens_map[proj+'-'+train_ver], lens_map[proj+'-'+test_ver])

def load(proj, ver):
    with open('../../promise_data/dbn/wpdp/' + proj +'-' + ver +'_padseq.json', 'r') as fh:
        datas = json.load(fh) 
    return datas

cnt = {}
mi, ma = 1e9, 0
def pre(datas):
    global mi, ma
    for item in datas:
        x, y = item
        orig = list(filter(lambda x: x > 0, x))
        mi = min(mi, min(orig))
        ma = max(ma, max(orig))
        for v in x:
            if v > 0:
                if v not in cnt.keys():
                    cnt[v] = 0
                cnt[v] += 1

def gen(datas):
    X, Y = [], []
    for item in datas:
        x, y = item
        x = list(filter(lambda x: x > 0 and cnt[x] >= 3, x))
        x = list(map(lambda x: float(x - mi + 1) / float(ma - mi + 1), x))    
        x += [0.0] * (lens - len(x))
        X.append(x)
        Y.append(1 if int(y) > 0 else 0)
    return X, Y

lis = [('rbm'+str(i), BernoulliRBM(n_components=100, learning_rate=1e-2, batch_size=10, n_iter=400)) for i in range(10)]
lis.append(('classifier', GaussianNB()))
# lis.append(('classifier', AdaBoostClassifier(n_estimators=100)))
# lis.append(('classifier', GradientBoostingClassifier(n_estimators=100, max_depth=1, learning_rate=1e-3)))

model = Pipeline(steps=lis)

def train():
    train_datas = load(proj, train_ver)
    random.shuffle(train_datas)
    train_count = len(train_datas)
    test_datas = load(proj, test_ver)
    test_count = len(test_datas)
    pre(train_datas)
    pre(test_datas)
    train_datas, train_labels = gen(train_datas)
    test_datas, test_labels = gen(test_datas)
    print('Start')
    X, y = train_datas, train_labels
    sm = SMOTE()
    X, y = sm.fit_resample(X, y)
    print(len(X), sum(y))
    model.fit(X, y)
    X_test, corrs = test_datas, test_labels
    preds = model.predict(X_test)
    scores = model.predict_log_proba(X_test)
    scores = scores[:, 1] 
    with open(proj+train_ver+test_ver+'-bnb.txt', 'w') as wfh:
        print(str(roc_auc_score(list(map(float, corrs)), scores)) + '\n')
        print(str(classification_report(corrs, preds, target_names=['0', '1'], digits=3)) + '\n')
        wfh.write(str(roc_auc_score(list(map(float, corrs)), scores)) + '\n')
        wfh.write(str(classification_report(corrs, preds, target_names=['0', '1'], digits=3)) + '\n')

if __name__ == '__main__':
    train()

