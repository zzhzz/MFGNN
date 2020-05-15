from functools import reduce
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from libtlda.tca import TransferComponentClassifier
from imblearn.over_sampling import SMOTE
from collections import namedtuple
import json
import csv
import sys

NoN, N1, N2, N3, N4 = 0, 1, 2, 3, 4

n1_scale = MinMaxScaler()
n2_scale = StandardScaler()
n3_scale = StandardScaler()
n4_scale = StandardScaler()

norm_func = [
    lambda x: x,
    lambda x: n1_scale.transform(x),
    lambda x: n2_scale.transform(x),
    lambda x: n3_scale.transform(x),
    lambda x: n4_scale.transform(x)
]

DCV = namedtuple('DCV', 'mean median min max std num')

train_proj = sys.argv[1]
test_proj = sys.argv[2]

with open('claz_list.json', 'r') as rfh:
    proj_list = json.load(rfh)

train_list = proj_list[train_proj]
test_list = proj_list[test_proj]

path = '../../promise_data/csv/'
train_csv = path + train_proj + '.csv'
test_csv = path + test_proj + '.csv'

def load(csv_path, checklist):
    X = []
    y = []
    with open(csv_path, 'r') as rfh:
        f_csv = csv.reader(rfh)
        f_csv = list(f_csv)[1:]
        for row in f_csv:
            if row[2] in checklist:
                X.append(list(map(float, row[3:-1])))
                y.append(int(int(row[-1]) > 0))
    n1_scale.partial_fit(X)
    n2_scale.partial_fit(X)
    return X, y

MM = lambda cs, ct: cs * 1.6 < ct
M = lambda cs, ct: cs * 1.3 < ct <= cs * 1.6
SM = lambda cs, ct: cs * 1.1 < ct <= cs * 1.3
S = lambda cs, ct: cs * .9 < ct <= cs * 1.1
SL = lambda cs, ct: cs * .7 < ct <= cs * .9
L = lambda cs, ct: cs * .4 < ct <= cs * .7
ML = lambda cs, ct: ct < cs * .4

cmp = lambda f, attrs, s, t: reduce(lambda x,y: x and y,
                map(lambda n: f(s[n], t[n]), attrs))

attrs_list = [['mean', 'std'], ['min', 'max', 'num'], ['std'], ['std', 'num']]

RULE1 = lambda s, t: cmp(S, attrs_list[0], s, t)
RULE2 = lambda s, t: cmp(MM, attrs_list[1], s, t) or cmp(ML, attrs_list[1], s, t)
RULE3 = lambda s, t: (cmp(MM, attrs_list[2], s, t) and (s['num'] > t['num'])) or (cmp(ML, attrs_list[2], s, t) and (s['num'] < t['num']))
RULE4 = lambda s, t: cmp(MM, attrs_list[2], s, t) or cmp(ML, attrs_list[2], s, t)


def norm(train_X, test_X, s, t):
    if RULE1(s, t):
        f = norm_func[NoN]
    elif RULE2(s, t):
        f = norm_func[N1]
    elif RULE3(s, t):
        f = norm_func[N3]
    elif RULE4(s, t):
        f = norm_func[N4]
    else:
        f = norm_func[N2]
    return f(np.array(train_X)), f(np.array(test_X))

train_X, train_y = load(train_csv, train_list)
n3_scale.fit(train_X)
test_X, test_y = load(test_csv, test_list)
n4_scale.fit(test_X)


def get_DCV(X):
    DIST = []
    for i, a in enumerate(X):
        for b in X[i+1:]:
            dist = np.linalg.norm(np.array(a) - np.array(b))
            DIST.append(dist)
    DIST = np.array(DIST).reshape((-1,))
    dist_mean = np.mean(DIST, axis=0)
    dist_median = np.median(DIST, axis=0)
    dist_min = np.min(DIST, axis=0)
    dist_max = np.max(DIST, axis=0)
    dist_std = np.std(DIST, axis=0)
    num = len(X)
    return DCV(mean=dist_mean, median=dist_median,
               min=dist_min, max=dist_max, std=dist_std, num=num)._asdict()

DCV_Tr = get_DCV(train_X)
DCV_T = get_DCV(test_X)
train_X, test_X = norm(train_X, test_X, DCV_Tr, DCV_T)
classifier = TransferComponentClassifier()
sm = SMOTE()
X, y = sm.fit_resample(train_X, train_y)
classifier.fit(X, y, test_X)
preds, prob = classifier.predict(test_X)
prob = np.array(prob).reshape((-1, 2))[:, 1]
corrs = test_y
with open(train_proj + '+' + test_proj + '-tca+.txt', 'w') as wfh:
    wfh.write(str(roc_auc_score(corrs, prob))+'\n')
    wfh.write(str(classification_report(corrs, preds)) + '\n')


if __name__ == '__main__':
    pass

