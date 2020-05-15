import json, sys, csv
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from tl_algs import tnb

train_proj = sys.argv[1]
test_proj = sys.argv[2]

with open('claz_list.json', 'r') as rfh:
    proj_list = json.load(rfh)

train_list = proj_list[train_proj]
test_list = proj_list[test_proj]

train_csv = '../../promise_data/csv/' + train_proj + '.csv'
test_csv = '../../promise_data/csv/' + test_proj + '.csv'


def load(csv_path, checklist):
    X = []
    y = []
    with open(csv_path, 'r') as rfh:
        f_csv = csv.reader(rfh)
        f_csv = list(f_csv)[1:]
        for row in f_csv:
            if row[2] in checklist:
                X.append(list(map(float, row[3:-1])))
                y.append(int(row[-1]) > 0)
    return pd.DataFrame(X), pd.Series(y)

train_X, train_y = load(train_csv, train_list)
test_X, test_y = load(test_csv, test_list)

w = tnb.TransferNaiveBayes(test_set_X=test_X,
        test_set_domain='a',
        train_pool_X=train_X,
        train_pool_y=train_y,
        train_pool_domain=None,
        rand_seed=None,
        similarity_func=tnb.sim_minmax,
        discretize=False)

prob, preds = w.train_filter_test()
corrs = test_y

corr_prob = list(map(float, test_y))

with open('./tnb_result/' + train_proj + '+' + test_proj + '-tnb.txt', 'w') as wfh:
    wfh.write(str(roc_auc_score(corr_prob, prob)) + '\n')
    wfh.write(str(classification_report(corrs, preds)) + '\n')
print('Done')


if __name__ == '__main__':
    pass


