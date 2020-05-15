import os
import math
from config import EPOCH, token_size, feature_size, hidden_size, train_proj, test_proj,\
    label_size, cuda_devices, with_df
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
import torch as t
import time
import numpy as np
import random
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tree_lstm import TreeLSTM
from datas import load_data, labels
from tqdm import trange
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, classification_report



def train():
    model = TreeLSTM(token_size, feature_size, hidden_size, label_size, 0.5, cell_type='sum')
    model = model.cuda()
    train_datas = load_data(train_proj)
    train_counts = len(train_datas)
    test_datas = load_data(test_proj)
    test_counts = len(test_datas)

    identify = 'wpdp'
    prob = train_proj + '+' + test_proj
    model_name = 'model-df' if with_df else 'model-nodf'
    model_name = model_name + '-' + str(identify) + '.pkl'
    logdir_name = '/logdir'
    logdir_name = logdir_name + '-' + str(identify) + '/'

    optim = t.optim.Adamax(model.parameters(), lr=0.001)
    train_acc = 0.0
    val_f1, best_f1 = 0.0, 0.0
    losses = []
    writer = SummaryWriter(log_dir='./' + prob + logdir_name)
    best_model = None
    try:
        for epoch in range(EPOCH):
            print('Epoch ', epoch)
            random.shuffle(train_datas)
            pbar = trange(train_counts)
            preds, corrs = [], []
            for data_id in pbar:
                data = train_datas[data_id]
                g = data['ast']
                n = g.number_of_nodes()
                h = t.zeros((n, hidden_size)).cuda()
                c = t.zeros((n, hidden_size)).cuda()
                _, loss = model(g, h, c, data['label'])
                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.set_description('Loss %.5lf best f1 %.3lf, val f1 %.3lf'%(float(loss.item()), best_f1, val_f1))
            """
            print('Validation')
            pbar = trange(val_counts)
            preds, corrs = [], []
            for data_id in pbar:
                data = val_datas[data_id]
                with t.no_grad():
                    g = data['ast']
                    n = g.number_of_nodes()
                    h = t.zeros((n, hidden_size)).cuda()
                    c = t.zeros((n, hidden_size)).cuda()
                    out, _ = model(g, h, c, data['label'])
                    pred = int(round(out.item()))
                    corr = data['label'].item()
                    preds.append(pred)
                    corrs.append(corr)

            val_acc = accuracy_score(corrs, preds)
            val_f1 = f1_score(corrs, preds, average='binary')
            writer.add_scalar(tag='test acc', scalar_value=val_acc, global_step=epoch)
            if val_f1 > best_f1 :
                best_f1 = val_f1
                best_model = model
                t.save(model.state_dict(), './' + prob + "/" + model_name)
            """
        t.save(model.state_dict(), './' + prob + "/" + model_name)
        raise KeyboardInterrupt
    except KeyboardInterrupt as e:
        if os.path.exists('./' + prob + '/' + model_name):
            model.load_state_dict(t.load('./' + prob + '/' + model_name))
            test_report = f'./{prob}/{prob}-{identify}-test.txt'
            model.eval()
            preds, corrs = [], []
            corrsp, logits  = [], []
            pbar = trange(test_counts)
            for data_id in pbar:
                data = test_datas[data_id]
                with t.no_grad():
                    g = data['ast']
                    n = g.number_of_nodes()
                    h = t.zeros((n, hidden_size)).cuda()
                    c = t.zeros((n, hidden_size)).cuda()
                    out, _ = model(g, h, c, data['label'])
                    logits.append(out.item())
                    pred = int(round(out.item()))
                    corr = data['label'].item()
                    preds.append(int(pred))
                    corrsp.append(float(corr))
                    corrs.append(int(corr))
            print('Test report')
            with open(test_report, 'w') as fh:
                fh.write('P:' + str(precision_score(corrs, preds, average='binary'))+'\n')
                fh.write('R:' + str(recall_score(corrs, preds, average='binary'))+'\n')
                fh.write('F:' + str(f1_score(corrs, preds, average='binary'))+'\n')
                fh.write('Auc:' + str(roc_auc_score(corrsp, logits))+'\n')
                print('AUC:' + str(roc_auc_score(corrsp, logits)))
        else:
            raise e


if __name__ == "__main__":
    train()


