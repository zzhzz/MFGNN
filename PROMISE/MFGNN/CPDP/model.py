import os
import math
from config import EPOCH, token_size, feature_size, hidden_size, \
    label_size, heads, edge_type, train_prob, test_prob, with_df, cuda_devices, test_mode
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
import torch as t
import time
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tree_conv import TBCNN, TreeEmbed
from block_emb import BlockEmb
from gat import GAT
from datas import load_data, labels
from tqdm import trange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, classification_report

class GSANN(nn.Module):
    def __init__(self):
        super(GSANN, self).__init__()
        self.hidden_size = hidden_size
        self.encode = TreeEmbed(token_size=token_size, embedding_dim=feature_size)
        self.tbcnn = TBCNN(embedding_dim=feature_size, conv_dim=hidden_size)
        self.block_emb = BlockEmb(hidden_size)
        self.gat_in = GAT(hidden_size, hidden_size, n_edge_type=edge_type, attn_heads=heads)
        self.gat_out = GAT(hidden_size, hidden_size, n_edge_type=edge_type, attn_heads=heads)
        self.mlp = nn.Linear(hidden_size, label_size)
        self.loss_fn = nn.BCEWithLogitsLoss()
    

    def forward(self, ast, emb, cfg, label):
        ast = self.encode(ast)
        conv_out = self.tbcnn(ast)
        n_blocks = cfg.number_of_nodes()
        emb.edata.update({
            'x': conv_out
        })
        features = self.block_emb(emb)
        block_features = features.narrow(0, 0, n_blocks)
        del features, conv_out
        g = cfg
        rg = g.reverse(share_ndata=True, share_edata=True)
        _, gat_in = self.gat_in(g, block_features)
        _, gat_out = self.gat_out(rg, block_features)
        fea = gat_in + gat_out + block_features

        pooled,_ = t.max(fea, dim=0, keepdim=True)
        logits = self.mlp(pooled)
        loss = self.loss_fn(logits.view(1), label.view(1))
        return F.sigmoid(logits), t.mean(loss), pooled

def train():
    model = GSANN()
    model = model.cuda()
    train_datas = load_data(train_prob)
    train_counts = len(train_datas)
    test_datas = load_data(test_prob)
    test_counts = len(test_datas)
    random.shuffle(test_datas)
    val_datas = test_datas[:int(test_counts*.3)]
    val_counts = len(val_datas)
    test_datas = test_datas[val_counts:]
    test_counts = len(test_datas)

    identify = 'transfer'
    prob = train_prob + '+' + test_prob
    model_name = 'model-df' if with_df else 'model-nodf'
    model_name = model_name + '-' + str(identify) + '-adamax.pkl'
    logdir_name = '/logdir-df' if with_df else '/logdir-nodf'
    logdir_name = logdir_name + '-' + str(identify) + '-adamax/'
    optim = t.optim.Adamax(model.parameters(), lr=0.001)
    train_acc = 0.0
    val_f1, best_f1 = 0.0, 0.0
    losses = []
    writer = SummaryWriter(log_dir='./' + prob + logdir_name)
    try:
        if test_mode:
            raise KeyboardInterrupt
        train_epoch = EPOCH
        for epoch in range(train_epoch):
            print('Epoch ', epoch)
            random.shuffle(train_datas)
            accs = []
            loss_per_epoch = []
            pbar = trange(train_counts)
            for data_id in pbar:
                data = train_datas[data_id]
                out, loss, _ = model(data['ast'], data['emb'], data['cfg'], data['label'])
                optim.zero_grad()
                loss.backward()
                optim.step()
        t.save(model.state_dict(), './' + prob + "/" + model_name)
        print('Transfer')
        pbar = trange(val_counts)
        X, y = [], []
        for data_id in pbar:
            with t.no_grad():
                data = val_datas[data_id]
                _, _, x = model(data['ast'], data['emb'], data['cfg'], data['label'])
                X.append(x.detach().cpu().numpy().reshape((-1,)))
                y.append(data['label'].cpu().numpy().reshape((-1,)))
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga', verbose=0)
        clf.fit(X, y)
        t.save(model.state_dict(), './' + prob + "/" + model_name)
        raise KeyboardInterrupt
    except KeyboardInterrupt as e:
        if os.path.exists('./' + prob + '/' + model_name):
            model.load_state_dict(t.load('./' + prob + '/' + model_name))
            test_report = f'./{prob}/{prob}-{with_df}-{identify}-test.txt'
            preds, corrs = [], []
            logits, corrsp = [], []
            model.eval()
            pbar = trange(test_counts)
            X, y = [], []
            for data_id in pbar:
                data = test_datas[data_id]
                with t.no_grad():
                    out, _, x = model(data['ast'], data['emb'], data['cfg'], data['label'])
                    X.append(x.detach().cpu().numpy().reshape((-1,)))
                    y.append(data['label'].cpu().numpy().reshape((-1)))
            preds = clf.predict(X)
            corrs = list(map(int, y))
            logits = clf.predict_log_proba(X)
            logits = np.array(logits)[:, 1]
            corrsp = list(map(float, y))
            print('Test report')
            with open(test_report, 'w') as fh:
                fh.write('P:' + str(precision_score(corrs, preds, average='binary'))+'\n')
                fh.write('R:' + str(recall_score(corrs, preds, average='binary'))+'\n')
                fh.write('F:' + str(f1_score(corrs, preds, average='binary'))+'\n')
                fh.write('Auc:' + str(roc_auc_score(corrsp, logits))+'\n')
                print('AUC:' + str(roc_auc_score(corrsp, logits)))
                print(str(classification_report(corrs, preds, target_names=labels, digits=3)))
        else:
            raise e


if __name__ == "__main__":
    train()


