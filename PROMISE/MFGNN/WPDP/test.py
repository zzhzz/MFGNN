import os
import math
from config import EPOCH, token_size, feature_size, hidden_size, \
    label_size, heads, edge_type, train_prob, val_prob, with_df, cuda_devices
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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, classification_report
import sys

class GSANN(nn.Module):
    def __init__(self):
        super(GSANN, self).__init__()
        self.hidden_size = hidden_size
        self.encode = TreeEmbed(token_size=token_size, embedding_dim=feature_size)
        self.tbcnn = TBCNN(embedding_dim=feature_size, conv_dim=hidden_size)
        self.block_emb = BlockEmb(hidden_size)
        self.gat_in = GAT(hidden_size, hidden_size, n_edge_type=edge_type, attn_heads=heads)
        self.gat_out = GAT(hidden_size, hidden_size, n_edge_type=edge_type, attn_heads=heads)
        self.gat_in_2 = GAT(hidden_size, hidden_size, n_edge_type=edge_type, attn_heads=heads)
        self.gat_out_2 = GAT(hidden_size, hidden_size, n_edge_type=edge_type, attn_heads=heads)
        self.gat_in_3 = GAT(hidden_size, hidden_size, n_edge_type=edge_type, attn_heads=heads)
        self.gat_out_3 = GAT(hidden_size, hidden_size, n_edge_type=edge_type, attn_heads=heads)
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

        _, gat_in_2 = self.gat_in_2(g, fea)
        _, gat_out_2 = self.gat_out_2(rg, fea)
        fea_2 = gat_in_2 + gat_out_2 + fea

        _, gat_in_3 = self.gat_in_3(g, fea_2)
        _, gat_out_3 = self.gat_out_3(rg, fea_2)
        block_features = gat_in_3 + gat_out_3 + fea_2
        pooled, _ = t.max(block_features, dim=0, keepdim=True)
        logits = self.mlp(pooled)
        loss = self.loss_fn(logits.view(1), label.view(1))
        return F.sigmoid(logits), t.mean(loss)


def train(identify, train_prob, val_prob):
    model = GSANN()
    model = model.cuda()
    train_datas = load_data(train_prob)
    train_counts = len(train_datas)
    val_datas = load_data(val_prob)
    val_counts = len(val_datas)
    test_datas = val_datas
    test_counts = val_counts
    prob = train_prob + '+' + val_prob
    model_name = 'model-df' if with_df else 'model-nodf'
    model_name = model_name + '-' + str(identify) + '-adamax.pkl'
    logdir_name = '/logdir-df' if with_df else '/logdir-nodf'
    logdir_name = logdir_name + '-' + str(identify) + '-adamax/'
    train_acc = 0.0
    val_f1, best_f1 = 0.0, 0.0
    losses = []
    if os.path.exists('./' + prob + '/' + model_name):
        model.load_state_dict(t.load('./' + prob + '/' + model_name))
        test_report = f'./{prob}/{prob}-{with_df}-{identify}-test.txt'
        preds, corrs = [], []
        logits, corrsp = [], []
        model.eval()
        pbar = trange(test_counts)
        for data_id in pbar:
            data = test_datas[data_id]
            with t.no_grad():
                out, _ = model(data['ast'], data['emb'], data['cfg'], data['label'])
                pred = out.item()
                logits.append(pred)
                pred = int(round(pred))
                corr = data['label'].item()
                preds.append(pred)
                corrs.append(int(corr))
                corrsp.append(corr)
        print('Test report')
        with open(test_report, 'w') as fh:
            fh.write('P:' + str(precision_score(corrs, preds, average='binary'))+'\n')
            fh.write('R:' + str(recall_score(corrs, preds, average='binary'))+'\n')
            fh.write('F:' + str(f1_score(corrs, preds, average='binary'))+'\n')
            fh.write('Auc:' + str(roc_auc_score(corrsp, logits))+'\n')
            fh.write('AUC:' + str(roc_auc_score(corrsp, logits)))
            # fh.write(str(classification_report(corrs, preds, target_names=labels, digits=3))+'\n')
            fh.write(str(classification_report(corrs, preds, target_names=labels, digits=3)))
    else:
        raise e


if __name__ == "__main__":
    for exp in os.listdir('.'):
        if exp.find('+') != -1:
            exp_path = './' + exp + '/'
            models = list(filter(lambda x: x.startswith('model'), os.listdir(exp_path)))
            train_proj, test_proj = exp.split('+')
            identifies = list(map(lambda x: int(x.split('-')[2]), models))
            identifies = sorted(identifies) 
            hiddens = [60, 200]
            global hidden_size
            for idx in range(2):
                hidden_size = hiddens[idx]
                train(identifies[idx], train_proj, test_proj)            


