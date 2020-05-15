import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import dgl
import torch as t
import numpy as np
import json
import random
import torch.functional as F
import torch.nn as nn
import torch.nn.functional as Fn
from tree_conv import TBCNN, TreeEmbed
from block_emb import BlockEmb
from gat import GAT
from config import EPOCH, token_size, feature_size, hidden_size, label_size, heads, edge_type, prob
from sklearn.metrics import classification_report, accuracy_score
from datas import load_data, labels
from tqdm import tqdm, trange



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
        self.loss_fn = nn.CrossEntropyLoss()

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
        gat_in = self.gat_in(cfg, block_features)
        gat_out = self.gat_out(cfg.reverse(share_ndata=True, share_edata=True), block_features)
        block_features = block_features + gat_in + gat_out
        pooled, _ = t.max(block_features, dim=0, keepdim=True)
        logits = self.mlp(pooled)
        loss = self.loss_fn(logits.view(1, label_size), label.view(1))
        return logits, t.mean(loss)


def train():
    save_dir = prob + '/model-df-1580004791.pkl'
    model = GSANN()
    model = model.cuda()
    model.load_state_dict(t.load(save_dir)) 
    model.eval()
    datas = load_data(prob)
    random.seed(0)
    random.shuffle(datas)
    train_counts = int(len(datas)*.8)
    train_datas = datas[:train_counts]
    test_datas = datas[train_counts:]
    test_counts = len(val_datas)
    preds, corrs = [], []
    print('Skip train')
    preds, corrs = [], []
    pbar = trange(test_count)
    for data_id in pbar:
        data = test_datas[data_id]
        try:
            with t.no_grad():
                out, _ = model(data['ast'], data['emb'], data['cfg'], data['label'])
                pred = t.argmax(out).item()
                corr = data['label'].item()
                preds.append(int(pred))
                corrs.append(int(corr))
        except Exception as e:
            t.cuda.empty_cache()
            continue
    print('Test report')
    print(accuracy_score(corrs, preds))
    print(classification_report(corrs, preds, target_names=labels, digits=3))

if __name__ == "__main__":
    train()


