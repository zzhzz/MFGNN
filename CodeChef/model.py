import os, sys
import math
from config import EPOCH, token_size, feature_size, hidden_size, \
    label_size, heads, edge_type, prob, with_df, cuda_devices, test_mode
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
from sklearn.metrics import accuracy_score, classification_report, f1_score

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
        g = cfg
        rg = cfg.reverse(share_ndata=True, share_edata=True)
        _, gat_in = self.gat_in(g, block_features)
        _, gat_out = self.gat_out(rg, block_features)
        fea = gat_in + gat_out + block_features
        _, gat_in_2 = self.gat_in_2(g, fea)
        _, gat_out_2 = self.gat_out_2(g, fea)
        fea_2 = gat_in_2 + gat_out_2 + fea
        _, gat_in_3 = self.gat_in_3(g, fea_2)
        _, gat_out_3 = self.gat_out_3(g, fea_2)

        block_features = gat_in_3 + gat_out_3 + fea_2
        pooled,_ = t.max(block_features, dim=0, keepdim=True)
        logits = self.mlp(pooled)
        loss = self.loss_fn(logits.view(1, label_size), label.view(1))
        return logits, t.mean(loss)


def train():
    model = GSANN()
    if test_mode:
        identify = sys.argv[1]
    else:
        identify = int(time.time())
    model_name = 'model-df' if with_df else 'model-nodf'
    model_name = model_name + '-' + str(identify) + '-adamax.pkl'
    logdir_name = '/logdir-df' if with_df else '/logdir-nodf'
    logdir_name = logdir_name + '-' + str(identify) + '-adamax/'
    if test_mode:
        model.load_state_dict(t.load('./' + prob + '/' + model_name, map_location=t.device('cpu')), strict=False)
        quit()
    model = model.cuda()
    datas = load_data(prob)
    random.seed(0)
    random.shuffle(datas)
    train_counts = int(len(datas)*.6)
    train_datas = datas[:train_counts]
    val_datas = datas[train_counts:int(len(datas)*.8)]
    val_counts = len(val_datas)
    test_datas = datas[train_counts+val_counts:]
    optim = t.optim.Adamax(model.parameters(), lr=0.001)
    train_acc = 0.0
    val_f1, best_f1 = 0.0, 0.0
    losses = []
    writer = SummaryWriter(log_dir='./' + prob + logdir_name)
    try:
        for epoch in range(EPOCH):
            print('Epoch ', epoch)
            # mail.send('[EPOCH]', f'EPOCH {epoch}')
            random.shuffle(train_datas)
            accs = []
            loss_per_epoch = []
            pbar = trange(train_counts)
            preds, corrs = [], []
            for data_id in pbar:
                data = train_datas[data_id]
                out, loss = model(data['ast'], data['emb'], data['cfg'], data['label'])
                pred = t.argmax(out).item()
                corr = data['label'].item()
                preds.append(pred)
                corrs.append(corr)
                loss_per_epoch.append(float(loss.item()))
                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.set_description('Loss %.5lf Train acc %.3lf, best f1 %.3lf, val f1 %.3lf'%(float(loss.item()), train_acc, best_f1, val_f1))
            train_acc = accuracy_score(corrs, preds)
            loss_mean = np.mean(loss_per_epoch)
            writer.add_scalar(tag='train loss', scalar_value=loss_mean, global_step=epoch)
            writer.add_scalar(tag='train acc', scalar_value=train_acc, global_step=epoch)
            losses.append(np.mean(loss_per_epoch))
            accs = []
            print('Validation')
            pbar = trange(val_counts)
            preds, corrs = [], []
            for data_id in pbar:
                data = val_datas[data_id]
                with t.no_grad():
                    out, _ = model(data['ast'], data['emb'], data['cfg'], data['label'])
                    pred = t.argmax(out).item()
                    corr = data['label'].item()
                    preds.append(pred)
                    corrs.append(corr)

            val_acc = accuracy_score(corrs, preds)
            val_f1 = f1_score(corrs, preds, average='macro')
            writer.add_scalar(tag='test acc', scalar_value=val_acc, global_step=epoch)
            if val_f1 > best_f1 :
                best_f1 = val_f1
                t.save(model.state_dict(), './' + prob + "/" + model_name)
        raise KeyboardInterrupt
    except KeyboardInterrupt as e:
        if os.path.exists('./' + prob + '/' + model_name):
            model.load_state_dict(t.load('./' + prob + '/' + model_name))
            train_report = f'./{prob}/{prob}-{with_df}-{identify}-train.txt'
            test_report = f'./{prob}/{prob}-{with_df}-{identify}-test.txt'
            preds, corrs = [], []
            pbar = trange(train_counts)
            model.eval()
            for data_id in pbar:
                data = train_datas[data_id]
                with t.no_grad():
                    out, loss = model(data['ast'], data['emb'], data['cfg'], data['label'])
                    pred = t.argmax(out).item()
                    corr = data['label'].item()
                    preds.append(int(pred))
                    corrs.append(int(corr))
            print('Train report')
            with open(train_report, 'w') as fh:
                fh.write(str(accuracy_score(corrs, preds))+'\n')
                fh.write(str(classification_report(corrs, preds, target_names=labels, digits=3))+'\n')
                print(accuracy_score(corrs, preds))
                print(classification_report(corrs, preds, target_names=labels, digits=3))
            preds, corrs = [], []
            pbar = trange(val_counts)
            for data_id in pbar:
                data = test_datas[data_id]
                with t.no_grad():
                    out, _ = model(data['ast'], data['emb'], data['cfg'], data['label'])
                    pred = t.argmax(out).item()
                    corr = data['label'].item()
                    preds.append(int(pred))
                    corrs.append(int(corr))
            print('Test report')
            with open(test_report, 'w') as fh:
                fh.write(str(accuracy_score(corrs, preds))+'\n')
                fh.write(str(classification_report(corrs, preds, target_names=labels, digits=3))+'\n')
                print(accuracy_score(corrs, preds))
                print(classification_report(corrs, preds, target_names=labels, digits=3))
        else:
            raise e


if __name__ == "__main__":
    train()


