import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import dgl


class GAT(nn.Module):
    def __init__(self, F, F_out, n_edge_type, attn_heads, heads_reduction='average'):
        super(GAT, self).__init__()
        self.n_edge_type = n_edge_type
        self.F = F
        self.F_out = F_out
        self.attn_heads = attn_heads
        self.reduction = heads_reduction
        self.fc = nn.Linear(F, attn_heads * F_out, bias=False)
        self.attn_l = nn.Parameter(t.FloatTensor(1, F_out, attn_heads))
        self.attn_r = nn.Embedding(n_edge_type, 1 * F_out * attn_heads)

        self.softmax = edge_softmax
        self.calc_e_fea = fn.src_mul_edge(src='x', edge='kernel', out='h')
        self.edge_attention = fn.v_add_e('a1', 'a2', out='a')
        self.bias = nn.Parameter(t.FloatTensor(attn_heads * F_out))

        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_r.weight)
        self.bias.data.uniform_(-0.02, 0.02)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = edge_softmax

        self.message = fn.src_mul_edge(src='x', edge='a', out='h')
        self.reduce = fn.sum(msg='h', out='h')

    def forward(self, g, x):
        n_fea = self.fc(x).view(-1, self.F_out, self.attn_heads)

        a1 = (n_fea * self.attn_l).sum(dim=1).view(-1, self.attn_heads, 1)

        type_attention = self.attn_r(g.edata['type']).view(-1, self.F_out, self.attn_heads)

        g.ndata.update({
            'x': n_fea
        })

        g.edata.update({
            'kernel': type_attention
        })
        g.apply_edges(self.calc_e_fea)
        a2 = g.edata.pop('h').sum(dim=1).view(-1, self.attn_heads, 1)
        g.edata.pop('kernel')

        g.ndata.update({
            'a1': a1
        })
        g.edata.update({
            'a2': a2,
        })

        g.apply_edges(self.edge_attention)
        a = self.leaky_relu(g.edata.pop('a'))
        g.ndata.pop('a1')
        g.edata.pop('a2')
        attention = self.softmax(g, a).view(-1, 1, self.attn_heads)
        
        g.edata.update({
            'a': attention
        })
        g.update_all(self.message, self.reduce)
        g.edata.pop('a')
        g.ndata.pop('x')
        ret = t.mean((g.ndata.pop('h').view(-1, self.attn_heads * self.F_out) + self.bias).view(-1, self.attn_heads, self.F_out), dim=1)
        return attention.detach(), F.elu(ret)
