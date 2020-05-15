import dgl
import math
import torch as t
import torch.nn as nn
import dgl.function as fn
import torch.functional as F

class TreeEmbed(nn.Module):
    def __init__(self, token_size, embedding_dim):
        super(TreeEmbed, self).__init__()
        self.emb = nn.Embedding(token_size, embedding_dim)

    def forward(self, ast):
        root_tensors = self.emb(ast.ndata['token'])
        ast.ndata.update({
            'x': root_tensors
            })
        return ast





class TBCNN(nn.Module):
    def __init__(self, embedding_dim, conv_dim, max_depth=2):
        super(TBCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.conv_out = conv_dim
        self.max_depth = max_depth

        self.message_func = fn.src_mul_edge(src='h', edge='eta', out='h')
        self.reduce_func = fn.sum(msg='h', out='h')

        self.W = nn.Parameter(t.FloatTensor(embedding_dim, 3 * conv_dim))
        self.bias = nn.Parameter(t.FloatTensor(conv_dim))

        nn.init.xavier_normal_(self.W)

        stdv = 1. / math.sqrt(conv_dim)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, ast):
        ast.ndata.update({
            'h':t.matmul(ast.ndata.pop('x'), self.W).view(-1, 3, self.conv_out)
        })
        ast.update_all(self.message_func, self.reduce_func)
        conv_out = t.tanh(t.sum(ast.ndata.pop('h'), dim=1) + self.bias)
        return conv_out
