import dgl
import torch as th
import torch.nn as nn
import dgl.function as F

class BlockEmb(nn.Module):
    def __init__(self, conv_out):
        self.conv_out = conv_out
        super(BlockEmb, self).__init__()
        self.message_func = F.copy_edge(edge='x', out='m')
        self.reduce_func = F.max(msg='m', out='x')

    def forward(self, g):
        g.update_all(self.message_func, self.reduce_func)
        g.edata.pop('x')
        return g.ndata.pop('x')
