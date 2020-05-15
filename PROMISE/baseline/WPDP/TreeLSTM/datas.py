import json
import numpy as np
import torch as th
import dgl
from tqdm import tqdm
import random

labels = [0, 1]
def load_data(proj):
    path = '../../promise_data/ast/' + proj + '.json'
            
    with open(path, 'r') as fh:
        datas = json.load(fh)
    data_list = []
    for data in tqdm(datas):
        ast_edges = data['edges']
        ast_nodes = data['labels']
        label = data['label']
        label = int(int(label) > 0)
        ast = dgl.DGLGraph()
        ast_edges = np.array(ast_edges)
        ast.add_nodes(len(ast_nodes))
        ast.ndata['token'] = th.LongTensor(th.from_numpy(np.array(ast_nodes))).cuda()
        ast.add_edges(ast_edges[:, 1], ast_edges[:, 0])
        data_list.append({
            'ast': ast,
            'label': th.FloatTensor([float(label)]).cuda()
        })
    print(len(data_list))
    return data_list


