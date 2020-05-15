import json
import numpy as np
import torch as th
import dgl
from config import edge_type as n_edge_type, with_df, data_dir
from tqdm import tqdm
import random

labels = ['0', '1']

def load_data():
    global labels
    path = data_dir
    with open(path, 'r') as fh:
        datas = json.load(fh)
    data_list = []
    cnt = [0 for _ in labels]
    for data in tqdm(datas):
        n_block = data['n_blocks']
        cfg_edges = data['graph']
        # cfg_emb = data['cfg_emb_block']
        cfg_emb = data['mapping']
        ast_edges = data['ast_edges']
        ast_nodes = data['ast_nodes']
        ast_depth = data['ast_depth']
        name = data['name']
        if n_block <= 2:
            continue
        if len(ast_nodes) > 2000000:
            print(len(ast_nodes))
            continue
        label = data['label']
        label = int(labels.index(label))
        ast = dgl.DGLGraph()
        ast_edges = np.array(ast_edges)
        ast.add_nodes(len(ast_nodes))
        ast.ndata['token'] = th.LongTensor(th.from_numpy(np.array(ast_nodes))).cuda()
        ast.add_edges(ast_edges[:, 1], ast_edges[:, 0])
        ast_g = [[] for _ in range(len(ast_nodes))]
        ast_max_depth = float(max(ast_depth))
        for eid, edge in enumerate(ast_edges):
            u, v = edge
            ast_g[u].append((v, eid))
        etas = [[] for _ in range(len(ast_edges))]
        for u in range(len(ast_nodes)):
            c = len(ast_g[u])
            for i, item in enumerate(ast_g[u]):
                v, eid = item
                eta_t = th.FloatTensor([float(ast_depth[v]) / float(ast_max_depth)])
                if c > 1:
                    eta_l = th.FloatTensor([(float(i) / c) * eta_t])
                    eta_r = th.FloatTensor([(float(c) - float(i)) / float(c) * eta_t])
                elif c == 1:
                    eta_l = th.FloatTensor([0.5 * eta_t])
                    eta_r = th.FloatTensor([0.5 * eta_t])
                etas[eid] = th.cat([eta_l, eta_r, eta_t]).view(1, 3, 1)
        ast.edata['eta'] = th.cat(etas, dim=0).cuda()

        emb_graph = dgl.DGLGraph()
        cfg_emb = np.array(cfg_emb)
        emb_graph.add_nodes(len(ast_nodes) + n_block)
        emb_graph.add_edges(cfg_emb[:, 1], cfg_emb[:, 0])
        cfg_graph = dgl.DGLGraph()
        cfg_graph.add_nodes(n_block)
        df_edge_list = []
        cfg_edge_list = []
        for edge in cfg_edges:
            u, v, t = edge
            if t == 3:
                df_edge_list.append(edge)
            else:
                cfg_edge_list.append(edge)
        if with_df:
            cfg_edges = np.array(cfg_edges)
        else:
            cfg_edges = np.array(cfg_edge_list)
        
        cfg_graph.add_edges(cfg_edges[:, 1], cfg_edges[:, 0])
        edge_types = np.array(cfg_edges[:, 2]).astype(np.int64)
        cfg_graph.edata['type'] = th.LongTensor(th.from_numpy(edge_types)).cuda()
        ast.to(th.device('cuda:0'))
        cfg_graph.to(th.device('cuda:0'))
        emb_graph.to(th.device('cuda:0'))
        data_list.append({
            'name': name,
            'ast': ast,
            'cfg': cfg_graph,
            'emb': emb_graph,
            'label': th.FloatTensor([int(label)]).cuda()
        })
        cnt[int(label)] += 1
    print(cnt)
    del datas
    return data_list


