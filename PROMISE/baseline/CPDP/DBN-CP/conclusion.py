import os
import numpy as np
l = []
d = {}
for f in os.listdir('./'):
    f_path = './' + f
    if f.endswith('.txt') and f not in ['names.txt', 'out.txt']:
        with open(f_path, 'r') as fh:
            lis = list(fh)
            auc = round(100.0*float(lis[0].replace('\n', '')), 1)
            line = lis[4].replace('\n', '') 
            items = line.split(' ')
            items = list(filter(lambda x: len(x) >0, items))
            f1 = round(100.0*float(items[3]), 1)
            l.append((f1, auc)) 
            f = f.replace('.txt', '')
            d[f] = [str(f1), str(auc)]


print(np.mean(np.array(l).reshape((-1, 2)), axis=0))

"""
for r in l:
    print(r)
"""
