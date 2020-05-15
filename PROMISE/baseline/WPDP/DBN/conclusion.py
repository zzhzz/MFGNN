import os
import numpy as np
l = []
for f in os.listdir('./'):
    f_path = './' + f
    if f.endswith('-bnb.txt'):
        with open(f_path, 'r') as fh:
            lis = list(fh)
            auc = float(lis[0].replace('\n', ''))
            line = lis[4].replace('\n', '') 
            items = line.split(' ')
            items = list(filter(lambda x: len(x) >0, items))
            f1 = float(items[3])
            l.append((f1, auc)) 
print(np.mean(np.array(l).reshape((-1, 2)), axis=0))
print(l)
