import os, shutil
import numpy as np

l = []
datas = {}
table_col = []

with open('seq.txt', 'r') as rfh:
    for line in rfh:
        line = line.replace('\n', '')
        table_col.append(line.split(' ')) 

for f in os.listdir('./'):
    f_path = './' + f
    if os.path.isdir(f_path) and f != '__pycache__':
        items = []
        for result_f in os.listdir(f_path):
            if result_f.find('wpdp-80') != -1:
                if result_f.find('.txt') != -1:
                    with open(f_path + '/' + result_f, 'r') as rfh:
                        item = []
                        for line in rfh:
                            line = line.replace('\n', '')
                            item.append(float(line.split(':')[1]))
                        items.append(item[2:])
        item = [0, 0]
        for d in items:
            if d[0] > item[0]:
                item = d
        item = list(map(lambda x: round(x*100.0, 1), item))
        l.append(item)
        datas[f] = item

print(np.mean(np.array(l).reshape((-1, 2)), axis=0))

for idx, key in enumerate(table_col):
    k = '+'.join(key)
    if idx % 2 == 0:
        print('\\midrule')
    s = key[0] + ' & ' + key[1] + ' & & & ' + str(datas[k][0]) + ' & ' + str(datas[k][1]) + '\\\\'
    print(s)

