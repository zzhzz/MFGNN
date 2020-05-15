import os

items = []

with open('names.txt', 'r') as rfh:
    for line in rfh:
        items.append(line.replace('\n', '')) 

for line in items:
    cmd = 'python model_trans_max.py ' + line
    r = os.system(cmd)
    if r != 0:
        print(cmd)
        raise ValueError
