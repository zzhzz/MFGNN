import os

items = []

with open('seq.txt', 'r') as rfh:
    for line in rfh:
        items.append(line.replace('\n', '')) 

start = True
for line in items:
    if start:
        cmd = 'python model.py ' + line
        r = os.system(cmd)
        if r != 0:
            print(cmd)
            raise ValueError
