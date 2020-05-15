import os

proj = {}

with open('temp.txt', 'r') as rfh:
    for line in rfh:
        line = line.replace('\n', '')
        _, proj_name, _ = line.split(' ')
        name, ver = proj_name.split('-')
        if name not in proj.keys():
            proj[name] = []
        proj[name].append(float(ver))

for p in proj.keys():
    l = sorted(proj[p]) 
    if len(l) == 2:
        cmd = 'python main.py ' + p + ' ' + str(l[0]) + ' ' + str(l[1])
        print(cmd)
        r = os.system(cmd)
        if r != 0:
            print(cmd)
            raise ValueError
    else:
        cmd = 'python main.py ' + p + ' ' + str(l[0]) + ' ' + str(l[1])
        print(cmd)
        r = os.system(cmd)
        if r != 0:
            print(cmd)
            raise ValueError
        cmd = 'python main.py ' + p + ' ' + str(l[1]) + ' ' + str(l[2])
        print(cmd)
        r = os.system(cmd)
        if r != 0:
            print(cmd)
            raise ValueError
        
