import os

with open('names.txt', 'r') as rfh:
    for line in rfh:
        line = line.replace('\n', '')
        cmd = 'python main.py ' + line
        print(cmd)
        r = os.system(cmd)
        if r != 0:
            raise ValueError
