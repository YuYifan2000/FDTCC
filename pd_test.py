import pandas as pd
import time
import numpy as np
f = open('./test/dt.ct')
cores = 10
Lines = f.readlines()


ss = np.linspace(0, len(Lines), cores+1).astype('int')
print(ss)
seperation = []
for i in range(1,len(ss)):
    print(i)
    tmp = Lines[ss[i]:ss[i]+100]
    for j in range(0,len(tmp)):
        if tmp[j][0] == '#':
            seperation.append(j+ss[i])
            break
print(seperation)
a = Lines[:seperation[0]]

b = Lines[seperation[0]:seperation[1]]

print(len(seperation))
print(np.linspace(0,2,3))