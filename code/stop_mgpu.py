import sys
import os

alogfn = 'temp.log'
#acmd = 'ps > {}'.format(alogfn)

#out = os.system(acmd)
#print(out)


ids = []
acount = 1008
with open(alogfn) as br:
    for aline in br.readlines():
        aline = aline.strip()
        aline = str(aline)
        if 'train_mgpu' in aline:
            print(aline)
            ids.append(aline.split(' ')[0])
        if len(ids) == acount:
            break

print(ids)
acmd = 'kill {}'.format(' '.join(ids))
print(acmd)
os.system(acmd)


