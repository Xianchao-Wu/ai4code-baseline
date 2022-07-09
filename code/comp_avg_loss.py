import sys
import os

import numpy as np

epoch2loss = dict()

for aline in sys.stdin:
    aline = aline.strip()
    aline = aline.replace('Epoch=', '\nEpoch=')
    #if aline.startswith('Epoch='):

    all_cols = aline.split('\n')
    for aline in all_cols:
        if not aline.startswith('Epoch='):
            continue

        cols = aline.split(', ')
        #print(cols)
        print(cols[0], cols[1])
        aepoch = int(cols[0].split('=')[-1])
        aloss = float(cols[1].split('=')[-1])

        alist = epoch2loss[aepoch] if aepoch in epoch2loss else []
        alist.append(aloss)

        epoch2loss[aepoch] = alist

for aepoch in epoch2loss:
    loss_list = epoch2loss[aepoch]
    avg_loss = np.mean(loss_list)

    print(aepoch, avg_loss, len(loss_list))



