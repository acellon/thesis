import chb
import os
import sys
import time
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

pth = '/tigress/acellon/data/'
def main(subjname, compressed=True):

    subject = chb.load_meta(subjname, tiger=True)
    sys.stdout.flush()

    op1 = set(['chb14', 'chb20', 'chb21', 'chb22'])
    op2 = set(['chb18', 'chb19'])
    op3 = set(['chb17'])
    op4 = set(['chb16'])
    op5 = set(['chb11'])
    op6 = set(['chb04'])
    op7 = set(['chb09'])

    if subjname in op1:
        pass
    elif subjname in op2:
        del subject[0]
        pklname = pth + subjname + '.p'
        pickle.dump(subject, open(pklname, 'wb'))
    elif subjname in op3:
        del subject[-1]
        pklname = pth + subjname + '.p'
        pickle.dump(subject, open(pklname, 'wb'))
    elif subjname in op4:
        del subject[-2:]
        pklname = pth + subjname + '.p'
        pickle.dump(subject, open(pklname, 'wb'))
    elif subjname in op5:
        pass
    elif subjname in op6:
        pass
    elif subjname in op7:
        pass


if __name__ == '__main__':
    main(sys.argv[1])
