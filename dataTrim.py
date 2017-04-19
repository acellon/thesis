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
    st = time.clock()

    subject = chb.load_dataset(subjname, tiger=True)

    op1 = set(['chb14', 'chb20', 'chb21', 'chb22'])
    op2 = set(['chb18', 'chb19'])
    op3 = set(['chb17'])
    op4 = set(['chb16'])
    op5 = set(['chb11'])
    op6 = set(['chb04'])
    op7 = set(['chb09'])

    if subjname in op1:
        for eeg in subject:
            eeg.add_rec(np.delete(eeg.get_rec(), [4, 9, 12, 17, 22], 0))
    elif subjname in op2:
        del subject[0]
        newsubj = chb.CHBsubj()
        for eeg in subject:
            eeg.add_rec(np.delete(eeg.get_rec(), [4, 9, 12, 17, 22], 0))
            newEEG = CHBfile(eeg.get_name())
            for szr in eeg.ict_idx:
                newEEG.add_szr(szr)
        pklname = pth + subjname + '.p'
        os.remove(pklname)
        pickle.dump(newsubj, open(pklname, 'wb'))
    elif subjname in op3:
        del subject[-1]
        newsubj = chb.CHBsubj()
        for eeg in subject:
            eeg.add_rec(np.delete(eeg.get_rec(), [4, 9, 12, 17, 22], 0))
            for szr in eeg.ict_idx:
                newEEG.add_szr(szr)
        pklname = pth + subjname + '.p'
        os.remove(pklname)
        pickle.dump(newsubj, open(pklname, 'wb'))
    elif subjname in op4:
        del subject[-2:]
        newsubj = chb.CHBsubj()
        for eeg in subject:
            eeg.add_rec(np.delete(eeg.get_rec(), [4, 9, 12, 17, 22], 0))
            for szr in eeg.ict_idx:
                newEEG.add_szr(szr)
        pklname = pth + subjname + '.p'
        os.remove(pklname)
        pickle.dump(newsubj, open(pklname, 'wb'))
    elif subjname in op5:
        for idx, eeg in enumerate(subject):
            if not idx:
                continue
            eeg.add_rec(np.delete(eeg.get_rec(), [4, 9, 12, 17, 22], 0))
    elif subjname in op6:
        for idx, eeg in enumerate(subject):
            if idx < 5:
                continue
            eeg.add_rec(np.delete(eeg.get_rec(), 23, 0))
    elif subjname in op7:
        for idx, eeg in enumerate(subject):
            if not idx:
                continue
            eeg.add_rec(np.delete(eeg.get_rec(), 23, 0))

    savedict = {}
    for eeg in subject:
        savedict[eeg.get_name()] = eeg.get_rec()

    savename = pth + subjname + '.npz'
    os.remove(savename)
    if compressed:
        np.savez_compressed(savename, **savedict)
    else:
        np.savez(savename, **savedict)
    print('Done: %f seconds elapsed.' % (time.clock() - st))

if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['subjname'] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs['compressed'] = bool(sys.argv[2])
    main(**kwargs)
