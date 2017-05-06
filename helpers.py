import numpy as np
import matplotlib.pyplot as plt
import sys, os
import scipy.io as sio

def plotprob(npz, thresh=0.8):
    for num in range(int(len(npz.files)/2)):
        trues = '_'.join(['true',str(num+1)])
        probs = '_'.join(['prob',str(num+1)])
        thr = np.ones(len(npz[trues]))*thresh
        plt.figure(num+1)
        #plt.axvspan(npz[trues].nonzero()[0][0],npz[trues].nonzero()[0][-1], alpha=0.25, color='r')
        plt.plot(npz[probs])
        plt.plot(npz[trues])
        plt.plot(thr)
        ttl = ' '.join(['Seizure', str(num+1)])
        plt.title(ttl)
        #plt.grid()


def plotflip(npz, figoff=0):
    for num in range(int(len(npz.files)/2)):
        trues = '_'.join(['true',str(num+1)])
        probs = '_'.join(['prob',str(num+1)])
        x = np.arange(len(npz[probs]))
        plt.figure(num+1+figoff)
        plt.plot(x, npz[trues], x, 1 - npz[probs]/np.max(npz[probs]))
        ttl = ' '.join(['Seizure', str(num+1)])
        plt.title(ttl)

def convertLucy(dirpath):
    for file_ in [f for f in os.listdir(dirpath) if f.endswith('.mat')]:
        temp = sio.loadmat(''.join([dirpath, file_]))
        inds = [int(t.squeeze()) for t in temp['testTimes']]
        prob = np.ones(inds[-1]+1) * np.min(temp['ySVMSO'])
        true = np.zeros(inds[-1]+1)
        for i in range(len(temp['ySVMSO'])):
            prob[inds[i]] = temp['ySVMSO'][i]
            true[inds[i]] = int(temp['trueLabels'][i]>0)
        np.savez(file_, svmprob=prob, true=true)

def szr_pc(subj, osr=1, usp=0):
    total, seizure = 0, 0
    for file_, (start, stop) in subj.get_ict():
        total += to_s(subj.get_file(file_).get_rec().shape[1])
        seizure += (stop - start)
    return ((seizure * osr)/(total * (1 - usp))) * 100

def to_s(hz):
    if type(hz) is int:
        return int(hz / 256)
    elif type(hz) is list:
        return [int(hz / 256) for f in hz]
    elif type(hz) is np.ndarray:
        return hz / 256

def metrics(npz, thresh=0.8):
    npzlen = int(len(npz.files)/2)
    for i in range(1, npzlen + 1):
        pred = (npz['_'.join(['prob', str(i)])] > thresh).astype('int32')
        mcc = matthews_corrcoef(npz['_'.join(['true', str(i)])], pred)
        cm = confusion_matrix(npz['_'.join(['true', str(i)])], pred)
        print('Matthews Correlation Coefficient: \t{}'.format(mcc))
        print('Confusion matrix:\n{}'.format(cm))
