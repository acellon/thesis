###############################################################################
# File:      chb.py
# Project:   Thesis
# Author:    Adam Cellon
#
# Data types for CHB-MIT EEG data files and associated functions for loading,
# plotting, and labelling data.
###############################################################################
from __future__ import print_function
import chbmit
from chbmit import CHBsubj, CHBfile
import numpy as np
import scipy.io as sio
import os
import re
import time
try:
    import cPickle as pickle
except:
    import pickle

mac = '/Users/adamcellon/Drive/senior/thesis/data/'
tigerdata = '/tigress/acellon/data/'

###############################################################################


def shuffle_in_unison(a, b):
    '''
    Retrieved from: http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    '''
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return rng_state


def to_hz(sec):
    if type(sec) is int or type(sec) is float:
        return int(sec * 256)
    elif type(sec) is list:
        return [int(s * 256) for s in sec]
    elif type(sec) is np.ndarray:
        return sec * 256


def to_s(hz):
    if type(hz) is int:
        return int(hz / 256)
    elif type(hz) is list:
        return [int(hz / 256) for f in hz]
    elif type(hz) is np.ndarray:
        return hz / 256


############################# Loading functions ###############################


def load_meta(subjname, tiger=False):
    '''
    Function to read/load metadata about list of EEG files from summary text
    file. Metadata includes filename, number of seizures, and seizure
    indices.
    '''
    if tiger:
        dirname = tigerdata + subjname
    else:
        dirname = mac + subjname
    filename = dirname + '-summary.txt'
    pklname = dirname + '.p'

    subject = chbmit.CHBsubj(subjname)

    if os.path.exists(pklname):
        subject = pickle.load(open(pklname, 'rb'))
        return subject

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            fn = re.match(r"File Name: (\w+).edf", line)
            if fn:
                # Add filename and skip two lines
                newfile = chbmit.CHBfile(fn.group(1))
                if not subjname == 'chb24':
                    f.readline()
                    f.readline()
                # Add number of seizures
                num_szr = int(
                    re.match(r".*Seizures in File: (\d+)", f.readline())
                    .group(1))
                for i in range(num_szr):
                    start = re.match(r".*Start Time: *(\d+) s", f.readline())
                    start = int(start.group(1))
                    end = re.match(r".*End Time: *(\d+) s", f.readline())
                    end = int(end.group(1))
                    newfile.add_szr((start, end))

                subject.add_file(newfile)

    f.closed
    pickle.dump(subject, open(pklname, 'wb'))
    return subject


def load_data(subject, verbose=False, exthd=True, tiger=False):
    '''
    Loads EEG records, either from compressed .npz file or by converting
    .mat files.

    Parameters
    ----------
    filelist : list of CHBfile
        One subject's list of CHBfile objects

    verbose : bool (default: False)
        Controls how much output info is given.

    exthd : bool (default: True)
        If True, data loaded from external HD. If False, data loaded from mac.

    tiger : bool (default: False)
        If True, overrides other paths and loads on server.
    '''
    st = time.clock()
    folder, _ = subject[0].get_name().split('_')
    if re.match(r"chb17.", folder):
        folder = 'chb17'
    if tiger:
        savename = tigerdata + folder + '.npz'
    elif exthd:
        savename = '/Volumes/extHD/CHBMIT/' + folder + '/' + folder + '.npz'
    else:
        savename = mac + folder + '.npz'

    if os.path.exists(savename):
        print('Loading:', savename)
        loaddict = np.load(savename)
        for eeg in subject:
            eeg.add_rec(loaddict[eeg.get_name()])
        print('Done: %f seconds elapsed.' % (time.clock() - st))
        return
    else:
        savedict = {}
        if not verbose:
            print('Converting mat files to np arrays...')

        for eeg in subject:
            if verbose:
                print('Converting %s.mat to np array' % eeg.get_name())

            eeg.add_rec(sio.loadmat(dirname + eeg.get_name())['rec'])
            savedict[eeg.get_name()] = eeg.get_rec()

        if verbose:
            print('Saving and compressing...')
        np.savez_compressed(savename, **savedict)
        print('Done: %f seconds elapsed.' % (time.clock() - timerstart))

    return


def load_dataset(subjname, exthd=False, tiger=False):
    subject = load_meta(subjname, tiger=tiger)
    load_data(subject, exthd=exthd, tiger=tiger)
    return subject


############################# Labeling functions ###############################


def get_labels(eeg, label_len=500):
    '''
    Format and label raw EEG data into non-overlapping 1-sec epochs.

    Returns
    -------
    train : list of ndarray len([...,(23, 256),...]) = 3600
        List of 1-second epochs of EEG data

    label : list of int
        Labels of corresponding epochs in train; 0 = non-seizure, 1 = ictal

    Notes
    -----
    Data is manipulated and returned in lists to avoid wasting memory on
    np.append(), which does not append in-place.
    '''
    # Create label list
    flen = int(eeg.get_rec().shape[1] / 256)
    label = [0] * flen
    for start, stop in eeg.ict_idx:
        for i in range(start, stop):
            label[i] = 1

    # Create train list of EEG epochs
    train = []
    for epoch in np.split(eeg.get_rec(), flen, axis=1):
        train.append(epoch)

    # Return epochs in lists
    return train, label


def label_epochs(subj, streamlen=1000):
    '''
    Converts EEG data for subject into non-overlapping 1-sec epochs labeled as
    either ictal or non-seizure.

    Parameters
    ----------
    filelist : list of CHBfile
        One subject's list of CHBfile objects.

    streamlen : int (default: 1000)
        Number of epochs to be returned. Controls number of non-seizure epochs
        that are randomly eliminated.

    Returns
    -------
    epochs : ndarray (streamlen, 23, 256), dtype=float
        Numpy array of streamlen non-overlapping 1-sec epochs.
    labels : ndarray (streamlen,), dtype=float
        Numpy array of streamlen labels for corresponding epochs.

    Notes
    -----
    - Epochs are NOT necessarily continuous (except within a seizure event).
    - Operations are performed on lists, then returned as ndarrays to avoid
      appending numpy arrays.
    '''
    labels = []
    epochs = []

    # Move each epoch/label from list for file to total list for subject
    for eeg in subj:
        epoch, label = eeg.get_labels()
        for idx in range(len(label)):
            epochs.append(epoch[idx])
            labels.append(label[idx])

    # Randomly remove non-seizure epochs
    while (len(labels) > streamlen):
        idx = np.random.randint(len(labels))
        if not labels[idx]:
            del epochs[idx]
            del labels[idx]

    # Return as numpy arrays
    epochs = np.expand_dims(np.asarray(epochs), axis=1)
    labels = np.asarray(labels, dtype='int32')
    return epochs, labels


def leaveOneOut(subj, testnum, trainlen=1000, testlen=100):

    loofile, (loostart, loostop) = subj.get_ict()[testnum - 1]

    # Create train, test, and label lists
    train, trainlab, test, testlab = [], [], [], []
    for eeg in subj:
        eeglen = int(eeg.get_rec().shape[1] / 256)

        if (loofile == eeg.get_name()):
            for t, epoch in enumerate(np.split(eeg.get_rec(), eeglen, axis=1)):
                if (loostart <= t <= loostop):
                    test.append(epoch)
                    testlab.append(1)
                else:
                    train.append(epoch)
                    trainlab.append(int(eeg.is_ict(t)))

        else:
            for t, epoch in enumerate(np.split(eeg.get_rec(), eeglen, axis=1)):
                train.append(epoch)
                trainlab.append(int(eeg.is_ict(t)))

    while (len(train) > trainlen):
        idx = np.random.randint(len(train))
        if not trainlab[idx]:
            if (len(test) < testlen):
                test.append(train.pop(idx))
                testlab.append(trainlab.pop(idx))
            else:
                del train[idx]
                del trainlab[idx]

    # Return as co-shuffled numpy arrays
    train = np.asarray(train, dtype='float32')
    train = np.expand_dims(train, axis=1)
    trainlab = np.asarray(trainlab, dtype='int32')
    test = np.asarray(test, dtype='float32')
    test = np.expand_dims(test, axis=1)
    testlab = np.asarray(testlab, dtype='int32')

    rng_train = shuffle_in_unison(train, trainlab)
    rng_test = shuffle_in_unison(test, testlab)
    return train, trainlab, test, testlab


def make_epoch(subj):
    epoch_len = 256 * 5
    stride = epoch_len / 2
    train, trainlab = [], []
    for eeg in subj:
        eeg_len = eeg.get_rec().shape[1]
        for st in range(0, eeg_len - epoch_len, stride):
            epoch = eeg.get_rec()[:, st:(st + epoch_len)]
            train.append(epoch)
            st_sec = int(st / 256)
            trainlab.append(int(eeg.is_ict(st_sec)))

    train = np.asarray(train, dtype='float32')
    train = np.expand_dims(train, axis=1)
    trainlab = np.asarray(trainlab, dtype='int32')
    return train, trainlab


def loo_epoch(subj, testnum, testlen=100):
    loofile, (ictstart, ictstop) = subj.get_ict()[testnum - 1]
    loofile_len = subj.get_file(loofile).get_rec().shape[1]
    if ictstart < 500:
        loostart = 0
        loostop = loostart + 1000
    elif ictstop > loofile_len - 505:
        loostop = loofile_len - 6
        loostart = loostop - 1000
    else:
        loostart = ictstart - 500
        loostop = loostart + 1000
    epoch_len = 256 * 5
    stride = epoch_len / 5
    train, trainlab, test, testlab = [], [], [], []
    for eeg in subj:
        eeg_len = eeg.get_rec().shape[1]

        if (loofile == eeg.get_name()):
            for st in range(0, eeg_len - epoch_len, stride):
                epoch = eeg.get_rec()[:, st:(st + epoch_len)]
                st_sec = int(st / 256)
                ep_label = int(eeg.is_ict(st_sec))
                if (loostart <= st_sec < loostop):
                    test.append(epoch)
                    testlab.append(ep_label)
                else:
                    train.append(epoch)
                    trainlab.append(int(eeg.is_ict(st_sec)))
        else:
            for st in range(0, eeg_len - epoch_len, stride):
                epoch = eeg.get_rec()[:, st:(st + epoch_len)]
                st_sec = to_s(st)
                ep_label = int(eeg.is_ict(st_sec))
                train.append(epoch)
                trainlab.append(ep_label)

    train = np.asarray(train, dtype='float32')
    train = np.expand_dims(train, axis=1)
    trainlab = np.asarray(trainlab, dtype='int32')
    test = np.asarray(test, dtype='float32')
    test = np.expand_dims(test, axis=1)
    testlab = np.asarray(testlab, dtype='int32')
    return train, trainlab, test, testlab


def epoch_gen(subj, batchsec=10, shuffle=False):
    batchhz = to_hz(batchsec)
    epochlen = to_hz(5)
    stride = to_hz(1)
    for eeg in subj:
        eeglen = eeg.get_rec().shape[1]
        #nonzero = 0 #####
        if shuffle:
            indices = np.arange(eeglen)
            np.random.shuffle(indices)
        epoch_list = []
        label_list = []
        for epochstart in range(0, eeglen - epochlen + to_hz(1), stride):
            if shuffle:
                excerpt = indices[epochstart:epochstart + epochlen]
                exStart = np.asscalar(excerpt[0])
                label = eeg.is_ict(to_s(exStart))
            else:
                excerpt = slice(epochstart, epochstart + epochlen)
                label = eeg.is_ict(to_s(epochstart))
            epoch_list.append(eeg.get_rec()[:, excerpt])
            label_list.append(label)
            if len(epoch_list) == batchsec:
                inputs = np.asarray(epoch_list, dtype='float32')
                inputs = np.expand_dims(inputs, axis=1)
                targets = np.asarray(label_list, dtype='int32')
                #nonzero += np.count_nonzero(targets)
                epoch_list, label_list = [], []
                yield inputs, targets
        #print('nonzero for ',eeg.get_name(),':',nonzero)

def loo_gen(subj, loonum, batchsec=10, shuffle=False):
    batchhz = to_hz(batchsec)
    winlen, stride = to_hz(5), to_hz(1)

    looname, (ictstart, ictstop) = subj.get_ict()[loonum - 1]
    loofile = subj.get_file(looname)
    loolensec = to_s(loofile.get_rec().shape[1])
    if ictstart < 500:
        loostart = 0
        loostop = loostart + 1000
    elif ictstop > loolensec - 505:
        loostop = loolensec - 6
        loostart = loostop - 1000
    else:
        loostart = ictstart - 500
        loostop = loostart + 1000
    testlist, testlabel = [], []
    for start in range(loostart, loostop, to_s(stride)):
        excerpt = loofile.get_rec()[:, to_hz(start):to_hz(start) + winlen]
        testlist.append(excerpt)
        testlabel.append(int(loofile.is_ict(start)))
    inputs = np.asarray(testlist, dtype='float32')
    inputs = np.expand_dims(inputs, axis=1)
    targets = np.asarray(testlabel, dtype='int32')
    #print('nonzero removed for test:',np.count_nonzero(targets)) #$#
    yield inputs, targets

    for eeg in subj:
        eeglen = eeg.get_rec().shape[1]
        #nonzero = 0 #$#
        if (eeg.get_name() == looname):
            first = list(range(to_hz(loostart)))
            last = list(range(to_hz(loostop), to_hz(loolensec - 6)))
            fullList = first + last
            indices = np.asarray(fullList)
        else:
            indices = np.arange(eeglen)
        if shuffle:
            np.random.shuffle(indices)
        indlen = len(indices)
        imglist, lablist = [], []
        for idx, start in enumerate(range(0, indlen - winlen + to_hz(1), stride)):
            excerpt = indices[start:start + winlen]
            exStart = np.asscalar(excerpt[0])
            label = eeg.is_ict(to_s(exStart))

            imglist.append(eeg.get_rec()[:, excerpt])
            lablist.append(int(label))
            if len(imglist) == batchsec:
                inputs = np.asarray(imglist, dtype='float32')
                inputs = np.expand_dims(inputs, axis=1)
                targets = np.asarray(lablist, dtype='int32')
                #nonzero += np.count_nonzero(targets) #$#
                imglist, lablist = [], []
                yield inputs, targets
        #print('nonzero for ',eeg.get_name(),':',nonzero) #$#

def lgus(subj, loonum, batchsec=60,  drop_prob=0, shuffle=True):
    batchhz = to_hz(batchsec)
    winlen, stride = to_hz(5), to_hz(1)

    # return the testing data on the first pass through the generator
    looname, (ictstart, ictstop) = subj.get_ict()[loonum - 1]
    loofile = subj.get_file(looname)
    loolensec = to_s(loofile.get_rec().shape[1])
    if ictstart < 500:
        loostart = 0
        loostop = loostart + 1000
    elif ictstop > loolensec - 505:
        loostop = loolensec - 6
        loostart = loostop - 1000
    else:
        loostart = ictstart - 500
        loostop = loostart + 1000
    testlist, testlabel = [], []
    for start in range(loostart, loostop, to_s(stride)):
        excerpt = loofile.get_rec()[:, to_hz(start):to_hz(start) + winlen]
        testlist.append(excerpt)
        testlabel.append(int(loofile.is_ict(start)))
    inputs = np.asarray(testlist, dtype='float32')
    inputs = np.expand_dims(inputs, axis=1)
    targets = np.asarray(testlabel, dtype='int32')
    #print('nonzero removed for test:',np.count_nonzero(targets)) #$#
    yield inputs, targets

    # undersample the majority group (is_ict == 0) with probability drop_prob

    for eeg in subj:
        eeglen = eeg.get_rec().shape[1]
        #nonzero = 0 #$#
        if (eeg.get_name() == looname):
            first = list(range(to_hz(loostart)))
            last = list(range(to_hz(loostop), to_hz(loolensec - 6)))
            fullList = first + last
            indices = np.asarray(fullList)
        else:
            indices = np.arange(eeglen)
        if shuffle:
            np.random.shuffle(indices)
        indlen = len(indices)
        imglist, lablist = [], []
        for idx, start in enumerate(range(0, indlen - winlen, stride)):
            excerpt = indices[start:start + winlen]
            exStart = np.asscalar(excerpt[0])
            label = eeg.is_ict(to_s(exStart))
            if (not label) and (np.random.random() < drop_prob):
                continue

            imglist.append(eeg.get_rec()[:, excerpt])
            lablist.append(int(label))
            if len(imglist) == batchsec:
                inputs = np.asarray(imglist, dtype='float32')
                inputs = np.expand_dims(inputs, axis=1)
                targets = np.asarray(lablist, dtype='int32')
                #nonzero += np.count_nonzero(targets) #$#
                imglist, lablist = [], []
                yield inputs, targets
        #print('nonzero for ',eeg.get_name(),':',nonzero) #$#


def loowinTest(subj, loonum, batchsec=10):
    batchhz = to_hz(batchsec)
    winlen, stride = to_hz(5), to_hz(1)

    looname, _ = subj.get_ict()[loonum - 1]
    loofile = subj.get_file(looname)
    loolen = loofile.get_rec().shape[1]
    testlist, testlabel = [], []
    for start in range(0, loolen - winlen + to_hz(1), stride):
        excerpt = loofile.get_rec()[:, start:start + winlen]
        testlist.append(excerpt)
        testlabel.append(int(loofile.is_ict(to_s(start))))
    inputs = np.asarray(testlist, dtype='float32')
    inputs = np.expand_dims(inputs, axis=1)
    targets = np.asarray(testlabel, dtype='int32')
    testlist, testlabel = [], []
    return inputs, targets

def loowinTrain(subj, loonum, osr=1, usp=0, batchsec=10):
    winlen, stride = to_hz(5), to_hz(1)
    looname, _ = subj.get_ict()[loonum - 1]
    for name, (istart, istop) in subj.get_ict():
        if name == looname:
            continue
        eeg = subj.get_file(name)
        eeglen = eeg.get_rec().shape[1]
        windows, labels = [], []
        start = 0
        while start < eeglen - winlen + to_hz(1):
            excerpt = slice(start, start + winlen)
            label = eeg.is_ict(to_s(start))
            if (not label) and (np.random.random() < usp):
                start += stride
                continue
            recExcerpt = eeg.get_rec()[:, excerpt]
            exLen = recExcerpt.shape[1]
            if exLen == winlen:
                windows.append(recExcerpt)
                labels.append(int(label))
            start += stride
            if to_s(start) == istart:
                stride = to_hz(1.0/osr)
            if to_s(start) == istop:
                stride = to_hz(1)
            if len(windows) == batchsec:
                inputs = np.asarray(windows, dtype='float32')
                inputs = np.expand_dims(inputs, axis=1)
                targets = np.asarray(labels, dtype='int32')
                windows, labels = [], []
                yield inputs, targets
