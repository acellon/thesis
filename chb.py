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
    train = np.asarray(train, dtype='float64')
    train = np.expand_dims(train, axis=1)
    trainlab = np.asarray(trainlab, dtype='int32')
    test = np.asarray(test, dtype='float64')
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
            ep_range = range(st_sec, st_sec + 5)
            trainlab.append(int(any(eeg.is_ict(ep_range))))

    train = np.asarray(train, dtype='float64')
    train = np.expand_dims(train, axis=1)
    trainlab = np.asarray(trainlab, dtype='int32')
    return train, trainlab

def loo_epoch(subj, testnum, testlen=100):
    '''
    # TODO: change this from a random sampling to the test array being
            a straight-up chunk lifted from the data.
            If it's alone in a file, take the whole file.
            If it's not, break the file into pieces.
            Think about normalizing length of training streams. They should
            be relatively similar across the board, though maybe don't need
            to be exact.

            This still requires us to figure out how to make the whole
            prediction thing work, but that can be the next step - this
            should be first.

            Training data should still be shuffled - we don't want the
            evolution of the weights to be screwed up by a long period of
            negative outputs... (I think?) -> I think this might be wrong.
    '''
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
                ep_range = range(st_sec, st_sec + 5)
                ep_label = int(eeg.is_ict(ep_range))
                if (loostart <= st_sec <= loostop):
                    test.append(epoch)
                    testlab.append(ep_label)
                else:
                    train.append(epoch)
                    trainlab.append(int(eeg.is_ict(ep_range)))
        else:
            for st in range(0, eeg_len - epoch_len, stride):
                epoch = eeg.get_rec()[:, st:(st + epoch_len)]
                st_sec = int(st / 256)
                ep_range = range(st_sec, st_sec + 5)
                ep_label = int(eeg.is_ict(ep_range))
                train.append(epoch)
                trainlab.append(ep_label)

    train = np.asarray(train, dtype='float64')
    train = np.expand_dims(train, axis=1)
    trainlab = np.asarray(trainlab, dtype='int32')
    test = np.asarray(test, dtype='float64')
    test = np.expand_dims(test, axis=1)
    testlab = np.asarray(testlab, dtype='int32')
    return train, trainlab, test, testlab
