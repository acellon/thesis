###############################################################################
# File:      chb.py
# Project:   Thesis
# Author:    Adam Cellon
#
# Data types for CHB-MIT EEG data files and associated functions for loading,
# plotting, and labelling data.
###############################################################################
from __future__ import print_function

import numpy as np
import scipy.io as sio
import os
import re
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
try:
    import cPickle as pickle
except:
    import pickle

PATH = '/Users/adamcellon/Drive/senior/thesis/data/'
tigerdata = '/tigress/acellon/data/'

genticks = [
    'FP1-F7', 'F7-T7', 'T7-P7',  'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3',
    'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8',
    'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10',
    'FT10-T8', 'T8-P8'
]

#################################### TO-DOs ####################################
# TODO: add electrodes to CHBfile object


def shuffle_in_unison(a, b):
    '''
    Retrieved from: http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    '''
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return None

def load_dataset(subjname, exthd=False, tiger=False):
    # Load data for subject
    subject = load_meta(subjname, tiger=tiger)
    subject.load_data(exthd=exthd, tiger=tiger)
    return subject

def load_meta(subjname, tiger=False):
    '''
    Function to read/load metadata about list of EEG files from summary text
    file. Metadata includes filename, number of seizures, and seizure
    indices.
    '''
    if tiger:
        dirname = tigerdata + subjname
    else:
        dirname = PATH + subjname
    filename = dirname + '-summary.txt'
    pklname = dirname + '.p'

    subject = CHBsubj()

    if os.path.exists(pklname):
        subject = pickle.load(open(pklname, 'rb'))
        return subject

    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            # Find file metadata and create dict for each file
            fn = re.match(r"File Name: (\w+).edf", line)
            if fn:
                # Add filename and skip two lines
                newfile = CHBfile(fn.group(1))
                if not subjname == 'chb24':
                    f.readline()
                    f.readline()
                # Add number of seizures
                num_szr = int(
                    re.match(r".*Seizures in File: (\d+)", f.readline())
                    .group(1))
                for i in range(num_szr):
                    start = re.match(r".*Start Time: *(\d+) s",
                                     f.readline())
                    start = int(start.group(1))
                    end = re.match(r".*End Time: *(\d+) s", f.readline())
                    end = int(end.group(1))
                    newfile.add_szr((start, end))

                subject.add_file(newfile)

    f.closed
    pickle.dump(subject, open(pklname, 'wb'))
    return subject


############################ CHB-MIT record datatype ###########################


class CHBfile:
    '''
    Dataype for EEG records from the CHB-MIT dataset, available via PhysioBank
    at https://physionet.org/pn6/chbmit/.

    Attributes
    ----------
    name : string
        Name of the EEG file.

    rec : Numpy array (23, 921600) or None
        One hour EEG recording at 256 Hz across 23 channels.

    ict_idx : List of tuples
        Indices (start, stop) of rec marking seizure within EEG file.

    pre_idx : List of tuples
        Indices (start, stop) of rec marking preictal period based on
        eventHorizon defined in add_szr().
    '''

    def __init__(self, name):
        self.name = name
        self.rec = None
        self.ict_idx = []
        #self.epoch   = None
        #self.pre_idx = []

    def __repr__(self):
        return '%r' % (self.__dict__)

    def add_szr(self, seizure):
        # def add_szr(self, seizure, eventHorizon):
        '''
        Add seizure to datatype.

        Parameters
        ----------
        seizure : tuple
            Time indices of seizure in sec from start of file (start, stop)
        '''
        self.ict_idx.append(seizure)
        #prestart = seizure[0] - (eventHorizon * 256)
        #self.pre_idx.append((prestart, seizure[0]))

    def add_rec(self, rec):
        self.rec = rec

    '''
    Functions to return attributes of CHB-MIT file.
        get_name() returns :string:  file name
        get_rec()  returns :ndarray: EEG record (23, 921600)
        get_num()  returns :int:     number of seizures in file
    '''

    def get_name(self):
        return self.name

    def get_rec(self):
        return self.rec

    def get_num(self):
        return len(self.ict_idx)

    def is_idx_ict(self, idx):
        for start, stop in self.ict_idx:
            if (start <= idx <= stop):
                return True
        return False

    def is_list_ict(self, idx_list):
        output = []
        for idx in idx_list:
            for start, stop in self.ict_idx:
                if (start <= idx <= stop):
                    output.append(True)
                else:
                    output.append(False)
        return output

    def copy_meta(self):
        copy = CHBfile(self.name)
        copy.ict_idx = self.ict_idx
        return copy

    def info(self):
        '''
        Print well-formatted, human-readable information about EEG file.
        '''
        print('Name:              %s' % self.name)
        print('Seizure Count:     %d' % self.get_num())
        for j in range(self.get_num()):
            print('-Seizure %d idx:    %s' % (j + 1, self.ict_idx[j]))
        if self.rec is not None:
            print('EEG data:          (%d, %d) array' % self.rec.shape)

    def plot(self, start=0, end=None, chStart=1, chEnd=23, ticks=genticks):
        '''
        Function to plot all EEG channels for a given time period within a
        CHBfile object. Modified from the Matplotlib example retrieved from:
        http://matplotlib.org/examples/pylab_examples/mri_with_eeg.html

        Parameters
        ----------
        start : int (default: 0)
            Start index of EEG to plot.

        end : int (default: length of rec)
            End index of EEG to plot.

        chStart : int (default: 1)
            First channel to plot.

        chEnd : int (default: 23)
            Last channel to plot.
        '''
        rec = self.get_rec()
        starthz = start * 256
        if end is None:
            endhz = rec.shape[1]
            end = int(endhz / 256)
        else:
            endhz = end * 256
        subrec = rec[(chStart - 1):chEnd, starthz:endhz]
        (numRows, numSamples) = subrec.shape

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, title='%s plot' % self.get_name())
        t = np.arange(starthz, endhz) / 256.0

        # Set x size and ticks, y size
        ticklocs = []
        ticksec = end - start
        if ticksec > 1000:
            tickdiff = 180
        elif ticksec > 500:
            tickdiff = 30
        elif ticksec > 100:
            tickdiff = 10
        elif ticksec > 30:
            tickdiff = 5
        else:
            tickdiff = 1

        ax.set_xlim(start, end)
        ax.set_xticks(np.arange(start, end + 1, tickdiff))
        tracemin = subrec.min()
        tracemax = subrec.max()
        traceheight = (tracemax - tracemin) * 0.7  # Crowd them a bit.
        y0 = tracemin
        y1 = (numRows - 1) * traceheight + tracemax
        ax.set_ylim(y0, y1)

        # Add traces for each channel
        traces = []
        for i in range(numRows):
            traces.append(
                np.hstack((t[:, np.newaxis], subrec[i, :, np.newaxis])))
            ticklocs.append(i * traceheight)

        offsets = np.zeros((numRows, 2), dtype=float)
        offsets[:, 1] = ticklocs

        lines = LineCollection(traces, offsets=offsets, transOffset=None)
        ax.add_collection(lines)

        # Set the yticks to use axes coordinates on the y axis
        ax.set_yticks(ticklocs)
        ax.set_yticklabels(ticks[(chStart - 1):chEnd])

        ax.set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()


############################ CHB Subject data type #############################


class CHBsubj(list):
    def __init__(self):
        self.name = None
        self.seizures = []

    def add_file(self, CHBfile):
        self.append(CHBfile)
        for seizure in CHBfile.ict_idx:
            self.seizures.append((CHBfile.get_name(), seizure))

    def get_file(self, filename):
        for CHBfile in self:
            if filename == CHBfile.get_name():
                return CHBfile

    def get_num(self):
        return len(self.seizures)

    def get_seizures(self):
        return self.seizures

    def info(self):
        seiz = self.get_seizures()
        for i in range(self.get_num()):
            print('   -Seizure %d: %s %s' % (i + 1, seiz[i][0], seiz[i][1]))
        sdur, tdur = 0, 0
        for _, (start, stop) in seiz:
            sdur += (stop - start)
        for eeg in self:
            tdur += (eeg.get_rec().shape[1]) / 256
        sper = (sdur / tdur) * 100

        print('Subject: %s' % self[0].get_name().split('_')[0])
        print(' Number of files:    %d' % len(self))
        print(' Number of seizures: %d' % self.get_num())
        print(' Total seizure duration: %d s (%f%%)' % (sdur, sper))


    def load_data(self, verbose=False, exthd=True, tiger=False):
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
            If True, data loaded from external HD. If False, data loaded from PATH.

        tiger : bool (default: False)
            If True, overrides other paths and loads on server.
        '''
        st = time.clock()
        folder, _ = self[0].get_name().split('_')
        if re.match(r"chb17.", folder):
            folder = 'chb17'
        if tiger:
            savename = tigerdata + folder + '.npz'
        elif exthd:
            savename = '/Volumes/extHD/CHBMIT/' + folder + '/' + folder + '.npz'
        else:
            savename = PATH + folder + '.npz'

        if os.path.exists(savename):
            print('Loading:', savename)
            loaddict = np.load(savename)
            for eeg in self:
                eeg.add_rec(loaddict[eeg.get_name()])
            print('Done: %f seconds elapsed.' % (time.clock() - st))
            return
        else:
            savedict = {}
            if not verbose:
                print('Converting mat files to np arrays...')

            for eeg in self:
                if verbose:
                    print('Converting %s.mat to np array' % eeg.get_name())

                eeg.add_rec(sio.loadmat(dirname + eeg.get_name())['rec'])
                savedict[eeg.get_name()] = eeg.get_rec()

            if verbose:
                print('Saving and compressing...')
            np.savez_compressed(savename, **savedict)
            print('Done: %f seconds elapsed.' % (time.clock() - timerstart))

        return


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

    loofile, (loostart, loostop) = subj.get_seizures()[testnum - 1]

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

    shuffle_in_unison(train, trainlab)
    shuffle_in_unison(test, testlab)
    return train, trainlab, test, testlab


def make_epoch(subj): #, testnum, trainlen=1000, testlen=100):
    epoch_len = 256 * 5
    stride = epoch_len / 2
    train, trainlab, test, testlab = [], [], [], []
    for eeg in subj:
        eeg_len = eeg.get_rec().shape[1]
        for st in range(0, eeg_len - epoch_len, stride):
            epoch = eeg.get_rec()[:, st:(st + epoch_len)]
            train.append(epoch)
            st_sec = int(st / 256)
            ep_range = range(st_sec, st_sec + 5)
            trainlab.append(int(any(eeg.is_list_ict(ep_range))))

    train = np.asarray(train, dtype='float64')
    train = np.expand_dims(train, axis=1)
    trainlab = np.asarray(trainlab, dtype='int32')
    return train, trainlab
