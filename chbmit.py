###############################################################################
# File:      chbmit.py
# Project:   Thesis
# Author:    Adam Cellon
#
# Data types for CHB-MIT EEG data files and associated functions for loading,
# plotting, and labelling data.
###############################################################################

from __future__ import print_function
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

############################ CHB-MIT record datatype ##########################


class CHBfile:
    '''
    Dataype for EEG records from the CHB-MIT dataset, available via PhysioBank
    at https://physionet.org/pn6/chbmit/. One CHBfile object corresponds to one
    .edf file in the database.

    Initialization:
        chbfile = CHBfile(name)

    Attributes:
        name     | :string:   | name of .edf file (e.g. 'chb01_05')
        rec      | :ndarray:  | 23 channel EEG recording
        ict_idx  | :[tuple]:  | list of (start, stop) indices of seizure events

    Functions - setting:
        add_szr(tuple)
        add_rec(ndarray)

    Functions - getting:
        get_name() returns :string: name
        get_rec()  returns :ndarray: rec
        get_num()  returns :int: number of seizures
        get_ict()  returns :[tuple]: list of seizure indices

    Functions - helper:
        is_ict(idx) returns :bool: does idx include a seizure event
        copy_meta() returns :CHBfile: copy, excluding rec
        info()      displays humean-readable information about CHBfile object
        plot(...)   displays plot of file - see comment below for parameters
    '''

    def __init__(self, name):
        self.name = name
        self.rec = None
        self.ict_idx = []

    def __repr__(self):
        return '%r' % (self.__dict__)

    def add_szr(self, seizure):
        self.ict_idx.append(seizure)

    def add_rec(self, rec):
        self.rec = rec

    def get_name(self):
        return self.name

    def get_rec(self):
        return self.rec

    def get_num(self):
        return len(self.ict_idx)

    def get_ict(self):
        return self.ict_idx

    def is_ict(self, idx):
        if type(idx) is int:
            idx = [idx]
        for i in idx:
            for start, stop in self.ict_idx:
                if (start <= i <= stop):
                    return True
        return False

    def copy_meta(self):
        copy = CHBfile(self.name)
        copy.ict_idx = self.get_ict()
        return copy

    def info(self):
        print('Name:              %s' % self.name)
        print('Seizure Count:     %d' % self.get_num())
        for j in range(self.get_num()):
            print('-Seizure %d idx:    %s' % (j + 1, self.ict_idx[j]))
        if self.rec is not None:
            print('EEG data:          (%d, %d) array' % self.rec.shape)

    def plot(self, start=0, end=None, chList=list(range(23))):
        '''
        Modified from the Matplotlib example retrieved from:
        http://matplotlib.org/examples/pylab_examples/mri_with_eeg.html

        Parameters:
            start   | :int: | plot start time (default: 0)
            end     | :int: | plot end time (default: length of rec)
            chStart | :int: | first channel to plot (default: 1)
            chEnd   | :int: | last channel to plot (default: 23)

        Note: traces build up from bottom (in reverse to preserve order)
        '''

        rec = self.get_rec()
        starthz = start * 256
        if end is None:
            endhz = rec.shape[1]
            end = int(endhz / 256)
        else:
            endhz = end * 256
        subrec = rec[chList, starthz:endhz]
        (numRows, numSamples) = subrec.shape

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, title='%s plot' % self.get_name())
        t = np.arange(starthz, endhz) / 256.0

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

        traces = []
        for i in range(numRows):
            traces.append(
                np.hstack((t[:, np.newaxis], subrec[-i - 1, :, np.newaxis])))
            ticklocs.append(i * traceheight)

        offsets = np.zeros((numRows, 2), dtype=float)
        offsets[:, 1] = ticklocs

        lines = LineCollection(traces, offsets=offsets, transOffset=None)
        ax.add_collection(lines)

        yticks = [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3',
            'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8',
            'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10',
            'FT10-T8', 'T8-P8'
        ]
        yticks = [yticks[i] for i in chList]
        ax.set_yticks(ticklocs)
        ax.set_yticklabels(yticks[::-1])

        ax.set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()


############################ CHB Subject data type ############################


class CHBsubj(list):
    '''
    Modified list for subject data from the CHB-MIT dataset. One CHBsubj object
    contains the (meta)data for a single subject in the dataset, built up of
    many .edf files.

    Initialization:
        chbsubj = CHBsubj(name)

    Attributes:
        name     | :string:  | name of subject (e.g. 'chb09')
        seizures | :[(string, tuple)]: | list of indices of all seizure events
                                         for subject

    Functions - setting:
        add_file(CHBfile)

    Functions - getting:
        get_name() returns :string: name
        get_num()  returns :int: total number of seizures
        get_ict()  returns :[tuple]: list of seizure indices
        get_file(fname)  returns :CHBfile: indexed by CHBfile.name

    Functions - helper:
        info()      displays humean-readable information about CHBsubj object
    '''

    def __init__(self, name):
        self.name = name
        self.seizures = []

    def add_file(self, chbfile):
        self.append(chbfile)
        for seizure in chbfile.ict_idx:
            self.seizures.append((chbfile.get_name(), seizure))

    def get_name(self):
        return self.name

    def get_num(self):
        return len(self.seizures)

    def get_ict(self):
        return self.seizures

    def get_file(self, fname):
        for chbfile in self:
            if fname == chbfile.get_name():
                return chbfile

    def info(self):
        seiz = self.get_ict()
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
