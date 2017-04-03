import numpy as np
import scipy.io as sio
import os
import re
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

'''
import theano
import theano.tensor as T

import lasagne
'''
PATH = '/Users/adamcellon/Drive/senior/thesis/data/'

class CHBfile:
    def __init__(self, name):
        self.name = name
        self.num_szr = 0
        self.start = []
        self.end = []
        self.rec = None
        self.ict = None
        self.preict = None

    def __repr__(self):
        return '%r' % (self.__dict__)

    def pretty(self):
        print('Name:              %s' % self.name)
        print('Seizure Count:     %d' % self.num_szr)
        for j in range(self.num_szr):
            print('-Seizure %d range:  [%d - %d]' % (j + 1, self.start[j], self.end[j]))
        if self.rec is not None:
            print('EEG data:          (%d, %d) array' % self.rec.shape)
        if self.ict is not None:
            print('Ictal mask:        (%d, %d) array' % self.ict.shape)
        if self.preict is not None:
            print('Preictal mask:     (%d, %d) array' % self.preict.shape)

    '''
    Function to plot all EEG channels for a given time period within a CHBfile
    object. Modified from the Matplotlib example retrieved from: http://matplotlib.org/examples/pylab_examples/mri_with_eeg.html
    '''
    def plot(self,start=0,end=None):
        if end is None:
            end = self.rec.shape[1]
        subrec = self.rec[:,start:end]
        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111, title='%s plot' % self.name)
        (numRows, numSamples) = subrec.shape
        t = np.arange(start,end) / 256.0

        # Set x size and ticks, y size - TODO: make dynamic
        ticklocs = []
        ax.set_xlim(int(start/256), int(end/256))
        ax.set_xticks(np.arange(int(start/256),int(end/256)+1, 60))
        dmin = subrec.min()
        dmax = subrec.max()
        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        y0 = dmin
        y1 = (numRows - 1) * dr + dmax
        ax.set_ylim(y0, y1)

        segs = []
        for i in range(numRows):
            segs.append(np.hstack((t[:, np.newaxis], subrec[i, :, np.newaxis])))
            ticklocs.append(i * dr)

        offsets = np.zeros((numRows, 2), dtype=float)
        offsets[:, 1] = ticklocs

        lines = LineCollection(segs, offsets=offsets, transOffset=None)
        ax.add_collection(lines)

        # Set the yticks to use axes coordinates on the y axis
        ax.set_yticks(ticklocs)
        ax.set_yticklabels(['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7',
         'Ch8', 'Ch9', 'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16',
         'Ch17', 'Ch18', 'Ch19', 'Ch20', 'Ch21', 'Ch22', 'Ch23'])

        ax.set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

def summary(folder):
    # Locate summary textfile
    dirname = PATH + folder
    #filename = dirname + '/' + folder + '-summary.txt'
    filename = dirname + '-summary.txt'

    # Create empty list of EEG files
    filelist = []
    with open(filename, 'r') as f:
        for line in f:
            # Find file metadata and create dict for each file
            fn = re.match(r"File Name: (\w+).edf", line)
            if fn:
                # Add filename and skip two lines
                newfile = CHBfile(fn.group(1))
                if not folder == 'chb24':
                    f.readline(); f.readline();
                # Add number of seizures
                num_szr = re.match(r".*Seizures in File: (\d+)", f.readline())
                newfile.num_szr = int(num_szr.group(1))

                # If file includes seizures, add start and end times
                for i in range(newfile.num_szr):
                    start = re.match(r".*Start Time: *(\d+) s", f.readline())
                    newfile.start.append(int(start.group(1)) * 256)
                    end = re.match(r".*End Time: *(\d+) s", f.readline())
                    newfile.end.append(int(end.group(1)) * 256)

                # Add file metadata to filelist
                filelist.append(newfile)

    # Close summary file and return filelist
    f.closed
    return filelist

def load_data(filelist, VERBOSE=False, EXTHD=True):
    # Save/load arrays with
    folder, _ = filelist[0].name.split('_')
    if re.match(r"chb17.", folder):
        folder = 'chb17'
    if EXTHD:
        savename = '/Volumes/extHD/CHBMIT/' + folder + '/' + folder + '.npz'
    else:
        savename = PATH + folder + '.npz'

    if os.path.exists(savename):
        print('Loading:', savename)
        loaddict = np.load(savename)
        for eeg in filelist:
            eeg.rec = loaddict[eeg.name]
        print('Done.')
        return filelist
    else:
        savedict = {}
        if not VERBOSE:
            print('Converting mat files to np arrays...')

        for eeg in filelist:
            if VERBOSE:
                print('Converting %s.mat to np array' % eeg.name)

            eeg.rec = sio.loadmat(dirname + eeg.name)['rec']
            savedict[eeg.name] = eeg.rec

        if VERBOSE:
            print('Saving and compressing...')
        np.savez_compressed(savename, **savedict)
        print('Done.')

    return filelist

def label(filelist, H=5):
    # Check to see if filelist contains rec data
    if filelist[0].rec is None:
        print('No data has been loaded for this filelist. Please use chb.load_data().')
        return filelist

    # Convert event horizon to sample size (minutes to 1/256 seconds)
        # need to decide what to do about event horizon going before start
    H = H * 60 * 256
    start, end = 0, 0
    for eegfile in filelist:
        ict = np.zeros_like(eegfile.rec)
        preict = np.copy(ict)

        for i in range(eegfile.num_szr):
            start = eegfile.start[i]
            end = eegfile.end[i]
            ict[:, start:end] = 1
            # print('ict indices: [',start,',',end,']',sep='')
            preict[:, (start - H):start] = 1
            # print('preict indices: [',start-H,',',start,']',sep='')

        eegfile.ict = ict
        eegfile.preict = preict

    return filelist
