import numpy as np
import scipy.io as sio
import os
import re
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PATH = '/Users/adamcellon/Drive/senior/thesis/data/'

# TODO: add electrodes to CHBfile object
genticks = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9',
            'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16', 'Ch17',
            'Ch18', 'Ch19', 'Ch20', 'Ch21', 'Ch22', 'Ch23']

class CHBfile:
    def __init__(self, name):
        self.name    = name
        self.rec     = None
        self.epoch   = None
        self.ict_idx = []
        self.pre_idx = []

    def __repr__(self):
        return '%r' % (self.__dict__)

    def add_szr(self, seizure, eventHorizon):
        self.ict_idx.append(seizure)
        prestart = seizure[0] - (eventHorizon * 256)
        self.pre_idx.append((prestart, seizure[0]))

    def add_rec(self, rec):
        self.rec = rec

    def get_name(self):
        return self.name

    def get_rec(self):
        return self.rec

    def get_num(self):
        return len(self.ict_idx)

    def get_labels(self):
        flen = self.get_rec().shape[1]
        ict = np.zeros(flen, dtype='bool')
        for start, stop in self.ict_idx:
            ict[start:stop] = True

        flen = int(flen/256)
        train = np.zeros((flen, 23, 256))
        label = np.zeros((flen,))
        for ind, epoch in enumerate(np.split(self.get_rec(), flen, axis=1)):
            train[ind,:,:] = epoch
            if ict[ind * 256]:
                label[ind] = 1

        return train, label
        '''
        stack = None
        for j in range(self.get_num()):
            szrlen = int((self.ict_idx[j][1] - self.ict_idx[j][0]) / 256)
            szr = self.get_rec()[:, self.ict_idx[j][0]:self.ict_idx[j][1]]
            stack = np.zeros((szrlen, 23, 256))
            for ind, epoch in enumerate(np.split(szr, szrlen, axis=1)):
                stack[ind,:,:] = epoch
        '''

    def info(self):
            print('Name:              %s' % self.name)
            print('Seizure Count:     %d' % self.get_num())
            for j in range(self.get_num()):
                print('-Seizure %d idx:    %s' % (j + 1, self.ict_idx[j]))
            if self.rec is not None:
                print('EEG data:          (%d, %d) array' % self.rec.shape)

    def plot(self, start=0, end=None, chStart=1, chEnd=23, ticks=genticks):
        '''
        Function to plot all EEG channels for a given time period within a CHBfile
        object. Modified from the Matplotlib example retrieved from: http://matplotlib.org/examples/pylab_examples/mri_with_eeg.html
        '''
        # TODO: check input args, add error handling
        if end is None:
            end = self.rec.shape[1]
        subrec = self.rec[(chStart - 1):chEnd,start:end]
        (numRows, numSamples) = subrec.shape

        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(111, title='%s plot' % self.name)
        t = np.arange(start,end) / 256.0

        # Set x size and ticks, y size
        ticklocs = []
        startsec = int(start/256)
        endsec = int(end/256)
        ticksec = endsec - startsec
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

        ax.set_xlim(startsec, endsec)
        ax.set_xticks(np.arange(startsec, endsec + 1, tickdiff))
        tracemin = subrec.min()
        tracemax = subrec.max()
        traceheight = (tracemax - tracemin) * 0.7  # Crowd them a bit.
        y0 = tracemin
        y1 = (numRows - 1) * traceheight + tracemax
        ax.set_ylim(y0, y1)

        # Add traces for each channel
        traces = []
        for i in range(numRows):
            traces.append(np.hstack((t[:, np.newaxis],
                                     subrec[i, :, np.newaxis])))
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

def load_meta(folder, eventHorizon=5):
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
                num_szr = int(re.match(r".*Seizures in File: (\d+)",
                                       f.readline()).group(1))
                for i in range(num_szr):
                    start = re.match(r".*Start Time: *(\d+) s", f.readline())
                    start = int(start.group(1)) * 256
                    end = re.match(r".*End Time: *(\d+) s", f.readline())
                    end = int(end.group(1)) * 256
                    newfile.add_szr((start, end), eventHorizon)

                # Add file metadata to filelist
                filelist.append(newfile)

    # Close summary file and return filelist
    f.closed
    return filelist

def load_data(filelist, VERBOSE=False, EXTHD=True):
    # Save/load arrays with
    folder, _ = filelist[0].get_name().split('_')
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
            eeg.add_rec(loaddict[eeg.get_name()])
        print('Done.')
        return filelist
    else:
        savedict = {}
        if not VERBOSE:
            print('Converting mat files to np arrays...')

        for eeg in filelist:
            if VERBOSE:
                print('Converting %s.mat to np array' % eeg.get_name())

            eeg.add_rec(sio.loadmat(dirname + eeg.get_name())['rec'])
            savedict[eeg.get_name()] = eeg.get_rec()

        if VERBOSE:
            print('Saving and compressing...')
        np.savez_compressed(savename, **savedict)
        print('Done.')

    return filelist

def trainlabel(filelist):
    '''
    Tried to simply return the ENTIRE FILELIST segmented into epochs in one large numpy array... this got super slow. I'm assuming the bottleneck is np.append(), but it could also just be the sheer amount of data. I'm trying to do it in lists of 2D numpy arrays instead of a 3D numpy array - this was much faster, but it may be smarter to start earlier anyway.
    '''
    trains = []
    labels = []
    for idx, eeg in enumerate(filelist):
        #print('Labeling %s' % eeg.get_name())
        train, label = eeg.get_labels()
        trains.append(train)
        labels.append(label)
        '''
        eegtrain, eeglabel = eeg.get_labels()
        if not idx:
            trains[idx] = eegtrain
            labels[idx] = eeglabel
        else:
            train = np.append(train, eegtrain, axis=0)
            label = np.append(label, eeglabel)
        '''
    return trains, labels
