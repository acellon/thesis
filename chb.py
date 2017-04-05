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
        self.num_szr = 0
        self.rec     = None
        self.ict_idx = []
        self.pre_idx = []

    def __repr__(self):
        return '%r' % (self.__dict__)

    def add_szr(self, seizure, eventHorizon):
        self.num_szr += 1
        self.ict_idx.append(seizure)
        prestart = seizure[0] - (eventHorizon * 256)
        self.pre_idx.append((prestart, seizure[0]))

    def add_rec(self, rec):
        self.rec = rec

    def info(self):
            print('Name:              %s' % self.name)
            print('Seizure Count:     %d' % self.num_szr)
            for j in range(self.num_szr):
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
            eeg.add_rec(loaddict[eeg.name])
        print('Done.')
        return filelist
    else:
        savedict = {}
        if not VERBOSE:
            print('Converting mat files to np arrays...')

        for eeg in filelist:
            if VERBOSE:
                print('Converting %s.mat to np array' % eeg.name)

            eeg.add_rec(sio.loadmat(dirname + eeg.name)['rec'])
            savedict[eeg.name] = eeg.rec

        if VERBOSE:
            print('Saving and compressing...')
        np.savez_compressed(savename, **savedict)
        print('Done.')

    return filelist

def trainstream(filelist, numstream=10, streamlen=30):
    streamlen = streamlen * 256

    norms = []
    for n in range(numstream):
        while True:
            eeg = np.random.choice(filelist)
            norm_start = np.random.randint(0, eeg.rec.shape[1] - streamlen)
            norm_end = norm_start + streamlen
            #print('(%d,%d)' % (streamst, streamend))
            if not ((eeg.pre_idx[0] <= norm_start <= eeg.ict_idx[1]) or (eeg.pre_idx[0] <= norm_end <= eeg.ict_idx[1])):
                break

        norm = eeg.rec[:,norm_start:norm_end]
        print(norm)
        norms.append(norm)

    icts = []
    for n in range(numstream):
        while True:
            eeg = np.random.choice(filelist)
            if eeg.num_szr == 0:
                continue
            ict_start = np.random.randint(0, eeg.rec.shape[1] - streamlen)
            ict_end = ict_start + streamlen
            #print('(%d,%d)' % (ictstreamst, ictstreamend))
            if (eeg.ict_idx[0] <= ict_start) and (ict_end <= eeg.ict_idx[1]):
                break

        ict = eeg.rec[:,ict_start:ict_end]
        print(ict)
        icts.append(ict)

    preicts = []
    for eeg in filelist:
        if eeg.num_szr > 0:
            while True:
                pre_start = np.random.randint(0, eeg.rec.shape[1] - streamlen)
                pre_end = pre_start + streamlen
                #print('(%d,%d)' % (pistreamst, pistreamend))
                if (eeg.pre_idx[0] <= pre_start) and (pre_end <= eeg.pre_idx[1]):
                    break

            preict = eeg.rec[:,pre_start:pre_end]
            print(pre)
            preicts.append(preict)

    normarray = np.zeros(len(norms), 23, streamlen)
    for j in range(normarray.shape[0]):
        normarray[j,:,:] = norms[j]
    ictarray = np.zeros(len(icts), 23, streamlen)
    for j in range(ictarray.shape[0]):
        ictarray[j,:,:] = icts[j]
    prearray = np.zeros(len(preicts), 23, streamlen)
    for j in range(prearray.shape[0]):
        prearray[j,:,:] = preicts[j]

    return normarray, ictarray, prearray
