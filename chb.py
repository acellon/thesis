import numpy as np
import scipy.io as sio
import os
import re

'''
import theano
import theano.tensor as T

import lasagne
'''
PATH = '/Users/adamcellon/Drive/senior/thesis/data/'

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
                newfile = {'filename': fn.group(1)}
                f.readline(); f.readline();

                # Add number of seizures
                num_szr = re.match(r".*Seizures in File: (\d+)", f.readline())
                newfile['num_szr'] = int(num_szr.group(1))

                # If file includes seizures, add start and end times
                    # note: assume max 1 seizure per file
                if newfile.get('num_szr', 0) > 0:
                    start = re.match(r".*Start Time: (\d+) s", f.readline())
                    newfile['start'] = int(start.group(1))
                    end = re.match(r".*End Time: (\d+) s", f.readline())
                    newfile['end'] = int(end.group(1))

                # Add file metadata to filelist
                filelist.append(newfile)

    # Close summary file and return filelist
    f.closed
    return filelist

def load_data(filelist, VERBOSE=False, EXTHD=True):
    # Save/load arrays with
    folder, _ = filelist[0].get('filename').split('_')
    if EXTHD:
        varpath = '/Volumes/extHD/CHBMIT/'
    else:
        varpath = PATH
    dirname = varpath + folder + '/'
    savename =  dirname + folder + '.npz'

    if os.path.exists(savename):
        print('Loading:', savename)
        loaddict = np.load(savename)
        for eeg in filelist:
            eeg['rec'] = loaddict[eeg.get('filename')]
        print('Done.')
        return filelist
    else:
        savedict = {}
        if not VERBOSE:
            print('Converting mat files to np arrays...')

        for eeg in filelist:
            if VERBOSE:
                print('Converting %s.mat to np array' % eeg.get('filename'))

            eeg['rec'] = sio.loadmat(dirname + eeg.get('filename'))['rec']
            savedict[eeg.get('filename')] = eeg.get('rec')

        if VERBOSE:
            print('Saving and compressing...')
        np.savez_compressed(savename, **savedict)
        print('Done.')

    return filelist

def label(filelist, H=5):
    # Check to see if filelist contains rec data
    if filelist[0].get('rec') is None:
        print('No data has been loaded for this filelist. Please use load_data().')
        return filelist

    # Convert event horizon to sample size (minutes to 1/256 seconds)
        # need to decide what to do about event horizon going before start
    H = H * 60 * 256
    start, end = 0, 0
    for eegfile in filelist:
        ict = np.zeros_like(eegfile['rec'])
        preict = np.copy(ict)

        if eegfile['num_szr'] > 0:
            start = eegfile['start'] * 256
            end = eegfile['end'] * 256
            ict[:, start:end] = 1
            # print('ict indices: [',start,',',end,']',sep='')
            preict[:, (start - H):start] = 1
            # print('preict indices: [',start-H,',',start,']',sep='')

        eegfile['ict'] = ict
        eegfile['preict'] = preict

    return filelist
