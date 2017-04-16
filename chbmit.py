import chb
import numpy as np
import scipy.io as sio
import os
import re
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PATH = '/Users/adamcellon/Drive/senior/thesis/data/'


class CHBsubj(list):
    def __init__(self):
        self.szr_num = 0
        self.seizures = []

    def add_file(self, CHBfile):
        self.append(CHBfile)
        self.szr_num += CHBfile.get_num()
        for seizure in CHBfile.ict_idx:
            self.seizures.append((CHBfile.get_name(), seizure))

    def get_file(self, filename):
        for CHBfile in self:
            if filename == CHBfile.get_name():
                return CHBfile

    def load_meta(self, folder):
        # def load_meta(folder, eventHorizon=5):
        '''
        Function to read/load metadata about list of EEG files from summary text
        file. Metadata includes filename, number of seizures, and seizure
        indices.

        Parameters
        ----------
        folder : string
            Name of folder (in format chbXX) corresponding to CHB-MIT subject.

        Returns
        -------
        filelist : list of CHBfile
            All CHBfiles in folder (NB: files do not include CHBfile.rec data)
        '''
        # Locate summary textfile
        dirname = PATH + folder
        #filename = dirname + '/' + folder + '-summary.txt'
        filename = dirname + '-summary.txt'

        # Create empty list of EEG files
        with open(filename, 'r') as f:
            for line in f:
                # Find file metadata and create dict for each file
                fn = re.match(r"File Name: (\w+).edf", line)
                if fn:
                    # Add filename and skip two lines
                    newfile = chb.CHBfile(fn.group(1))
                    if not folder == 'chb24':
                        f.readline()
                        f.readline()
                    # Add number of seizures
                    num_szr = int(
                        re.match(r".*Seizures in File: (\d+)", f.readline())
                        .group(1))
                    for i in range(num_szr):
                        start = re.match(r".*Start Time: *(\d+) s",
                                         f.readline())
                        start = int(start.group(1)) * 256
                        end = re.match(r".*End Time: *(\d+) s", f.readline())
                        end = int(end.group(1)) * 256
                        newfile.add_szr((start, end))
                        #newfile.add_szr((start, end), eventHorizon)

                    # Add file metadata to filelist
                    self.add_file(newfile)

        # Close summary file and return filelist
        f.closed
        return

    def load_data(self, verbose=False, exthd=True):
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
            If True, data loaded from external HD. If False, data loaded from
            PATH.

        Returns
        -------
        filelist : list of CHBfile
            All CHBfiles for single subject (now including rec data).
        '''
        timerstart = time.clock()
        folder, _ = self[0].get_name().split('_')
        if re.match(r"chb17.", folder):
            folder = 'chb17'
        if exthd:
            savename = '/Volumes/extHD/CHBMIT/' + folder + '/' + folder + '.npz'
        else:
            savename = PATH + folder + '.npz'

        if os.path.exists(savename):
            print('Loading:', savename)
            loaddict = np.load(savename)
            for eeg in self:
                eeg.add_rec(loaddict[eeg.get_name()])
            print('Done: %f seconds elapsed.' % (time.clock() - timerstart))
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
