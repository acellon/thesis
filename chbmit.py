import chb
import numpy as np
import scipy.io as sio
import os
import re
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class CHBsubject(list):

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
            if filename is CHBfile.get_name():
                return CHBfile
