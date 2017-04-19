from __future__ import print_function

import sys
import os

import numpy as np

import chb

subjname = sys.argv[1]

subject = chb.load_dataset(subjname, tiger=True)
sys.stdout.flush()
print(subjname, 'rec shape:\n')
for eeg in subject:
    print(eeg.get_rec().shape)
    sys.stdout.flush()
print('-' * 80)
print()
sys.stdout.flush()
