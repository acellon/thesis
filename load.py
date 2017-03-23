import chb
import os

PATH = '/Users/adamcellon/Drive/senior/thesis/data/'
EXTHD = '/Volumes/extHD/CHBMIT/'

for dirname, dirnames, filenames in os.walk(EXTHD):
    # print path to all subdirectories first.
    for subdirname in dirnames:
        subdir = os.path.join(dirname, subdirname)
        npzfile = subdir + '/' + subdirname + '.npz'
        if os.path.exists(npzfile):
            print('%s has already been loaded.' % (subdirname +
            '.npz'))
        else:
            check = input('Press "y" to load %s: ' % (subdirname +
            '.npz'))
            if check == 'y':
                summary = chb.summary(subdirname)
                data = chb.load_data(summary)
