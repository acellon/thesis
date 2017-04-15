import chb
import os
import re
import sys

PATH = '/Users/adamcellon/Drive/senior/thesis/data/'
EXTHD = '/Volumes/extHD/CHBMIT/'


def main(subject=0, VERBOSE=False):
    for dirname, dirnames, filenames in os.walk(EXTHD):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            num = int(re.match(r"chb(\d+)", subdirname).group(1))
            if (subject == num) or not subject:
                subdir = os.path.join(dirname, subdirname)
                npzfile = subdir + '/' + subdirname + '.npz'
                if os.path.exists(npzfile):
                    print('%s has already been loaded.'
                          % (subdirname + '.npz'))
                else:
                    print('Loading %s' % subdirname)
                    filelist = chb.load_meta(subdirname)
                    data = chb.load_data(filelist, VERBOSE)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Converts CHBMIT mat files and saves them as .npz.")
        print("Usage: %s [SUBJECT [VERBOSE]]" % sys.argv[0])
        print("     SUBJECT: subject folder to convert and save (type: int, default: all).")
        print("     VERBOSE: bool to modify text output (default: False).")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['subject'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['VERBOSE'] = sys.argv[2]
        main(**kwargs)
