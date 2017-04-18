import chb
import os
import re
import sys
import time

# TODO: fix file nav on tigress serrver - right now the navigation
#       doesn't reflect how files are actually organized

def main(subject=0, pathchoice='exthd', VERBOSE=False):
    start = time.clock()
    tiger, exthd = False, False
    if pathchoice == 'local':
        pth = '/Users/adamcellon/Drive/senior/thesis/data/'
    elif pathchoice == 'tiger':
        pth = '/tigress/acellon/'
        tiger = True
    else:
        pth = '/Volumes/extHD/CHBMIT/'
        exthd = True

    for dirname, dirnames, filenames in os.walk(pth):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            num = int(re.match(r"chb(\d+)", subdirname).group(1))
            if (subject == num) or not subject:
                subdir = os.path.join(dirname, subdirname)
                npzfile = subdir + '/' + subdirname + '.npz'
                if os.path.exists(npzfile):
                    print('%s has already been loaded.' %
                          (subdirname + '.npz'))
                else:
                    print('Loading %s' % subdirname)
                    savedict = {}
                    for matfile in os.walk(subdirname)[2]:
                        print('Converting %s to np array' % matfile)

                        matname, _ = matfile.split('.')
                        savedict[matname] = sio.loadmat(dirname + matname)['rec']

                    print('Saving and compressing...')
                    np.savez_compressed(savename, **savedict)
                    print('Done: %f seconds elapsed.' % (time.clock() - timerstart))




if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Converts CHBMIT mat files and saves them as .npz.")
        print("Usage: %s [SUBJECT [VERBOSE]]" % sys.argv[0])
        print("     SUBJECT: subject folder to convert and save \
              (type: int, default: all).")
        print("     VERBOSE: bool to modify text output (default: False).")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['subject'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['pathvar'] = sys.argv[2]
        if len(sys.argv) > 3:
            kwargs['VERBOSE'] = sys.argv[3]
        main(**kwargs)
