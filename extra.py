
def load_data(filelist, VERBOSE=False):
    # Pickle everything individually to get around file size pickle bug...
    folder, _ = filelist[0].get('filename').split('_')
    dirname = PATH + folder + '/'
    for eegfile in filelist:
            filename = dirname + eegfile.name
            pklname = filename + '.pkl'
            if os.path.exists(pklname):
                if VERBOSE:
                    print('Loading: ' + pklname)
                eegfile.rec = pickle.load(open(pklname, 'rb'))
            else:
                if VERBOSE:
                    print('Pickling: ' + pklname)
                eegfile.rec = sio.loadmat(filename)['rec']
                pickle.dump(eegfile.rec, open(pklname, 'wb')) #, protocol=4

    return filelist

def label(filelist, H=5):
    # Check to see if filelist contains rec data
    if filelist[0].rec is None:
        print('No data has been loaded for this filelist. Please use chb.load_data().')
        return filelist

    # Convert event horizon to sample size (minutes to 1/256 seconds)
        # TODO: decide what to do about event horizon going before start
    H = H * 60 * 256
    start, end = 0, 0
    for eegfile in filelist:
        ict = np.zeros_like(eegfile.rec)
        preict = np.copy(ict)

        for i in range(eegfile.num_szr):
            start = eegfile.start[i]
            end = eegfile.end[i]
            ict[:, start:end] = 1
            preict[:, (start - H):start] = 1

        eegfile.ict = ict
        eegfile.preict = preict

    return filelist
