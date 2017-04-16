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
            pickle.dump(eegfile.rec, open(pklname, 'wb'))  #, protocol=4

    return filelist


def label(filelist, H=5):
    # Check to see if filelist contains rec data
    if filelist[0].rec is None:
        print(
            'No data has been loaded for this filelist. Please use chb.load_data().'
        )
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


def make_epochs(filelist):
    epochlen = 256 * 1
    num_epochs = 100

    for eeg in filelist:
        for j in range(eeg.get_num()):
            pass

    norms = []
    for n in range(numstream):
        while True:
            eeg = np.random.choice(filelist)
            norm_start = np.random.randint(0,
                                           eeg.get_rec().shape[1] - streamlen)
            norm_end = norm_start + streamlen
            #print('(%d,%d)' % (streamst, streamend))
            if not ((eeg.pre_idx[0] <= norm_start <= eeg.ict_idx[1]) or
                    (eeg.pre_idx[0] <= norm_end <= eeg.ict_idx[1])):
                break

        norm = eeg.get_rec()[:, norm_start:norm_end]
        #print(norm)
        norms.append(norm)

    icts = []
    for n in range(numstream):
        while True:
            eeg = np.random.choice(filelist)
            if eeg.get_num() == 0:
                continue
            ict_start = np.random.randint(0,
                                          eeg.get_rec().shape[1] - streamlen)
            ict_end = ict_start + streamlen
            #print('(%d,%d)' % (ictstreamst, ictstreamend))
            if (eeg.ict_idx[0] <= ict_start) and (ict_end <= eeg.ict_idx[1]):
                break

        ict = eeg.get_rec()[:, ict_start:ict_end]
        #print(ict)
        icts.append(ict)

    preicts = []
    for eeg in filelist:
        if eeg.get_num() > 0:
            while True:
                pre_start = np.random.randint(
                    0, eeg.get_rec().shape[1] - streamlen)
                pre_end = pre_start + streamlen
                #print('(%d,%d)' % (pistreamst, pistreamend))
                if (eeg.pre_idx[0] <= pre_start) and (
                        pre_end <= eeg.pre_idx[1]):
                    break

            preict = eeg.get_rec()[:, pre_start:pre_end]
            #print(pre)
            preicts.append(preict)

    normarray = np.zeros(len(norms), 23, streamlen)
    for j in range(normarray.shape[0]):
        normarray[j, :, :] = norms[j]
    ictarray = np.zeros(len(icts), 23, streamlen)
    for j in range(ictarray.shape[0]):
        ictarray[j, :, :] = icts[j]
    prearray = np.zeros(len(preicts), 23, streamlen)
    for j in range(prearray.shape[0]):
        prearray[j, :, :] = preicts[j]

    return normarray, ictarray, prearray
