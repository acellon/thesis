
def load_data(filelist, VERBOSE=False):
    # Pickle everything individually to get around file size pickle bug...
    folder, _ = filelist[0].get('filename').split('_')
    dirname = PATH + folder + '/'
    for eegfile in filelist:
            filename = dirname + eegfile.get('filename')
            pklname = filename + '.pkl'
            if os.path.exists(pklname):
                if VERBOSE:
                    print('Loading: ' + pklname)
                eegfile['rec'] = pickle.load(open(pklname, 'rb'))
            else:
                if VERBOSE:
                    print('Pickling: ' + pklname)
                eegfile['rec'] = sio.loadmat(filename)['rec']
                pickle.dump(eegfile['rec'], open(pklname, 'wb')) #, protocol=4

    return filelist
