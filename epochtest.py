from __future__ import print_function
import sys
import numpy as np
import chb

subject = chb.load_dataset(sys.argv[1], tiger=True)
sys.stdout.flush()
for sznum in range(1,subject.get_num() + 1):
    train, trainlab, test, testlab = chb.loo_epoch(subject, sznum)

    print('Seizure %d of %d' % (sznum, subject.get_num()))
    print(train.shape)
    print(trainlab.shape)
    print(sum(trainlab))
    print(test.shape)
    print(testlab.shape)
    print(sum(testlab))
    sys.stdout.flush()
