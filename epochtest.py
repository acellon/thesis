from __future__ import print_function

import numpy as np
import chb

subject = chb.load_dataset('chb01', tiger=True)
train, trainlab = chb.make_epoch(subject)

print(train)
print(trainlab)
