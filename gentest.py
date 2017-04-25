from __future__ import print_function
import sys
import numpy as np
import chb

subject = chb.load_dataset(sys.argv[1], tiger=True)
sys.stdout.flush()
for (inputs, targets) in chb.epoch_gen(subject):
    print(inputs.shape)
    print(targets.shape)
    print(sum(targets))
    sys.stdout.flush()
