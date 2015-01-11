import sys
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import matplotlib.pyplot as plt
from net import load_fruitspeech, labels_to_chars
import numpy as np
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_fruitspeech()

f = open(sys.argv[1])
clf = cPickle.load(f)

all_frames = np.vstack(train_x)
means = np.mean(all_frames, axis=0)
std = np.std(all_frames, axis=0)
for n, t in enumerate(train_x):
    train_x[n] = (t - means) / std

for n, v in enumerate(valid_x):
    valid_x[n] = (v - means) / std

for n, v in enumerate(valid_y):
    y = labels_to_chars(v)
    y_hat = labels_to_chars(clf.predict(valid_x[n])[0])
    print("Expected: %s, predicted: %s" % (y, y_hat))
