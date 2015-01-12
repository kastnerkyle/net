from net import RecurrentNetwork, load_librispeech, labels_to_chars
import numpy as np
import matplotlib.pyplot as plt

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_librispeech()
clf = RecurrentNetwork(learning_alg="rmsprop", hidden_layer_sizes=[1000, 1000, 1000, 1000, 1000],
                       max_iter=100, cost="ctc", bidirectional=True,
                       learning_rate=0.00002, momentum=0.9,
                       recurrent_activation="lstm",
                       random_seed=1999)
print(labels_to_chars(train_y[2]))
means = np.mean(train_x[2], axis=0)
std = np.std(train_x[2], axis=0)
tx = (train_x[2] - means) / std
clf.fit(tx, train_y[2])
from IPython import embed; embed()
