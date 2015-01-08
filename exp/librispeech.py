from net import RecurrentNetwork, load_librispeech
import matplotlib.pyplot as plt

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_librispeech()
clf = RecurrentNetwork(learning_alg="rmsprop", hidden_layer_sizes=[500, 500, 500],
                       max_iter=1000, cost="ctc", bidirectional=True,
                       learning_rate=0.1, momentum=0.9,
                       recurrent_activation="lstm",
                       random_seed=1999)
clf.fit(train_x[2], train_y[2])
from IPython import embed; embed()
