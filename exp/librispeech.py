from net import RecurrentNetwork, load_librispeech
import matplotlib.pyplot as plt

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_librispeech()
clf = RecurrentNetwork(learning_alg="sgd", hidden_layer_sizes=[500],
                       max_iter=50, cost="ctc", bidirectional=True,
                       learning_rate=0.01, momentum=0.99,
                       recurrent_activation="lstm",
                       random_seed=1999)
clf.fit(train_x[:2], train_y[:2])
from IPython import embed; embed()
