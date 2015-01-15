from net import RecurrentNetwork, load_cmuarctic, labels_to_chars
import matplotlib.pyplot as plt
import numpy as np

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_cmuarctic()
clf = RecurrentNetwork(learning_alg="sfg",
                       hidden_layer_sizes=[500],
                       max_col_norm=1.9635,
                       max_iter=1000, cost="ctc", bidirectional=True,
                       learning_rate=0.0001, momentum=0.9,
                       recurrent_activation="lstm",
                       random_seed=1999)

tx = train_x[2]
tx = (tx - tx.mean()) / tx.std()
clf.fit(train_x[2], train_y[2])
y = labels_to_chars(train_y[2])
print(y)
