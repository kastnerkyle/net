from net import RecurrentNetwork, load_fruitspeech, labels_to_chars
import matplotlib.pyplot as plt

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_fruitspeech()
clf = RecurrentNetwork(learning_alg="sgd",
                       hidden_layer_sizes=[500],
                       max_iter=10000, cost="ctc", bidirectional=True,
                       learning_rate=0.0001, momentum=0.9,
                       recurrent_activation="lstm",
                       random_seed=1999)

clf.fit(train_x[0], train_y[0])
y_hat = labels_to_chars(clf.predict(train_x[0])[0])
y = labels_to_chars(train_y[0])
print(y_hat)
print(y)
from IPython import embed; embed()
