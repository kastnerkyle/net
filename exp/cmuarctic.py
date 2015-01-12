from net import RecurrentNetwork, load_cmuarctic
import matplotlib.pyplot as plt
import numpy as np

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_cmuarctic()
raise ValueError()
clf = RecurrentNetwork(learning_alg="rmsprop",
                       hidden_layer_sizes=[500], max_col_norm=.75,
                       max_iter=100, cost="ctc", bidirectional=True,
                       learning_rate=0.00002, momentum=0.9,
                       recurrent_activation="lstm",
                       random_seed=1999)

all_frames = np.vstack(train_x)
means = np.mean(all_frames, axis=0)
std = np.std(all_frames, axis=0)
for n, t in enumerate(train_x):
    train_x[n] = (t - means) / std

for n, v in enumerate(valid_x):
    valid_x[n] = (v - means) / std
clf.fit(train_x, train_y, valid_x, valid_y)
y_hat = labels_to_chars(clf.predict(valid_x[0])[0])
y = labels_to_chars(valid_y[0])
print(y_hat)
print(y)
