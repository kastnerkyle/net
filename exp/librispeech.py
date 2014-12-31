from net import RecurrentNetwork, load_librispeech
import matplotlib.pyplot as plt

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_librispeech()
clf = RecurrentNetwork(learning_alg="sgd", hidden_layer_sizes=[100],
                       max_iter=100, cost="ctc", bidirectional=True,
                       learning_rate=0.1, momentum=0.99,
                       recurrent_activation="lstm",
                       random_seed=1999)
clf.fit(train_x[:2], train_y[:2])

fig = plt.figure()
ax1 = plt.subplot(211)
plt.imshow(train_x[0].T, interpolation='nearest', cmap='gray')
plt.title("Ground truth")
y_pred = clf.predict_proba(train_x[0])
ax2 = plt.subplot(212)
# Ignore "blanks"
plt.imshow(y_pred[0].T[:-1], interpolation='nearest', cmap='gray')
plt.title("Estimated")
plt.show()
from IPython import embed; embed()
