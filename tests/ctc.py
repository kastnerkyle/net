from net import RecurrentNetwork, load_scribe
import matplotlib.pyplot as plt

(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_scribe()
clf = RecurrentNetwork(learning_alg="sgd", hidden_layer_sizes=[9], max_iter=100,
                       cost="ctc", bidirectional=True, learning_rate=0.1,
                       momentum=0.99, recurrent_activation="lstm",
                       random_seed=1999)
clf.fit(train_x, train_y)

plt.imshow(train_x[0].T, interpolation='nearest', cmap='gray')
plt.title("Ground truth alignment")
plt.figure()
y_pred = clf.predict_proba(train_x[0])
# Ignore "blanks"
plt.imshow(y_pred[0].T[:-1], interpolation='nearest', cmap='gray')
plt.title("Estimated alignment")
plt.show()
