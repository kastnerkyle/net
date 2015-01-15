from net import RecurrentNetwork
import numpy as np
import matplotlib.pyplot as plt

# Test adapted from Mohammad P
# https://github.com/mohammadpz/Recurrent-Neural-Networks

n_u = 2
n_y = 3
time_steps = 10
n_seq = 100
# n_y is equal to the number of calsses
random_state = np.random.RandomState(1999)

seq = random_state.randn(n_seq, time_steps, n_u)
targets = np.zeros((n_seq, time_steps), dtype=np.int32)

thresh = 0.5
targets[:, 2:][seq[:, 1:-1, 1] > seq[:, :-2, 0] + thresh] = 1
targets[:, 2:][seq[:, 1:-1, 1] < seq[:, :-2, 0] - thresh] = 2

clf = RecurrentNetwork(learning_alg="sgd", hidden_layer_sizes=[6, 6],
                       max_iter=1E3, cost="softmax", learning_rate=0.1,
                       momentum=0.99, recurrent_activation="lstm",
                       random_seed=1999)

clf.fit(seq, targets)

plt.close('all')
fig = plt.figure()
plt.grid()
ax1 = plt.subplot(211)

plt.scatter(np.arange(time_steps), targets[1], marker='o', c='b')
plt.grid()

guess = clf.predict_proba(seq[1])
guessed_probs = plt.imshow(guess[0].T, interpolation='nearest', cmap='gray')
ax1.set_title('blue points: true class, grayscale: model output (white mean class)')

ax2 = plt.subplot(212)
plt.plot(clf.training_loss_)
plt.grid()
ax2.set_title('Training loss')
plt.show()
