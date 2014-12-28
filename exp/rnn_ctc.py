from net import RecurrentNetwork, load_scribe
import matplotlib.pyplot as plt


(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_scribe()
clf = RecurrentNetwork(learning_alg="sgd", hidden_layer_sizes=[9],
                       cost="ctc", learning_rate=0.1, momentum=0.99,
                       recurrent_activation="lstm", random_seed=1999)
clf.fit(train_x, train_y, valid_x, valid_y)
clf.print_alignment(valid_x[0])
plt.plot(clf.training_loss_, label="train")
plt.plot(clf.validation_loss_, label="valid", color="red")
plt.title("Loss")
plt.legend()
plt.show()
