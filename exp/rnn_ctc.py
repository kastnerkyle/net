import pickle
import theano
import numpy as np
from net import RecurrentCTC
import matplotlib.pyplot as plt

with open('data/data.pkl', 'rb') as pkl_file:
    data = pickle.load(pkl_file)

nClasses = data['nChars']
data_x, data_y = [], []
for x, y in zip(data['x'], data['y']):
    # Need to make alternate characters blanks (index as nClasses)
    y1 = [nClasses]
    for char in y:
        y1 += [char, nClasses]
    data_y.append(np.asarray(y1, dtype=np.int32))
    data_x.append(np.asarray(x, dtype=theano.config.floatX).T)

train_x = data_x[:750]
valid_x = data_x[750:]
train_y = data_y[:750]
valid_y = data_y[750:]

clf = RecurrentCTC(learning_alg="sgd", learning_rate=0.001,
                   activation="tanh", random_seed=1999)
clf.fit(train_x, train_y, valid_x, valid_y)
clf.print_alignment(valid_x[0])
plt.plot(clf.training_loss_, label="train")
plt.plot(clf.validation_loss_, label="valid", color="red")
plt.title("Loss")
plt.legend()
plt.show()
