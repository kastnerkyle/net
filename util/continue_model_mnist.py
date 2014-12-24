import cPickle
import time
import os
from net import load_data

datasets = load_data('mnist.pkl.gz')
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

f = open('model.save', 'rb')
classifier = cPickle.load(f)
print '... training'
start_time = time.clock()
classifier.fit(train_set_x, train_set_y, valid_set_x, valid_set_y)
end_time = time.clock()
print('The code for file ' + os.path.split(__file__)[1] +
      ' ran for %.2fm' % ((end_time - start_time) / 60.))
