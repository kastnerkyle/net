import cPickle
from net import load_data
import sys

datasets = load_data('mnist.pkl.gz')
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

f = open(sys.argv[1], 'rb')
classifier = cPickle.load(f)

from IPython import embed; embed()
