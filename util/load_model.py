try:
    import cPickle
except ImportError:
    import pickle as cPickle
import sys

f = open(sys.argv[1], 'rb')
clf = cPickle.load(f)

from IPython import embed; embed()
