import gensim
import sys
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(sys.argv[1])
