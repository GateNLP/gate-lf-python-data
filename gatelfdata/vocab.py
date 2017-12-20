from __future__ import print_function
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *
import json
from io import open    # use with open("asas",'rt',encoding='utf-8')
from collections import Counter, OrderedDict, defaultdict
import re


class Vocab(object):
    """From the counter object, create string to id and id to string
    mappings. Has attributes stoi and itos"""
    def __init__(self,counts,max_size=None,min_freq=1,add_symbols=[]):
        """Create a vocabulary instance from the counts. If max_size is
        given sorts by frequency and only retains the max_size most frequent
        ones. Removes everything less the min_freq. Finally adds the
        strings in add_symbols. Raises an error if any of these symbols
        is already there. NOTE: if a padding symbol is added, it must
        be the first in the list! If no padding symbol is added, some
        arbitrary symbol will get assigned index 0!!"""
        self.freqs = Counter(counts)
        for s in add_symbols:
            if s in counts:
                raise Exception("Additional symbol already in counts:",s)
        self.min_freq = min_freq
        # got through the entries and put all the keys into a list
        self.itos = [s for s in self.freqs if self.freqs[s] >= min_freq]
        # sort the keys by frequency in reverse order
        self.itos = sorted(self.itos)
        self.itos = sorted(self.itos, reverse=True, key=lambda x: self.freqs[x])
        # add the additional symbols at the beginning
        self.itos = add_symbols + self.itos
        if max_size and len(self.itos) > max_size:
            self.itos = self.itos[:max_size]
        # now create the reverse map
        self.stoi = defaultdict(int)
        for i,s in enumerate(self.itos):
            self.stoi[s] = i
