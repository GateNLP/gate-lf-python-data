from __future__ import print_function
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *
import json
from io import open    # use with open("asas",'rt',encoding='utf-8')
# from collections import Counter, OrderedDict
# ?? from future.builtins.disabled import *

# from . import Feature


class Dataset(object):
    def __init__(self,metafile):
        self.metafile = metafile
        with open(metafile, "rt", encoding="utf-8") as inp:
            self.meta = json.load(inp)
        self.datafile = self.meta["dataFile"]

    def string_iter(self):
        class StringIterator(object):
            def __init__(self, datafile):
                self.datafile = datafile

            def __iter__(self):
                with open(self.datafile, "rt", encoding="utf=8") as inp:
                    for line in inp:
                        yield line
        return StringIterator(self.datafile)

    def __next__(self):
        pass

