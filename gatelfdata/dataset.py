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
import re

from .features import Features
from .target import Target

# from . import Feature


class Dataset(object):

    @staticmethod
    def data4meta(metafilename):
        "Given the path to a meta file, return the path to a data file"
        return re.sub("\.meta\.json", ".data.json", metafilename)

    @staticmethod
    def load_meta(metafile):
        with open(metafile, "rt", encoding="utf-8") as inp:
            return json.load(inp)

    def __init__(self, metafile):
        self.metafile = metafile
        with open(metafile, "rt", encoding="utf-8") as inp:
            self.meta = json.load(inp)
        # we do not use the dataFile field because this will be invalid
        # if the files have been moved from their original location
        # self.datafile = self.meta["dataFile"]
        self.datafile = Dataset.data4meta(metafile)

    def instances_as_string(self):
        class StringIterable(object):
            def __init__(self, datafile):
                self.datafile = datafile

            def __iter__(self):
                with open(self.datafile, "rt", encoding="utf=8") as inp:
                    for line in inp:
                        yield line
        return StringIterable(self.datafile)

    def instances_as_data(self):
        class DataIterable(object):
            def __init__(self, meta, datafile):
                self.meta = meta
                self.datafile = datafile
                self.meta = meta
                self.features = Features(meta)
                self.target = Target.make(meta)

            def __iter__(self):
                with open(self.datafile, "rt", encoding="utf=8") as inp:
                    for line in inp:
                        (indep,dep) = json.loads(line)
                        # a list of lists
                        indep_converted = self.features(indep)
                        dep_converted = self.target(dep)
                        yield [indep_converted, dep_converted]
        return DataIterable(self.meta, self.datafile)



    def __next__(self):
        pass

