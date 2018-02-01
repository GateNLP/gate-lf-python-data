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
import sys
import logging

from .features import Features
from .target import Target


class Dataset(object):
    """Class representing training data present in the meta and data files.
    After creating the Dataset instance, the attribute .meta contains the loaded metadata.
    Then, the instances_as_string and instances_as_data methods can be used to return an
    iterable."""

    @staticmethod
    def data4meta(metafilename):
        """Given the path to a meta file, return the path to a data file"""
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
        self.features = Features(self.meta)
        self.target = Target.make(self.meta)

    def instances_as_string(self):
        class StringIterable(object):
            def __init__(self, datafile):
                self.datafile = datafile

            def __iter__(self):
                with open(self.datafile, "rt", encoding="utf=8") as inp:
                    for line in inp:
                        yield line
        return StringIterable(self.datafile)

    def convert_indep(self, indep):
        return self.features(indep)

    def convert_dep(self, dep):
        return self.target(dep)

    def convert_instance(self, instance):
        """Convert a list representation as read from json to the final representation"""
        (indep, dep) = instance
        indep_converted = self.convert_indep(indep)
        dep_converted = self.convert_dep(dep)
        return [indep_converted, dep_converted]

    def instances_as_data(self):
        class DataIterable(object):
            def __init__(self, meta, datafile, features, target):
                self.meta = meta
                self.datafile = datafile
                self.features = features
                self.target = target

            def __iter__(self):
                logger = logging.getLogger(__name__)
                with open(self.datafile, "rt", encoding="utf=8") as inp:
                    for line in inp:
                        (indep, dep) = json.loads(line)
                        logger.debug("Dataset read: indep/dep=%r/%r", indep, dep)
                        yield [self.features(indep), self.target(dep)]
        return DataIterable(self.meta, self.datafile, self.features, self.target)

    def pp_meta(self, file=sys.stdout):
        """Produce a nice and pretty print out of the meta information"""
        print("Dataset.pp_meta: NOT YET IMPLEMENTED")

