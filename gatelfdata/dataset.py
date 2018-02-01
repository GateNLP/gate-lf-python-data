from __future__ import print_function
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *
import json
import numpy as np
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
    def converted4meta(metafilename):
        """Given the path to a meta file, return the path to the converted data file"""
        return re.sub("\.meta\.json", ".converted.json", metafilename)

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
        self.isSequence = self.meta["isSequence"]
        if self.isSequence:
            self.maxSequenceLength = self.meta["sequLengths.max"]
            self.avgSequenceLength = self.meta["sequLengths.mean"]
        else:
            self.maxSequenceLength = None
            self.avgSequenceLength = None
        self.nFeatures = len(self.meta["featureInfo"]["attributes"])
        self.nInstances = int(self.meta["linesWritten"])
        target_is_string = self.meta["targetStats"]["isString"]
        if target_is_string:
            self.targetType = "nominal"
            self.targetClasses = list(self.meta["targetStats"]["stringCounts"].keys())
        else:
            self.targetType = "numeric"
            self.targetClasses = None
        self.convertedFile = None

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

    def instances_converted(self, convertedFile=None):
        if not convertedFile:
            convertedFile = self.convertedFile

        class ConvertedIterable(object):

            def __init__(self, convertedFile):
                self.convertedFile = convertedFile

            def __iter__(self):
                logger = logging.getLogger(__name__)
                with open(self.convertedFile, "rt") as inp:
                    for line in inp:
                        converted = json.loads(line)
                        logger.debug("Converted read: %r", converted)
                        yield converted
        return ConvertedIterable(convertedFile)

    def batches_converted(self, convertedFile=None, batch_size=100, as_numpy=False, pad_left=False):
        """Return a batch of instances for training. This reshapes the data in the following ways:
        For classification, the independent part is a list of batchsize values for each feature. So for
        a batch size of 100 and 18 features, the inputs are a list of 18 lists of 100 values each.
        If the feature itself is a sequence (i.e. comes from an ngram), then the list corresponding
        to that feature contains 100 lists.
        For sequence tagging, the independent part is a list of features, where each of the per-feature lists
        contains 100 (batch size) elements, and each of these elements is a list with as many elements
        as the corresponding sequence contains.
        """
        if not convertedFile:
            convertedFile = self.convertedFile

        class BatchIterable(object):

            def __init__(self, convertedFile, batchsize):
                self.convertedFile = convertedFile
                self.batchsize = batchsize

            def __iter__(self):
                logger = logging.getLogger(__name__)
                with open(self.convertedFile, "rt") as inp:
                    while True:
                        collect = []
                        eof = False
                        for i in range(self.batchsize):
                            line = inp.readline()
                            if line:
                                converted = json.loads(line)
                                ## TODO: properly collect
                                collect.append(converted)
                                logger.debug("Batch read: %r", converted)
                            else:
                                eof = True
                                break
                        # TODO: if necessary, convert the collected stuff to numpy, padding any sequences
                        # !! this should be done using a public method so the same method can be used to
                        # convert the validation set to numpy!
                        yield collect
                        if eof:
                            break
        return BatchIterable(convertedFile, batch_size)



    # TODO: return the validation set already re-shaped so it looks the same shape as a batch!
    # Also, allow to return it in numpy format with padding etc. Use the public method for converting from
    # one to the other for this!!
    def convert_to_file(self, outfile=None, return_validationset=True, validation_size=None, validation_part=0.1, random_seed=1):
        """Convert the whole data file to the given output file. If return_validationset is true, returns a list
        of converted instances for the validation set which are not written to the output file. If this is done,
        the size of the validation set as well as the random seed for selecting the instances can be specified.
        If validation_size is specified, it takes precedence over validation_part."""
        if not outfile:
            outfile = self.converted4meta(self.metafile)
        self.convertedFile = outfile
        logger = logging.getLogger(__name__)
        valinstances = []
        valindices = set()
        if return_validationset:
            if validation_size:
                valsize = int(validation_size)
            else:
                valsize = int(self.nInstances * validation_part)
            if valsize <= 1 or valsize > int(self.nInstances/ 2.0):
                raise Exception("Validation set size should not be less than 1 or more than half the data, but is %s (n=%s)" % (valsize, self.nInstances))
            # now get valsize integers from the range 0 to nInstances-1: these are the instance indices
            # we want to reserve for the validation set
            choices = np.random.choice(self.nInstances, size=valsize, replace=False)
            logger.debug("convert_to_file, nInst=%s, valsize=%s, choices=%s" % (self.nInstances, valsize, len(choices)))
            for choice in choices:
                valindices.add(choice)
        i = 0
        with open(outfile,"w") as out:
            for instance in self.instances_as_data():
                if i in valindices:
                    valinstances.append(instance)
                else:
                    print(json.dumps(instance), file=out)
                i += 1
        return valinstances

    def get_info(self):
        """Return a concise description of the learning problem that makes it easier to understand
        what is going on and what kind of network needs to get created."""
        ret = {}
        ret["isSequence"] = self.isSequence
        ret["maxSequenceLength"] = self.maxSequenceLength
        ret["avgSequenceLength"] = self.avgSequenceLength
        ret["nFeatures"] = self.nFeatures
        ret["nInstances"] = self.nInstances
        ret["targetType"] = self.targetType
        ret["targetClasses"] = self.targetClasses
        ret["features"] = self.features
        ret["target"] = self.target
        return ret

    def __str__(self):
        return "Dataset(meta=%s,isSeq=%s,nFeat=%s,N=%s)" % (self.metafile, self.isSequence, self.nFeatures, self.nInstances)

    def __repr__(self):
        return "Dataset(meta=%s,isSeq=%s,nFeat=%s,N=%s,features=%r,target=%r)" % (self.metafile, self.isSequence, self.nFeatures, self.nInstances, self.features, self.target)
