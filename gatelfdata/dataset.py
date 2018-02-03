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
        self.nAttrs = len(self.meta["featureInfo"]["attributes"])
        self.nFeatures = len(self.meta["features"])
        self.nInstances = int(self.meta["linesWritten"])
        target_is_string = self.meta["targetStats"]["isString"]
        if target_is_string:
            self.targetType = "nominal"
            self.targetClasses = list(self.meta["targetStats"]["stringCounts"].keys())
            self.nClasses = len(self.targetClasses)
        else:
            self.targetType = "numeric"
            self.targetClasses = None
            self.nClasses = 0
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
        # now normalize the numeric features, if necessary
        # TODO: ultimately this should depend on the individual settings for each feature, but for
        # now we simply always normalize all numeric features here
        assert len(indep_converted) == len(self.meta["features"])
        ## TODO: the whole normalization code should maybe get factored into separate methods,
        ## so we can do it separately and maybe also have several methods available.
        ## In addition to normalization, we may also want to be able to support squashing functions and similar here.
        for i in range(len(self.meta["features"])):
            if self.meta["features"][i]["datatype"] == "numeric":
                # normalize it based on the feature stats
                fName = self.meta["features"][i]["name"]
                mean = self.meta["featureStats"][fName]["mean"]
                var = self.meta["featureStats"][fName]["variance"]
                # if var is > larger than 0.0 then do normalization by mapping the mean to 0
                # and normalizing the variance to 1.0
                if var > 0.0:
                    val = indep_converted[i]
                    val = (val - mean)/var
                    indep_converted[i] = val
        return [indep_converted, dep_converted]

    def instances_as_data(self):
        class DataIterable(object):
            def __init__(self, meta, datafile, features, target, parent):
                self.meta = meta
                self.datafile = datafile
                self.features = features
                self.target = target
                self.parent = parent

            def __iter__(self):
                logger = logging.getLogger(__name__)
                with open(self.datafile, "rt", encoding="utf=8") as inp:
                    for line in inp:
                        instance = json.loads(line)
                        logger.debug("Dataset read: instance=%r" % instance)
                        yield self.parent.convert_instance(instance)
        return DataIterable(self.meta, self.datafile, self.features, self.target, self)

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

    def reshape_batch(self, instances, as_numpy=False, pad_left=False):
        """Reshape the list of converted instances into what is expected for training on a batch.
        NOTE: this only works for cases for now where we do not have nested sequences!!!!!
        """
        logger = logging.getLogger(__name__)
        # instances is just a list of instances, where each instance is the format of "converted instances",
        # which consists of two sub lists for the independent and dependent part.
        batch_size = len(instances)
        # we will create as many lists for the independent part as we have features
        features_list = []
        targets = []
        max_seq_lengths = [0 for i in range(self.nAttrs)]
        max_target_seq = 0
        nClasses = self.nClasses
        isSequence = self.isSequence
        for i in range(self.nAttrs):
            features_list.append([])
        # we now got a list of empty lists, one empty list for each feature, now put the values
        # of each of the features in the independent part in there.
        for instance in instances:
            (indep, dep) = instance
            # print("DEBUG: len(indep)=%r, nFeature=%r" % (len(indep), self.nFeatures))
            assert len(indep) == self.nFeatures
            for i in range(self.nAttrs):
                fv = indep[i]
                if isinstance(fv, list):
                    l = len(fv)
                    if l > max_seq_lengths[i]:
                        max_seq_lengths[i] = l
                features_list[i].append(fv)
            targets.append(dep)
            if isSequence:
                l = len(dep)
                if l > max_target_seq:
                    max_target_seq = l
        logger.debug("reshape_batch: max_seq_lengths=%r" % max_seq_lengths)
        if as_numpy:
            # convert each feature and also the targets to numpy arrays of the correct shape
            # We start with a list of nFeatures features, each represented as a list
            # if that list contains itself lists, i.e. max_seq_lengths for it is > 0,
            # then convert that list of lists into a numpy matrix
            for i in range(self.nAttrs):
                if max_seq_lengths[i] > 0:
                    # this feature is represented as batchsize sublists
                    values = features_list[i]
                    # we check the type of the first element to figure out if we need floats or ints
                    tmpval = values[0][0]
                    maxlen = max_seq_lengths[i]
                    if isinstance(tmpval,int):
                        arr = np.zeros((batch_size, maxlen), np.int_)
                    else:
                        arr = np.zeros((batch_size, maxlen), np.float_)
                    for j in range(batch_size):
                        value = values[j]
                        if pad_left:
                            arr[j, -len(value):] = value
                        else:
                            arr[j, :len(value)] = value
                    # replace by numpy
                    features_list[i] = arr
                else:
                    # we just have batchsize values
                    arr = np.array(features_list[i])
                    features_list[i] = arr
            # also convert the targets to numpy, if requested
            # for each instance in the batch, a target is one of the following:
            # - a one-hot list, indicating a nominal class
            # - a list of one-hot lists, in case of sequence tagging
            # - (not yet:) a numeric target, a single float value
            if as_numpy:
                if nClasses == 0:
                    arr = np.array(targets, np.float_)
                elif isSequence:
                    # sequence tagging, nominal classes: we need to create an array of
                    # shape batchsize, maxseq, nclasses and fill in the values either left or right padded
                    arr = np.zeros((batch_size, max_target_seq, nClasses), np.float_)
                    # fill in the values for each instance
                    for j in range(batch_size):
                        v = targets[j]
                        if pad_left:
                            arr[j, -len(v):] = v
                        else:
                            arr[j, :len(v)] = v
                else:
                    # classification, nominal classes: we need an array of shape batchsize, nclasses
                    arr = np.array(targets, np.float_)
                targets = arr
        ret = (features_list, targets)
        return ret


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

            def __init__(self, convertedFile, batchsize, parent):
                self.convertedFile = convertedFile
                self.batchsize = batchsize
                self.parent = parent

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
                                collect.append(converted)
                                logger.debug("Batch read: %r", converted)
                            else:
                                eof = True
                                break
                        batch = self.parent.reshape_batch(collect, as_numpy=as_numpy, pad_left=pad_left)
                        yield batch
                        if eof:
                            break
        return BatchIterable(convertedFile, batch_size, self)



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
        ret["nAttrs"] = self.nAttrs
        ret["nfeatures"] = self.nFeatures
        ret["nInstances"] = self.nInstances
        ret["targetType"] = self.targetType
        ret["nClasses"] = self.nClasses
        ret["targetClasses"] = self.targetClasses
        ret["features"] = self.features
        ret["target"] = self.target
        return ret

    def __str__(self):
        return "Dataset(meta=%s,isSeq=%s,nFeat=%s,N=%s)" % (self.metafile, self.isSequence, self.nAttrs, self.nInstances)

    def __repr__(self):
        return "Dataset(meta=%s,isSeq=%s,nFeat=%s,N=%s,features=%r,target=%r)" % (self.metafile, self.isSequence, self.nAttrs, self.nInstances, self.features, self.target)
