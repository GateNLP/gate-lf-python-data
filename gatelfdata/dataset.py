from builtins import *
import json
import numpy as np
from io import open    # use with open("asas",'rt',encoding='utf-8')
import re
import os
import logging
import sys
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

    def modified4meta(self, dir=None, name_part=None):
        """Helper method to construct the full path of one of the files this class creates from
        the original meta/data files. If dir is given, then it is the containing directory for the file.
        The name_part parameter specifies the part in the file name that will replace the "meta"
        in the original metafilename."""
        if not name_part:
            raise Exception("Parameter name_part must be specified")
        path, name = os.path.split(self.metafile)
        newname = re.sub("\.meta\.json", "."+name_part+".json", name)
        if dir:
            pathdir = dir
        else:
            pathdir = path
        newpath = os.path.join(pathdir, newname)
        return newpath



    @staticmethod
    def load_meta(metafile):
        """Static method for just reading and returning the meta data for this dataset."""
        with open(metafile, "rt", encoding="utf-8") as inp:
            return json.load(inp)

    def __init__(self, metafile):
        """Creating an instance will read the metadata and create the converters for
        converting the instances from the original data format (which contains the original
        values and strings) to a converted representation where strings are replaced by
        word indices related to a vocabulary."""
        self.metafile = metafile
        self.meta = Dataset.load_meta(metafile)
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
        self.have_conv_split = False
        self.have_orig_split = False
        self.outdir = None
        self.orig_train_file = None
        self.orig_val_file = None
        self.converted_train_file = None
        self.converted_val_file = None
        self.converted_data_file = None

    def instances_as_string(self, train=False, file=None):
        """Returns an iterable that allows to read the original instance data rows as a single string.
        That string can be converted into the actual original representation by parsing it as json.
        If train is set to True, then instead of the original data file, the train file created
        with the split() method is used. If file is not None, train is ignored and the file specified
        is read instead.
        """
        class StringIterable(object):
            def __init__(self, datafile):
                self.datafile = datafile

            def __iter__(self):
                with open(self.datafile, "rt", encoding="utf=8") as inp:
                    for line in inp:
                        yield line
        if file:
            whichfile = file
        else:
            if train:
                if not self.have_orig_split:
                    raise Exception("We do not have a train file in original format for instances_as_string")
                whichfile = self.orig_train_file
            else:
                whichfile = self.datafile
        return StringIterable(whichfile)

    def instances_original(self, train=False, file=None):
        """Returns an iterable that allows to read the instances from a file in original format.
        This file is the original data file by default, but could also the train file created with
        the split() method or any other file derived from the original data file."""
        class StringIterable(object):
            def __init__(self, datafile):
                self.datafile = datafile

            def __iter__(self):
                with open(self.datafile, "rt", encoding="utf=8") as inp:
                    for line in inp:
                        yield json.loads(line, encoding="UTF-8")
        if file:
            whichfile = file
        else:
            if train:
                if not self.have_orig_split:
                    raise Exception("We do not have a train file in original format for instances_as_string")
                whichfile = self.orig_train_file
            else:
                whichfile = self.datafile
        return StringIterable(whichfile)

    def normalize_features(self, indep, normalize="meanvar"):
        """This normalizes the converted features (indep) according to the giving normalization strategy.
        The features must correspond to the features described in the meta data, i.e. the indep parameter
        has to contain the exact independent part of an instance."""
        assert len(indep) == len(self.meta["features"])
        # TODO: We may also want to be able to support squashing functions and similar here.
        if normalize=="meanvar":
            for i in range(len(self.meta["features"])):
                if self.meta["features"][i]["datatype"] == "numeric":
                    # normalize it based on the feature stats
                    fname = self.meta["features"][i]["name"]
                    mean = self.meta["featureStats"][fname]["mean"]
                    var = self.meta["featureStats"][fname]["variance"]
                    # if var is > larger than 0.0 then do normalization by mapping the mean to 0
                    # and normalizing the variance to 1.0
                    if var > 0.0:
                        val = indep[i]
                        val = (val - mean)/var
                        indep[i] = val
        return indep

    def convert_indep(self, indep, normalize=None):
        """Convert the independent part of an original representation into the converted representation
        where strings are replaced by word indices or one hot vectors. If normalize is set to "meanvar", then
        normalization is performed based on the mean/variance statistics for the feature. If it is set to None,
        no normalization is performed.
        [NOT YET: if normalize is set to "config" then normalization is performed as configured for the feature]"""
        converted = self.features(indep)
        if normalize:
            converted = self.normalize_features(converted, normalize=normalize)
        return converted

    def convert_dep(self, dep):
        """Convert the dependent part of an original representation into the converted representation
        where strings are replaced by one hot vectors."""
        return self.target(dep)

    def convert_instance(self, instance, normalize="meanvar"):
        """Convert an original representation of an instance as read from json to the converted representation.
        This will also by default automatically normalize all numeric features, this can be changed by setting
        the normalize parameter (see convert_indep).
        Note: if the instance is a string, it is assumed it is still in json format and will get converted first."""
        if isinstance(instance, str):
            instance = json.loads(instance, encoding="utf=8")
        (indep, dep) = instance
        indep_converted = self.convert_indep(indep,normalize=normalize)
        dep_converted = self.convert_dep(dep)
        return [indep_converted, dep_converted]

    def split(self, outdir=None, validation_size=None, validation_part=0.1, random_seed=1, convert=False, keep_orig=False):
        """This splits the original file into an actual training file and a validation set file.
        This creates two new files in the same location as the original files, with the "data"/"meta"
        parts of the name replaced with "val" for validation and "train" for training. If converted is
        set to True, then instead of the original data, the converted data is getting split and
        saved, in that case the name parts are "converted.var" and "coverted.train". If keep_orig is
        set to True, then both the original and the converted format files are created. Depending on which
        format files are created, subsequent calls to the batches_converted or batches_orig can be made.
        If outdir is specified, the files will get stored in that directory instead of the directory
        where the meta/data files are stored.
        If random_seed is set to 0 or None, the random seed generator does not get initialized."""
        logger = logging.getLogger(__name__)
        valindices = set()
        if validation_size or validation_part:
            if random_seed:
                np.random.seed(random_seed)
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
            # print("DEBUG: valindices=%s" % valindices, file=sys.stderr)
        else:
            # we just keep the empy valindices set
            pass
        origtrainfile = self.modified4meta(name_part="orig.train", dir=outdir)
        origvalfile = self.modified4meta(name_part="orig.val", dir=outdir)
        convtrainfile = self.modified4meta(name_part="converted.train", dir=outdir)
        convvalfile = self.modified4meta(name_part="converted.val", dir=outdir)
        outorigtrain = None
        outorigval = None
        outconvtrain = None
        outconvval = None
        if not convert or (convert and keep_orig):
            self.have_orig_split = True
            outorigtrain = open(origtrainfile, "w", encoding="utf-8")
            self.orig_train_file = origtrainfile
            outorigval = open(origvalfile, "w", encoding="utf-8")
            self.orig_val_file = origvalfile
        if convert:
            self.have_conv_split = True
            outconvtrain = open(convtrainfile, "w", encoding="utf-8")
            self.converted_train_file = convtrainfile
            outconvval = open(convvalfile, "w", encoding="utf-8")
            self.converted_val_file = convvalfile
        # print("DEBUG origtrain/origval/convtrain/convval=%s/%s/%s/%s" % (outorigtrain, outorigval, outconvtrain, outconvval), file=sys.stderr)
        self.outdir = outdir
        i = 0
        for line in self.instances_as_string():
            if convert:
                converted = self.convert_instance(line)
            else:
                converted = None
            if i in valindices:
                if outorigval:
                    print(line, file=outorigval, end="")
                if outconvval:
                    print(converted, file=outconvval)
            else:
                if outorigtrain:
                    print(line, file=outorigtrain, end="")
                if outconvtrain:
                    print(converted, file=outconvtrain)
            i += 1
        if outorigtrain:
            outorigtrain.close()
        if outorigval:
            outorigval.close()
        if outconvtrain:
            outconvtrain.close()
        if outconvval:
            outconvval.close()

    def convert_to_file(self, outfile=None, infile=None):
        """Copy the whole data file (or if infile is not None, that file) to a converted version.
        The default file name is used if outfile is None, otherwise the file specified is used."""
        if not outfile:
            outfile = self.modified4meta(name_part="converted.data")
        self.converted_data_file = outfile
        logger = logging.getLogger(__name__)
        with open(outfile, "w") as out:
            for instance in self.instances_converted(train=False, convert=True):
                print(json.dumps(instance), file=out)

    def validation_set_orig(self):
        """Read and return the validation set rows in original format. For this to work, the split()
        method must have been run and either convert have been False or convert True and keep_orig True."""
        if not self.have_orig_split:
            raise Exception("We do not have the splitted original file, run the split method.")
        validationsetfile = self.modified4meta(name_part="orig.val", dir=self.outdir)
        valset = []
        with open(validationsetfile, "rt", encoding="utf-8") as inp:
            for line in inp:
                valset.append(json.loads(line))
        return valset

    def validation_set_converted(self, as_numpy=False, as_batch=False):
        """Read and return the validation set instances in converted format, optionally converted to
        batch format and if in batch format, optionally with numpy arrays. Fir this to work the split()
        method must have been run before with convert set to True."""
        if not self.have_conv_split:
            raise Exception("We do not have the splitted converted file, run the split method.")
        validationsetfile = self.modified4meta(name_part="converted.val", dir=self.outdir)
        valset = []
        with open(validationsetfile, "rt") as inp:
            for line in inp:
                valset.append(json.loads(line))
        if as_batch:
            valset = self.reshape_batch(valset, as_numpy=as_numpy)
        return valset

    def instances_converted(self, train=True, file=None, convert=False):
        """This reads instances and returns them in converted format. The instances are either
        read from a file in original format and converted on the fly (convert=True) or from a file
        that has already been converted as e.g. created with the split() or copy_to_converted() methods.
        If the file parameter is not None, then that file is read, otherwise if the train parameter is
        False then the original data file is read, otherwise if the train parameter is True, the
        train file is read."""
        class DataIterable(object):
            def __init__(self, meta, datafile, parent, convert):
                self.meta = meta
                self.datafile = datafile
                self.parent = parent

            def __iter__(self):
                logger = logging.getLogger(__name__)
                with open(self.datafile, "rt", encoding="utf=8") as inp:
                    for line in inp:
                        instance = json.loads(line)
                        logger.debug("Dataset read: instance=%r" % instance)
                        if convert:
                            yield self.parent.convert_instance(instance)
                        else:
                            yield instance
        if file:
            whichfile = file
        else:
            if train:
                if convert:
                    if not self.have_orig_split:
                        raise Exception("We do not have a train file in original format for instances_as_string")
                    whichfile = self.orig_train_file
                else:
                    if not self.have_conv_split:
                        raise Exception("We do not have a train file in converted format for instances_as_string")
                    whichfile = self.converted_train_file
            else:
                if convert:
                    whichfile = self.datafile
                else:
                    if self.converted_data_file:
                        whichfile = self.converted_data_file
                    else:
                        raise Exception("We do not have a data file in converted format for instance_as_string")
        return DataIterable(self.meta, whichfile, self, convert=convert)

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
        n_classes = self.nClasses
        is_sequence = self.isSequence
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
            if is_sequence:
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
                if n_classes == 0:
                    arr = np.array(targets, np.float_)
                elif is_sequence:
                    # sequence tagging, nominal classes: we need to create an array of
                    # shape batchsize, maxseq, nclasses and fill in the values either left or right padded
                    arr = np.zeros((batch_size, max_target_seq, n_classes), np.float_)
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

    def batches_original(self, train=True, file=None, reshape=True, batch_size=100, pad_left=False):
        """Return a batch of instances in original format for training.
        """
        class BatchOrigIterable(object):

            def __init__(self, thefile, batchsize, parent):
                self.file = thefile
                self.batchsize = batchsize
                self.parent = parent

            def __iter__(self):
                logger = logging.getLogger(__name__)
                with open(self.file, "rt", encoding="utf-8") as inp:
                    while True:
                        collect = []
                        eof = False
                        for i in range(self.batchsize):
                            line = inp.readline()
                            if line:
                                instance = json.loads(line)
                                collect.append(instance)
                                logger.debug("BatchOrig read: %r", instance)
                            else:
                                eof = True
                                break
                        if reshape:
                            batch = self.parent.reshape_batch(collect, pad_left=pad_left)
                        else:
                            batch = collect
                        yield batch
                        if eof:
                            break
        if file:
            whichfile = file
        else:
            if train:
                    if not self.have_orig_split:
                        raise Exception("We do not have a train file in converted format for batches_converted")
                    whichfile = self.orig_train_file
            else:
                    if self.converted_data_file:
                        whichfile = self.orig_data_file
                    else:
                        raise Exception("We do not have a data file in converted format for batches_converted")
        return BatchOrigIterable(whichfile, batch_size, self)

    def batches_converted(self, train=True, file=None, reshape=True, convert=False, batch_size=100, as_numpy=False, pad_left=False):
        """Return a batch of instances for training. If reshape is True, this reshapes the data in the following ways:
        For classification, the independent part is a list of batchsize values for each feature. So for
        a batch size of 100 and 18 features, the inputs are a list of 18 lists of 100 values each.
        If the feature itself is a sequence (i.e. comes from an ngram), then the list corresponding
        to that feature contains 100 lists.
        For sequence tagging, the independent part is a list of features, where each of the per-feature lists
        contains 100 (batch size) elements, and each of these elements is a list with as many elements
        as the corresponding sequence contains.
        If reshape is True (the default), then the batch gets reshaped using the reshape_batch method.
        """
        class BatchConvertedIterable(object):

            def __init__(self, convertedfile, batchsize, parent):
                self.convertedfile = convertedfile
                self.batchsize = batchsize
                self.parent = parent

            def __iter__(self):
                logger = logging.getLogger(__name__)
                with open(self.convertedfile, "rt", encoding="utf-8") as inp:
                    while True:
                        collect = []
                        eof = False
                        for i in range(self.batchsize):
                            line = inp.readline()
                            if line:
                                converted = json.loads(line)
                                if convert:
                                    converted = self.parent.convert_instance(converted)
                                collect.append(converted)
                                logger.debug("Batch read: %r", converted)
                            else:
                                eof = True
                                break
                        if reshape:
                            batch = self.parent.reshape_batch(collect, as_numpy=as_numpy, pad_left=pad_left)
                        else:
                            batch = collect
                        yield batch
                        if eof:
                            break
        if file:
            whichfile = file
        else:
            if train:
                if convert:
                    if not self.have_orig_split:
                        raise Exception("We do not have a train file in original format for batches_converted")
                    whichfile = self.orig_train_file
                else:
                    if not self.have_conv_split:
                        raise Exception("We do not have a train file in converted format for batches_converted")
                    whichfile = self.converted_train_file
            else:
                if convert:
                    whichfile = self.datafile
                else:
                    if self.converted_data_file:
                        whichfile = self.converted_data_file
                    else:
                        raise Exception("We do not have a data file in converted format for batches_converted")
        return BatchConvertedIterable(whichfile, batch_size, self)

    def get_info(self):
        """Return a concise description of the learning problem that makes it easier to understand
        what is going on and what kind of network needs to get created."""
        ret = {}
        ret["isSequence"] = self.isSequence
        ret["maxSequenceLength"] = self.maxSequenceLength
        ret["avgSequenceLength"] = self.avgSequenceLength
        ret["nAttrs"] = self.nAttrs
        ret["nFeatures"] = self.nFeatures
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
