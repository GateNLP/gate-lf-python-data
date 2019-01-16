"""Module for the Dataset class"""

import json
import numpy as np
from io import open    # use with open("asas",'rt',encoding='utf-8')
import re
import os
import logging
from gatelfdata.features import Features
from gatelfdata.target import Target
from gatelfdata.vocabs import Vocabs
import sys

from gatelfdata.lib.dataset import ShuffledDataset, LineTsvDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class Dataset(object):
    """Class representing training data present in the meta and data files.
    After creating the Dataset instance, the attribute .meta contains the loaded metadata.
    Then, the instances_as_string and instances_as_data methods can be used to return an
    iterable."""

    @staticmethod
    def data4meta(metafilename):
        """Given the path to a meta file, return the path to a data file"""
        return re.sub(r'\.meta\.json', ".data.json", metafilename)

    @staticmethod
    def _modified4meta(metafile, name_part=None, dirname=None):
        if not name_part:
            raise Exception("Parameter name_part must be specified")
        path, name = os.path.split(metafile)
        newname = re.sub(r'\.meta\.json', '.'+name_part+'.json', name)
        if dirname:
            pathdir = dirname
        else:
            pathdir = path
        newpath = os.path.join(pathdir, newname)
        return newpath

    def modified4meta(self, name_part=None, dirname=None):
        """Helper method to construct the full path of one of the files this class creates from
        the original meta/data files. If dir is given, then it is the containing directory for the file.
        The name_part parameter specifies the part in the file name that will replace the "meta"
        in the original metafilename."""
        return Dataset._modified4meta(self.metafile, name_part=name_part, dirname=dirname)

    @staticmethod
    def load_meta(metafile):
        """Static method for just reading and returning the meta data for this dataset."""
        with open(metafile, "rt", encoding="utf-8") as inp:
            return json.load(inp)

    def __init__(self, metafile, reuse_files=False, config={}, targets_need_padding=False):
        """Creating an instance will read the metadata and create the converters for
        converting the instances from the original data format (which contains the original
        values and strings) to a converted representation where strings are replaced by
        word indices related to a vocabulary.
        If reuse_files is True, then any files found that look like training or validation files
        in the same directory are re-used, otherwise the split or convert_to_file methods
        must be run to re-create them before they can be used.
        The config parameter is expected to be a map with config settings. These settings
        override all other settings with highest priority. Currently the config parameter
        can only take the following settings:
        * embs=id:dims:train:minfreq,id:dims:drain:minfreq - a list of settings each for some id,
        specifying the dimensions, train mode, and minimum frequency.
        """
        self.config = config
        self.seed = self.config.get("seed", 0)
        # print("DEBUG creating dataset from ", metafile, "config is", config, file=sys.stderr)
        remove_embs = config.get("remove_embs", True)
        remove_counts = config.get("remove_counts", True)
        self.vocabs = Vocabs(remove_embs=remove_embs, remove_counts=remove_counts)
        self.metafile = metafile
        self.meta = Dataset.load_meta(metafile)
        # we do not use the dataFile field because this will be invalid
        # if the files have been moved from their original location
        # self.datafile = self.meta["dataFile"]
        self.orig_data_file = Dataset.data4meta(metafile)
        # create the indexed line dataset wrapper and shuffled dataset wrappers
        self.line_dataset = LineTsvDataset(self.orig_data_file)
        self.shuffled_dataset = ShuffledDataset(self.line_dataset)
        # override meta settings for the embeddings
        if config and "embs" in config and config["embs"] is not None:
            embs = config["embs"]
            embs_settings = embs.split(",")
            sdict = {}
            for setting in embs_settings:
                if ":" not in setting:
                    raise Exception("No colon in emb-setting, should be of the form id:dim:train:minfrequ %s" % (setting, ))
                (embid, embdims, embtrain, embminfreq, embfile) = (setting.split(":") + [""]*4)[:5]
                tmpsetting = {}
                if embdims:
                    tmpsetting["emb_dims"] = int(embdims)
                if embtrain:
                    tmpsetting["emb_train"] = embtrain
                if embminfreq:
                    tmpsetting["emb_minfreq"] = int(embminfreq)
                if embfile:
                    tmpsetting["emb_file"] = embfile
                tmpsetting["emb_id"] = embid
                sdict[embid] = tmpsetting
            for attrinfo in self.meta.get("featureInfo").get("attributes"):
                attr_eid = attrinfo.get("emb_id")
                if attr_eid is not None:
                    osetting = sdict.get(attr_eid)
                    if osetting:
                        for k, v in osetting.items():
                            if k.startswith("emb_"):
                                attrinfo[k] = v

        self.features = Features(self.meta, self.vocabs)
        self.target = Target.make(self.meta, self.vocabs, targets_need_padding=targets_need_padding)
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
        self.outdir = None
        self.orig_train_file = None
        self.orig_val_file = None
        self.have_conv_split = False
        self.have_orig_split = False
        self.converted_train_file = None
        self.converted_val_file = None
        self.converted_data_file = None
        if reuse_files:
            if os.path.exists(Dataset._modified4meta(self.metafile, name_part="orig.train")):
                if os.path.exists(Dataset._modified4meta(self.metafile, name_part="orig.val")):
                    self.have_orig_split = True
                    self.orig_train_file = Dataset._modified4meta(self.metafile, name_part="orig.train")
                    self.orig_val_file = Dataset._modified4meta(self.metafile, name_part="orig.val")
                    logger.debug("Found and using original train/validation files %s/%s" %
                                (self.orig_train_file, self.orig_val_file))
            if os.path.exists(Dataset._modified4meta(self.metafile, name_part="converted.train")):
                if os.path.exists(Dataset._modified4meta(self.metafile, name_part="converted.val")):
                    self.have_conv_split = True
                    self.converted_train_file = Dataset._modified4meta(self.metafile, name_part="converted.train")
                    self.converted_val_file = Dataset._modified4meta(self.metafile, name_part="converted.val")
                    logger.debug("Found and using converted train/validation files %s/%s" %
                                (self.converted_train_file, self.converted_val_file))
            if os.path.exists(Dataset._modified4meta(self.metafile, name_part="converted.data")):
                self.converted_data_file = Dataset._modified4meta(self.metafile, name_part="converted.data")
                logger.debug("Found and using converted data file %s" % self.converted_data_file)
        # private fields
        self._have_feature_idxs = False
        self._index_feature_idxs = None
        self._float_feature_idxs = None
        self._indexlist_feature_idxs = None
        self._index_features = None
        self._float_features = None
        self._indexlist_features = None

    def instances_as_string(self, train=False, file=None):
        """Returns an iterable that allows to read the original instance data rows as a single string.
        That string can be converted into the actual original representation by parsing it as json.
        If train is set to True, then instead of the original data file, the train file created
        with the split() method is used. If file is not None, train is ignored and the file specified
        is read instead.
        """

        class StringIterable(object):
            def __init__(self, datafile, seed=0):
                self.datafile = datafile
                self.seed = seed
                self.line_dataset = LineTsvDataset(self.datafile)
                if len(self.line_dataset) == 0:
                    raise Exception("Dataset {} contains no instances, will not process")
                self.shuffled_dataset = ShuffledDataset(self.line_dataset, seed=self.seed)

            def __iter__(self):
                for i in range(len(self.shuffled_dataset)):
                    yield self.shuffled_dataset[i]

        if file:
            whichfile = file
        else:
            if train:
                if not self.have_orig_split:
                    raise Exception("We do not have a train file in original format for instances_as_string")
                whichfile = self.orig_train_file
            else:
                whichfile = self.orig_data_file
        return StringIterable(whichfile, seed=self.seed)

    def instances_original(self, train=False, file=None):
        """Returns an iterable that allows to read the instances from a file in original format.
        This file is the original data file by default, but could also the train file created with
        the split() method or any other file derived from the original data file."""

        class StringIterable(object):
            def __init__(self, datafile, seed=0):
                self.datafile = datafile
                self.seed = seed
                self.line_dataset = LineTsvDataset(self.datafile)
                if len(self.line_dataset) == 0:
                    raise Exception("Dataset {} contains no instances, will not process")
                self.shuffled_dataset = ShuffledDataset(self.line_dataset, seed=self.seed)

            def __iter__(self):
                for i in range(len(self.shuffled_dataset)):
                    yield self.shuffled_dataset[i]
        if file:
            whichfile = file
        else:
            if train:
                if not self.have_orig_split:
                    raise Exception("We do not have a train file in original format for instances_as_string")
                whichfile = self.orig_train_file
            else:
                whichfile = self.orig_data_file
        return StringIterable(whichfile, seed=self.seed)


    def convert_indep(self, indep, normalize=None):
        """Convert the independent part of an original representation into the converted representation
        where strings are replaced by word indices or one hot vectors.
        If normalize is None then the normalization will be performed according to the default
        for the feature, otherwise it should be one of "minmax", "meanvar", or False or a normalizing function.
        If False, normalization is turned off explicitly, otherwise the normalization function is used.
        This parameter is ignored for all features which are not numeric.
        """
        return self.features(indep, normalize=normalize)

    def convert_dep(self, dep, is_batch=False, as_onehot=False):
        """Convert the dependent part of an original representation into the converted representation
        where strings are replaced by one hot vectors.
        If as_onehot is True, then nominal targets is onverted to onehot float vectors instead of
        integer indices (ignored for other target types).
        """
        if is_batch:
            ret = [self.target(dep[v], as_onehot=as_onehot) for v in dep]
            if isinstance(dep, np.ndarray):
                ret = np.array(ret)
            return ret
        else:
            return self.target(dep, as_onehot=as_onehot)

    def convert_instance(self, instance, normalize="meanvar", is_reshaped_batch=False):
        """Convert an original representation of an instance as read from json to the converted representation.
        This will also by default automatically normalize all numeric features, this can be changed by setting
        the normalize parameter (see convert_indep).
        If is_reshaped_batch is True, then we expect a batch of reshaped instances instead of
        a single instance
        Note: if the instance is a string, it is assumed it is still in json format and will get converted first."""
        if is_reshaped_batch:
            # todo: use the per-feature conversion methods instead!
            raise Exception("NOT YET IMPLEMENTED!")
        if isinstance(instance, str):
            instance = json.loads(instance, encoding="utf=8")
        (indep, dep) = instance
        indep_converted = self.convert_indep(indep, normalize=normalize)
        dep_converted = self.convert_dep(dep)
        return [indep_converted, dep_converted]

    def split(self, outdir=None, validation_size=None, validation_part=0.1, random_seed=1,
              convert=False, keep_orig=False, reuse_files=False,
              validation_file=None):
        """This splits the original file into an actual training file and a validation set file.
        This creates two new files in the same location as the original files, with the "data"/"meta"
        parts of the name replaced with "val" for validation and "train" for training. If converted is
        set to True, then instead of the original data, the converted data is getting split and
        saved, in that case the name parts are "converted.var" and "coverted.train". If keep_orig is
        set to True, then both the original and the converted format files are created. Depending on which
        format files are created, subsequent calls to the batches_converted or batches_orig can be made.
        If outdir is specified, the files will get stored in that directory instead of the directory
        where the meta/data files are stored.
        If random_seed is set to 0 or None, the random seed generator does not get initialized.
        If reuse_files is True and the files that would have been created are already there
        the method does nothing for that file, assuming, but not checking that the contents is correct.
        If validation_file is not none, then validation_size and validation_part are ignored and
        the whole original file is used as training file, the given validation_file is expected to
        be a data file that fits the meta, and the content of the validation file is used.
        """
        logger.debug("Called split with validation_size=%s validation_part=%s validation_file=%s",
                     (validation_size, validation_part, validation_file))
        valindices = set()

        # the following is only relevant if we do not have a defined validation file
        if not validation_file:
            if validation_size or validation_part:
                if random_seed:
                    np.random.seed(random_seed)
                if validation_size:
                    valsize = int(validation_size)
                else:
                    valsize = int(self.nInstances * validation_part)
                if valsize <= 1 or valsize > int(self.nInstances / 2.0):
                    raise Exception('Validation set size should not be less than 1 or more '
                                    'than half the data, but is %s (n=%s)' % (valsize, self.nInstances))
                # now get valsize integers from the range 0 to nInstances-1: these are the instance indices
                # we want to reserve for the validation set
                choices = np.random.choice(self.nInstances, size=valsize, replace=False)
                logger.debug("convert_to_file, nInst=%s, valsize=%s, choices=%s" % (self.nInstances, valsize, len(choices)))
                for choice in choices:
                    valindices.add(choice)
            else:
                # we just keep the empy valindices set
                pass

        origtrainfile = self.modified4meta(name_part="orig.train", dirname=outdir)
        origvalfile = self.modified4meta(name_part="orig.val", dirname=outdir)
        convtrainfile = self.modified4meta(name_part="converted.train", dirname=outdir)
        convvalfile = self.modified4meta(name_part="converted.val", dirname=outdir)
        outorigtrain = None
        outorigval = None
        outconvtrain = None
        outconvval = None
        if not convert or (convert and keep_orig):
            self.have_orig_split = True
            if not (reuse_files and os.path.exists(origtrainfile)):
                outorigtrain = open(origtrainfile, "w", encoding="utf-8")
            self.orig_train_file = origtrainfile
            if not (reuse_files and os.path.exists(origvalfile)):
                outorigval = open(origvalfile, "w", encoding="utf-8")
            self.orig_val_file = origvalfile
        if convert:
            self.have_conv_split = True
            if not (reuse_files and os.path.exists(convtrainfile)):
                outconvtrain = open(convtrainfile, "w", encoding="utf-8")
            self.converted_train_file = convtrainfile
            if not (reuse_files and os.path.exists(convvalfile)):
                outconvval = open(convvalfile, "w", encoding="utf-8")
            self.converted_val_file = convvalfile
        # print("DEBUG origtrain/origval/convtrain/convval=%s/%s/%s/%s" % (outorigtrain, outorigval,
        # outconvtrain, outconvval), file=sys.stderr)
        self.outdir = outdir
        i = 0
        n_train = 0
        n_val = 0

        # if we do not need to do anything, reurn
        if not outorigtrain and not outorigval and not outconvval and not outconvtrain:
            return

        # if we actually need to split, run the following
        if validation_file is None:
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
                    n_val += 1
                else:
                    if outorigtrain:
                        print(line, file=outorigtrain, end="")
                    if outconvtrain:
                        print(converted, file=outconvtrain)
                    n_train += 1
                i += 1
        else:
            # we have a separate validation file: in this case, we copy the original file
            # to the outorigintrain, if necessary, write the converted original file to
            # outconvtrain, if necessary, then do the same with the separate validation file
            for line in self.instances_as_string():
                if convert:
                    converted = self.convert_instance(line)
                else:
                    converted = None
                if outorigtrain:
                    print(line, file=outorigtrain, end="")
                if outconvtrain:
                    print(converted, file=outconvtrain)
                n_train += 1
                i += 1
            for line in self.instances_as_string(file=validation_file):
                if convert:
                    converted = self.convert_instance(line)
                else:
                    converted = None
                if outorigval:
                    print(line, file=outorigval, end="")
                if outconvval:
                    print(converted, file=outconvval)
                n_val += 1
                i += 1

        logger.info("Created training/validation files %s / %s instances (total %s)" % (n_train, n_val, i))
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
        with open(outfile, "w") as out:
            for instance in self.instances_converted(train=False, convert=True, file=infile):
                print(json.dumps(instance), file=out)

    def validation_set_orig(self):
        """Read and return the validation set rows in original format. For this to work, the split()
        method must have been run and either convert have been False or convert True and keep_orig True."""
        if not self.have_orig_split:
            raise Exception("We do not have the splitted original file, run the split method.")
        validationsetfile = self.modified4meta(name_part="orig.val", dirname=self.outdir)
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
        validationsetfile = self.modified4meta(name_part="converted.val", dirname=self.outdir)
        valset = []
        with open(validationsetfile, "rt") as inp:
            for line in inp:
                valset.append(json.loads(line))
        if as_batch:
            # print("DEBUG: valset[0]dep is:", valset[0][1], file=sys.stderr)
            # print("DEBUG: len valset[0]indep is:", len(valset[0][0]), file=sys.stderr)
            # print("DEBUG: len valset[0]dep is:", len(valset[0][1]), file=sys.stderr)
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
            def __init__(self, meta, datafile, parent):
                self.meta = meta
                self.datafile = datafile
                self.parent = parent

            def __iter__(self):
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
                    whichfile = self.orig_data_file
                else:
                    if self.converted_data_file:
                        whichfile = self.converted_data_file
                    else:
                        raise Exception("We do not have a data file in converted format for instance_as_string")
        return DataIterable(self.meta, whichfile, self)

    @staticmethod
    def pad_list_(thelist, tosize, pad_left=False, pad_value=None):
        """Pads the list to have size elements, inserting the pad_value as needed,
        left or right depending on pad_left. CAUTION! Modifies thelist and also returns it.
        """
        n_needed = tosize - len(thelist)
        if n_needed <= 0:
            return thelist
        if pad_left:
            for i in range(n_needed):
                thelist.insert(0, pad_value)
        else:
            for i in range(n_needed):
                thelist.append(pad_value)
        return thelist

    @staticmethod
    def pad_matrix_(matrix, tosize=None, pad_left=False, pad_value=None):
        """Given a list of lists, pads all the inner list to size length or if size is None, first determines
        the longest inner list and pads to that length. CAUTION: modifies the matrix!"""
        if not tosize:
            tosize = max([len(x) for x in matrix])
        for l in matrix:
            Dataset.pad_list_(l, tosize, pad_left=pad_left, pad_value=pad_value)

    @staticmethod
    def reshape_batch_helper(instances, as_numpy=False, pad_left=False, from_original=False, pad=True,
                             n_features=None, is_sequence=None, feature_types=None, target=None,
                             indep_only=False):
        """Reshapes a list of instances where each instance is a two-element list of an independent and dependent/target
        part into a tuple where the first part is a list of features and the second part is the list of targets.
        If the instances are not for sequence tagging, then each list that corresponds to a feature contains as many
        values as there are instances. If the value of the feature is a sequence, then each value is a padded list.
        For sequence tagging, The independent part contains as many lists as there are features, each of these
        lists contains as many elements as there are instances. These elements in turn are lists, representing the
        values of the feature for each feature vector in the sequence for the instance.
        The feature_types list must be specified if is_sequence is True, in that case, n_features is not needed.
        If indep_only is True, this will not expect targets and only reshape the independent features.
        IMPORTANT: this pads all independent features based on their type, with indices getting padded using 0
        and all dependent indices using -1!!! If target is specified, the targets are represented by a
        onehot vector instead."""

        if feature_types:
            n_features = len(feature_types)

        if not pad and as_numpy:
            raise Exception("Parameter pad must be True if as_numpy is True.")

        # create the target data structures: for the independent part this is a list with n_features sublists
        # for the targets this is a single list
        out_indep = [[] for _ in range(n_features)]
        out_dep = []

        # if not sequences, transpose the input features and add the targets
        if not is_sequence:
            flist_max_len = [0 for _ in range(n_features)]
            for instance in instances:
                if indep_only:
                    indep = instance
                else:
                    (indep, dep) = instance
                assert len(indep) == n_features
                for i in range(n_features):
                    out_indep[i].append(indep[i])
                    if isinstance(indep[i], list):
                        flist_max_len[i] = max(flist_max_len[i], len(indep[i]))
                if not indep_only:
                    out_dep.append(dep)
            # now that we have all the lists of feature values, if the values are themself lists, pad them:
            # Here it is easy to find the pad_value: since we can only have ngrams as the reason for list values,
            # we use a "" for the original and 0 for converted
            if from_original:
                pad_value = ''
            else:
                pad_value = 0
            for i in range(n_features):
                if flist_max_len[i] > 0 and pad:
                    Dataset.pad_matrix_(out_indep[i], tosize=flist_max_len[i], pad_left=pad_left, pad_value=pad_value)
            if as_numpy:
                if not indep_only:
                    out_dep = np.array(out_dep)
                for i in range(n_features):
                    out_indep[i] = np.array(out_indep[i])
        else:  # is_sequence is True
            if not feature_types:
                raise Exception("Need a list of feature types if is_sequence is True")
            # this are instances with sequences of feature vectors and sequences of targets
            # for each feature, there are as many values as there are feature vectors in the sequence for that instance
            # for the final output, we need to pad all the features to the length of the longest sequence
            seq_max_len = 0
            for instance in instances:
                if indep_only:
                    indep = instance
                else:
                    (indep, dep) = instance
                # indep is a list of feature vectors!
                # the number of featue vectors must be equal to the number of targets
                seq_len = len(indep)
                # print("DEBUG: indep_len/dep_len=%s/%s" % (len(indep), len(dep)), file=sys.stderr)
                # print("DEBUG: dep is:", dep, file=sys.stderr)
                # print("DEBUG: indep is:", indep, file=sys.stderr)
                if not indep_only:
                    assert len(dep) == seq_len
                seq_max_len = max(seq_len, seq_max_len)
                # now we need to add a list to each of the features, each list is the
                # values for a feature for all the sequence elements.
                for feature_idx in range(n_features):
                    values = []
                    for el_idx in range(seq_len):
                        val = indep[el_idx][feature_idx]
                        if isinstance(val, list):
                            raise Exception("Sequences/ngrams within sequences of feature vectors not supported")
                        values.append(val)
                    out_indep[feature_idx].append(values)
                if not indep_only:
                    out_dep.append(dep)
            # we now have all the features and targets, need to pad all of those to the maximum sequence length
            # NOTE: currently the targets for sequence tagging are always nominal, so padding is done with
            # '' for original and 0 otherwise. The exception is if the target gets represented as a one hot
            # vector, in which case the appropriate zero-vector needs to get used instead
            if not indep_only:
                if from_original:
                    pad_value = ''
                else:
                    if target and target.as_onehot:
                        pad_value = target.zero_onehotvec()
                    else:
                        # TODO: this was 0 previously, but we use -1 here directly for sequences,
                        # when our target vocab does NOT use separate padding symbol!
                        pad_value = -1
                if pad:
                    Dataset.pad_matrix_(out_dep, tosize=seq_max_len, pad_left=pad_left, pad_value=pad_value)
            # to pad the features, we need to know the type of the feature and if we have original format:
            # For original:
            # "nominal" - ""
            # "number" - 0.0
            # "boolean" - False
            # For converted:
            # "float" - 0.0
            # "index" - 0
            # Note that for sequences we cannot have nested ngrams so "ngram" / "indexlist" cannot occur here!
            for i in range(n_features):
                ftype = feature_types[i]
                if from_original:
                    if ftype == "nominal":
                        pad_value = ''
                    elif ftype == "number":
                        pad_value = 0.0
                    elif ftype == "boolean":
                        pad_value = False
                    else:
                        raise Exception("Odd type for feature %s: %s" % (i, ftype))
                else:
                    if ftype == "float":
                        pad_value = 0.0
                    else:
                        pad_value = 0
            # pad each feature
            for f in out_indep:
                if pad:
                    Dataset.pad_matrix_(f, seq_max_len, pad_left=pad_left, pad_value=pad_value)
            if as_numpy:
                if not indep_only:
                    out_dep = np.array(out_dep)
                for i in range(n_features):
                    out_indep[i] = np.array(out_indep[i])

        if as_numpy:
            # convert the features list itself to numpy
            # NOTE: even with dtype=object, numpy will try to broadcast as much as possible, so if
            # the list contains numpy arrays for 2 features which are sequences of different max size,
            # the there will be two matrices and the outermost array cannot be built.
            # Instead we create an empty object array of the right size first and then assign the elemts
            tmp = np.empty(n_features, dtype=object)
            for i in range(n_features):
                tmp[i] = out_indep[i]
            out_indep = tmp
        if not indep_only:
            ret = (out_indep, out_dep)
        else:
            ret = out_indep
        return ret

    def reshape_batch(self, instances, as_numpy=False, pad_left=False, from_original=False, pad=True, indep_only=False):
        """Reshape the list of converted instances into what is expected for training on a batch.
        NOTE: for non-sequence instances, we pad all list-typed features to the maximum length. If from_original
        is true, the padding is done with empty strings, otherwise with integer zeros.
        NOTE: as_numpy=True for from_original=True currently only converts the result of converting
        the outermost list to a numpy array which will automatically also convert the embedded lists.
        """
        if from_original:
            feature_types = self.feature_types_original()
        else:
            feature_types = self.feature_types_converted()
        return Dataset.reshape_batch_helper(instances, as_numpy=as_numpy, pad_left=pad_left,
                                            from_original=from_original, pad=pad,
                                            feature_types=feature_types,
                                            is_sequence=self.isSequence,
                                            indep_only=indep_only,
                                            target=self.target)

    def batches_original(self, train=True, file=None, reshape=True, batch_size=100, pad_left=False, as_numpy=False):
        """Return a batch of instances in original format for training.
        """
        class BatchOrigIterable(object):

            def __init__(self, thefile, batchsize, parent):
                self.file = thefile
                self.batchsize = batchsize
                self.parent = parent

            def __iter__(self):
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
                        if len(collect) == 0:
                            break
                        if reshape:
                            batch = self.parent.reshape_batch(collect, pad_left=pad_left, as_numpy=as_numpy,
                                                              from_original=True)
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
                    if self.orig_data_file:
                        whichfile = self.orig_data_file
                    else:
                        raise Exception("We do not have a data file in original format for batches_converted")
        return BatchOrigIterable(whichfile, batch_size, self)

    def batches_converted(self, train=True, file=None, reshape=True, convert=False, batch_size=100, as_numpy=False,
                          pad_left=False):
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
                        if len(collect) == 0:
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
                    whichfile = self.orig_data_file
                else:
                    if self.converted_data_file:
                        whichfile = self.converted_data_file
                    else:
                        raise Exception("We do not have a data file in converted format for batches_converted")
        return BatchConvertedIterable(whichfile, batch_size, self)

    def feature_types_converted(self):
        """Returns a list with the converted types of the features as a string name.
        Possible values are 'float', 'index', 'indexlist'
        """
        return [f.type_converted() for f in self.features]

    def feature_types_original(self):
        """Returns a list with the original types of the features as a string name.
        Possible values are 'nominal', 'number', 'ngram', 'boolean'
        """
        return [f.type_original() for f in self.features]

    def _calculate_feature_idxs(self):
        """Helper method to calculate and cache all the per-converted type feature index lists.
        NOTE: the type we get for each feature is either float, index, or indexlist where index indicates
        the index of a nominal value.
        """
        if self._have_feature_idxs:
            return
        self._float_feature_idxs = []
        self._index_feature_idxs = []
        self._indexlist_feature_idxs = []
        self._float_features = []
        self._index_features = []
        self._indexlist_features = []
        idx = 0
        for f in self.features:
            t = f.type_converted()
            if t == "float":
                self._float_feature_idxs.append(idx)
                self._float_features.append(f)
            elif t == "index":
                self._index_feature_idxs.append(idx)
                self._index_features.append(f)
            elif t == "indexlist":
                self._indexlist_feature_idxs.append(idx)
                self._indexlist_features.append(f)
            else:
                raise Exception("Feature type unknown, looks like a bug: %s" % t)
            idx += 1

    def get_float_feature_idxs(self):
        """Return a list of indices of all numeric or boolean features"""
        if not self._have_feature_idxs:
            self._calculate_feature_idxs()
        return self._float_feature_idxs

    def get_index_feature_idxs(self):
        """Return a list of indices of all nominal features represented by some index and
        ultimately by a vector"""
        if not self._have_feature_idxs:
            self._calculate_feature_idxs()
        return self._index_feature_idxs

    def get_indexlist_feature_idxs(self):
        """Return a list of indices for all features which are ngrams, i.e. lists of embs."""
        if not self._have_feature_idxs:
            self._calculate_feature_idxs()
        return self._indexlist_feature_idxs

    def get_float_features(self):
        """Return a list of numeric or boolean features"""
        if not self._have_feature_idxs:
            self._calculate_feature_idxs()
        return self._float_features

    def get_index_features(self):
        """Return a list of all nominal features represented by some index and
        ultimately by a vector"""
        if not self._have_feature_idxs:
            self._calculate_feature_idxs()
        return self._index_features

    def get_indexlist_features(self):
        """Return a list of features which are ngrams, i.e. lists of embs."""
        if not self._have_feature_idxs:
            self._calculate_feature_idxs()
        return self._indexlist_features

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
        return "Dataset(meta=%s,isSeq=%s,nFeat=%s,N=%s)" % \
               (self.metafile, self.isSequence, self.nAttrs, self.nInstances)

    def __repr__(self):
        return "Dataset(meta=%s,isSeq=%s,nFeat=%s,N=%s,features=%r,target=%r)" % \
               (self.metafile, self.isSequence, self.nAttrs, self.nInstances, self.features, self.target)
