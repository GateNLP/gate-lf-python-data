#!/usr/bin/env python
'''
Temporary implementation of the Dataset interface without dependency on
pytorch to make accessing the data file in a sorted way possible.
IMPORTANT! This has been copied over from a different library and currently
contains some features which are not used or relevant in here.
'''

import os
import random
import sys
import pickle
import json


class ExtendedDataset(object):
    """
    Our own base class for datasets which adds a few conventions:
    is_writable is False by default but some Datasets can be defined to be writable.
    Writable datasets implement __setitem(self, key, item)__ for a key that is an integer.
    Note: all classes that derive from this class should invoke the parent init method!
    """
    def __init__(self):
        self.is_writable = False

    def __setitem__(self, key, value):
        """
        Write/save a specific instance (identified by instance number)
        :param key:  the instance number, must be an integer
        :param value:  the instance
        :return:
        """
        raise Exception("Dataset is not writable!")


class ShuffledDataset(ExtendedDataset):
    """
    Represents a shuffled version of another dataset.
    """

    def __init__(self, dataset, seed=None):
        """
        :param seed: if not None, shuffle the list of instances randomly, using the given seed.
          If the seed is 0, the system time is used, if seed is -1, the seed is not set at all
          and whatever the current state of the random generator is is used.
        """
        super().__init__()
        self.dataset = dataset
        self.seed = seed
        self.idxs = list(range(len(dataset)))
        self.shuffle(seed)

    def shuffle(self, seed=0):
        """
        Shuffle instance list order,
        :param seed: random seed to set, if seed is 0, system time is used, if -1, seed is not set.
        :return:
        """
        if seed != -1:
            if seed == 0:
                random.seed()
            else:
                random.seed(seed)
        random.shuffle(self.idxs)

    def __getitem__(self, key):
        return self.dataset[self.idxs[key]]

    def __len__(self):
        return len(self.idxs)


class TransformDataset(ExtendedDataset):

    def __init__(self, dataset, transforms):
        super().__init__()
        self.dataset = dataset
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [transforms]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        tmp = self.dataset[key]
        for tr in self.transforms:
            tmp = tr(tmp)
        return tmp


class LineTsvDataset(ExtendedDataset):
    """
    Represent a large TSV file or simple one document/text per line file as a dataset.
    When creating the instance, an index file is
    is created and stored along with the original file, unless it already exists.
    NOTE: this works only if lines are separated with "\n"!!!
    """

    def have_current_index(self):
        if not os.path.exists(self.indexfile):
            return False
        # if we have an index file, check if its modification date is more recent than
        # that of the data file: if not, return false
        return os.path.getmtime(self.indexfile) > os.path.getmtime(self.file)

    def __init__(self, file, indexfile=None, reinit=False,
                 encoding="utf8", cols=None, logevery=1000):
        """
        Create the dataset instance from the given file.
        :param file: the tsv file
        :param indexfile: index file to use, by default the original file path with ".dsindex" appended
        :param reinit: if True, forces re-creation of the index file even if it already exists
        :param if cols is None, the whole line is returned to the iterator, otherwise if it is a number, that
          column is returned, otherwise if it is a list of numbers, those fields are returned
        """
        self.reader = None   # set this first so even if the super init throws an exception, __del__ still finds it
        super().__init__()
        self.file = file
        if not os.path.exists(file):
            raise Exception("File does not exist: {}".format(file))
        if indexfile is None:
            indexfile = file + ".dsindex"
        self.indexfile = indexfile
        self.encoding = encoding
        self.cols = cols
        self.logevery = logevery
        # if we need to create the cache file, do this now.
        if reinit or not self.have_current_index():
            self.idx2offlen = self._index4file(file)
            with open(indexfile, "wb") as indexwriter:
                pickle.dump(self.idx2offlen, indexwriter)
        else:
            with open(indexfile, "rb") as indexloader:
                self.idx2offlen = pickle.load(indexloader)
        self.len = len(self.idx2offlen)

    def __del__(self):
        # print("DEBUG: calling __del__")
        if self.reader is not None:
            # print("DEBUG: closing reader!")
            self.reader.close()

    def _index4file(self, file):
        idx2offlen = []
        with open(file, "rb") as reader:
            startoffset = 0
            linenr = 0
            # since this is reading in binary mode, the terminator is always "\n"
            # NOTE: we could also read in text mode, specify the newline or automatically
            # recognize both both Windows and Linux newlines and then count by encoding the
            # utf8 string we get into bytes and hope for the best. However, we expect all
            # line corpora to be in Linux format for now!
            for linebytes in reader:
                # line = bytes.decode(self.encoding)
                linelen = len(linebytes)
                idx2offlen.append((startoffset, linelen))
                # print("DEBUG indexing {}/{}".format(startoffset,l))
                startoffset += linelen
                linenr += 1
                if self.logevery is not None and linenr % self.logevery == 0:
                    print("Lines indexed: {}".format(linenr), file=sys.stderr)
        return idx2offlen

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.reader is None:
            # print("DEBUG: opening reader!")
            self.reader = open(self.file, "rb")
        if index >= self.len or index < -self.len:
            raise IndexError()
        off, linelen = self.idx2offlen[index]
        self.reader.seek(off, os.SEEK_SET)
        bytes = self.reader.read(linelen)
        line = bytes.decode(self.encoding)
        if self.cols is None:
            return line
        else:
            fields = line.split("\t")
            if isinstance(self.cols, list):
                return [fields[i] for i in self.cols]
            else:
                return fields[self.cols]


class DirFilesDataset(ExtendedDataset):

    def path4(self, index):
        if self.basenames is not None:
            name = self.basenames[index]
        else:
            name = str(index)
        fname = "{}.{}".format(name, self.as_format)
        fpath = os.path.join(self.directory, fname)
        return fpath

    def __init__(self, directory, as_format='pickle', basenames=None, files_exist=True):
        """
        Create a dataset where instances are files in a directory. This is a very simple
        implementation so far which only supports a single directory.

        If basenames is given, then it must be a list of file basenames that correspond to each
        id that exists in the dataset. The length of that list is taken as the length of this dataset.
        These files are expected to already exist.
        If no basenames are given, then the file base names are assumed to be the numbers from 0
        to len(dataset)-1.
        :param directory:
        :param as_format:
        :param basenames:
        :param files_exist: if True, all files must already exist, if False, this is mainly for
          use as a cache.
        """
        self.directory = directory
        self.is_writable = True
        self.basenames = basenames
        self.files_exist = files_exist
        if as_format not in ['pickle', 'json', 'torch']:
            raise Exception("Format must be one of pickle, json, torch")
        self.as_format = as_format
        if basenames is None:
            if not files_exist:
                self.len = 0  # this will get updated whenever a file is stored
                return  # do not need to do any checking!
            # find all the files in the directory that exist
            i = 0
            len = 0
            while True:
                if os.path.exists(os.path.join(self.directory, str(i)+"."+self.as_format)):
                   len = i+1
                   i += 1
                else:
                    break
            if len == 0:
                raise Exception("No files found!")
            self.len = len
        else:
            self.len = len(basenames)
            if len == 0:
                raise Exception("The basenames list must not be empty")
            # check if all files are actually there
            if files_exist:
                for f in basenames:
                    filename = os.path.join(self.directory, f+"."+self.as_format)
                    if not os.path.exists(filename):
                        raise Exception("File does not exist:", filename)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fpath = self.path4(index)
        if self.as_format == "json":
            with open(fpath, "rt", encoding="utf8") as reader:
                return json.load(reader)
        elif self.as_format == "pickle":
            with open(fpath, "rb") as reader:
                return pickle.load(reader)
        elif self.as_format == "torch":
            with open(fpath, "rb") as reader:
                return torch.load(reader, map_location="cpu")

    def __setitem__(self, index, value):
        fpath = self.path4(index)
        fname = "{}.{}".format(str(index), self.as_format)
        fpath = os.path.join(self.directory, fname)
        if self.as_format == "json":
            with open(fpath, "wt", encoding="utf8") as writer:
                json.dump(value, writer)
        elif self.as_format == "pickle":
            with open(fpath, "wb") as writer:
                pickle.dump(value, writer)
        elif self.as_format == "torch":
            with open(fpath, "wb") as writer:
                torch.save(value, writer)
        if not self.files_exist and index > (self.len-1):
            self.len = index-1


class DirCachedDataset(DirFilesDataset):

    def __init__(self, dataset=None, directory=None, as_format='pickle', basenames=None):
        """
        Create a caching dataset. If dataset is specified, then its length is used and
        if an entry is not found, it is retrieved from that dataset and stored to the directory.
        If basenames is given, then it must be a list of file basenames that correspond to each
        id that exists in the dataset. The length of that list is taken as the length of this dataset.
        The files do not have to exist initially.
        If no dataset is given, and basenames are given, these files are expected to be in the
        directory already. If no dataset and no basenames are given, then all files in the directory,
        starting with basename 0 and incremented by one are used, until one is not found.
        :param dataset:
        :param directory:
        :param as_format:
        :param basenames:
        """
        if directory is None:
            raise("directory must be specified")
        fe = (dataset is None)
        super().__init__(directory, as_format=as_format, basenames=basenames, files_exist=fe)
        self.dataset = dataset

    def __len__(self):
        if self.dataset is None:
            return self.len
        else:
            return len(self.dataset)

    def __getitem__(self, index):
        fpath = self.path4(index)
        if os.path.exists(fpath):
            return super().__getitem__(index)
        else:
            if self.dataset is None:
                raise Exception("No dataset and unexpected missing file")
            ret = self.dataset.__getitem__(index)
            super().__setitem__(index, ret)
            return ret


if __name__ == "__main__":
    # just run a quick sanity test
    with open("tmp_linetsvdataset.tsv", "wt", encoding="utf8") as writer:
        print("this is the first line!", file=writer)
        print("Some umlauts like ä or Ü or ś and Ñ and ì...", file=writer)
        print("this is another line", file=writer)
        print("and another", file=writer)
        print("Last one!!!", file=writer)

    ds = LineTsvDataset(file="tmp_linetsvdataset.tsv", reinit=True)
    for line in ds:
        print(line)

    print("Last line: ", ds[-1])
    print("First line: ", ds[-5])

    from torch.utils.data import DataLoader

    def cfn1(l):
        print("We got:",l)
        return l

    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=cfn1)

    for batch in dl:
        print("Batch: ", batch)

    ds2tmp = LineTsvDataset(file="tmp_linetsvdataset.tsv", reinit=False)
    ds2 = TransformDataset(ds2tmp, len)
    dl2 = DataLoader(ds2, batch_size=2, shuffle=True)
    for batch in dl2:
        print("Batch2: ", batch)

