"""Module for the Vocab class"""

from collections import Counter, defaultdict
import logging
import gzip
import re
import numpy as np
import math
# TODO: maybe make use of the gensim library optional?
# import gensim
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)

# OK, the protocol for using this is this:
# * create a preliminary instance using "Vocab(...)"
#   This can be a completely empty instance or already contain an initial set of counts using counts=
#   NOTE: if an embedding file should be used, it should be specified at creation time
#   NOTE: emb_train can be "yes", "mapping", "no" or "onehot":
# * n times, add to the counts using v.add_counts() and also set embeddings-information using v.set_xxx()
#   and other info (e.g. additional symbols use) using v.set_xxx() to override what was specified
#   These settings should all be settings which only are relevant once finish() is called
# * Once all the counts have been set, finish the datastructure and prepare for use using v.finish()
# * only after f.finish() is run the methods for mapping from and to indices, frequencies, or vectors
#   are usable, before finish() these methods will abort with an error!
# * In general:
#   * words which should be in the vocab get mapped to their index
#   * words not in the vocab get mapped to the special "OOV" index (1 by default)
#   * there is the special PAD symbol which is 0 by default
# * Once finish has been completed, a numpy matrix with the embeddings can be retrieved using
#   get_embeddings()
# * At some point, all the unimportant data can be cleaned up using v.cleanup(embeddings=True,frequencies=True)
#   After this, get_embeddings() raises an exception

# Overview over how indices and embeddings are created using the various configuration settings:
#
# train=yes, file=None:
# * all words in the training set not filtered by minfrequ or maxsize get mapped to an index
# * all other words get mapped to the OOV index
# * if there are k words, there will be k+2 (OOV, PAD) indices
# * Embeddings: all embeddings are some random vectors, PAD is zero
#
# train=yes, file=embfile
# * index for all words in the training set > minfreq, maxsize which are also in the embeddings file
# * OOV for all other words (random vector)
# * Embeddings: all embeddings from file except OOV (random) and PAD(zero)
# * TODO: at some point could average low-freq vectors in the embeddings
#
# train=no, file=None
# * same as for train=yes, file=None
# * only pytorch layer works differently
#
# train=no, file=embfile
# * same as for train=yes, file=embfile
# * only pytorch layer works differently
#
# train=mapping, file=None
# * signal an ERROR
#
# train=mapping, file=embfile
# * index for all words either in our vocab or in the embeddings file, but NOT in our vocab and < minfreq!
# * so we have to load all embeddings from the file, except those which correspond to the words we
#   have filtered. In addition, all the words not filtered and not in the embeddings get random vectors,
#   and we create an OOV vector as well.
#
# train=onehot, file=none
# * ignore minfreq, dims
# * dims is equal to number of values
# * no OOV dim, only padding, unless suppressed
# * embeddings matrix is diagonal matrix


class Vocab(object):
    """From the counter object, create string to id and id to string
    mappings."""
    def __init__(self, counts=None, max_size=None,
                 emb_id=None, emb_train=None, emb_dims=0, emb_file=None, emb_minfreq=1,
                 no_special_indices=False,
                 pad_index_only=False,
                 emb_dir=None,
                 pad_string="", oov_string="<<oov>>"):
        """Create a vocabulary instance from the counts. If max_size is
        given sorts by frequency and only retains the max_size most frequent
        ones. Removes everything less the emb_minfreq.
        The padding index is always 0, however the string for which the padding index is returned can be set.
        The OOV index is always 1, however the string for which the padding index is returned can be set.
        If no_special_indices is true, only indices for words are added, not for padding or oov. looking up
        a word not in the vocabulary will result in an exception.
        If pad_index_only is true then no oov index will be used, looking up a word not in the vocabulary raises
        an exception. However, the index 0 is reserved for padding.
        NOTE: if emb_train is onehot and neither no_special_indices nor pad_index_only is true,
        for now we automatically use pad_index_only!!!!
        If emb_dir is not None, then all references to relative (embeddings) files are relative to that
        directory.
        """
        if counts:
            self.freqs = Counter(counts)
        else:
            self.freqs = Counter()
        if emb_train == "onehot" and not no_special_indices:
            pad_index_only = True
        self.no_special_indices = no_special_indices
        self.pad_index_only = pad_index_only
        self.emb_minfreq = emb_minfreq or 1
        self.max_size = max_size
        self.emb_dims = emb_dims
        self.emb_id = emb_id
        self.emb_file = emb_file
        self.emb_train = emb_train
        self.itos = None
        self.stoi = None
        self.stoe = {}
        self.emb_dir = emb_dir
        self.n = 0
        self.pad_string = pad_string
        self.oov_string = oov_string
        self.finished = False
        self.embeddings_loaded = False
        self.embeddings = None
        self.oov_emb = None
        if self.emb_train and self.emb_train not in ["yes", "mapping", "no", "onehot"]:
            raise Exception("Vocab emb_train must be one of yes, mapping, no, onehot but is "+str(self.emb_train))
        if not self.emb_file and self.emb_train == "mapping":
            raise Exception("Vocab emb_train 'mapping' not usable without embeddings file, "
                            "got emb_train=%s and emb_file=%s" % (self.emb_train, self.emb_file))
        if self.emb_file and self.emb_train == "onehot":
            raise Exception("Vocab emb_train 'onehot' not usable with embeddings file, "
                            "got emb_train=%s and emb_file=%s" % (self.emb_train, self.emb_file))
        self.have_oov = True
        self.have_pad = True
        self.have_vocab = False  # this indicates if we have already built the final vocab
        if no_special_indices:
            self.have_oov = False
            self.have_pad = False
        if pad_index_only:
            self.have_oov = False

    def check_finished(self, method="method"):
        if not self.finished:
            raise Exception("Cannot call", method, "unless the finish() method has been called first!")

    def check_nonfinished(self, method="method"):
        if self.finished:
            raise Exception("Cannot call", method, "after the finish() method has been called!")

    def embs4line(self, line, fromidx, dims):
        embs = []
        toidx = fromidx
        for i in range(dims):
            fromidx = toidx + 1
            toidx = line.find(" ", fromidx)
            if toidx < 0:
                toidx = len(line)
            embs.append(float(line[fromidx:toidx]))
        return embs

    def load_embeddings(self, emb_file, filterset=None):
        """Load pre-calculated embeddings from the given file. This will update embd_dim as needed!
        Currently only supports text format, compressed text format or a two file format where
        the file with extension ".vocab" has one word per line and the file with extension ".npy"
        is a matrix with as many rows as there are words and as many columns as there are dimensions.
        The format is identified by the presence of one of the extensions ".txt", ".vec", ".txt.gz",
        or ".vocab" and ".npy" in the emb_file given. (".vec" is an alias for ".txt")
        The text formats may or may not have a first line that indicates the number of words and
        number of dimensions.
        If filterset is non-empty, all embeddings not in the set are loaded, otherwise all embeddings
        which are also already in the vocabulary are loaded.
        NOTE: this will not check if the case conventions or other conventions (e.g. hyphens) for the tokens
        in our vocabulary are compatible with the conventions used for the embeddings.
        """
        if filterset is None:
            filterset = set()
        n_lines = 0
        n_added = 0
        n_vocab = len(self.itos)
        if emb_file.endswith(".txt") or emb_file.endswith(".vec") or emb_file.endswith(".txt.gz"):
            if emb_file.endswith(".txt.gz"):
                reader = gzip.open
            else:
                reader = open
            # TODO: if emb_file is relative, try to make it relative to the directory where the metafile is
            logger.info("Loading embeddings for %s from %s (%s words)" % (self.emb_id, emb_file, n_vocab))
            n_expected = 0
            with reader(emb_file, 'rt', encoding="utf-8") as infile:
                for line in infile:
                    if n_added == n_vocab:
                        logger.info("Got all %s embeddings needed, stopping reading the embeddings file" % (n_vocab,))
                        break
                    if n_lines == 0:
                        m = re.match(r'^\s*([0-9]+)\s+([0-9]+)\s*$', line)
                        if m:
                            n_expected = int(m.group(1))
                            self.emb_dims = int(m.group(2))
                            n_lines += 1
                            continue
                        else:
                            # assume the first line is already an embedding line and get dims from there
                            self.emb_dims = len(line.split())-1
                            n_expected = -1
                    n_lines += 1
                    if n_lines % 100000 == 0:
                        logger.info("Read lines from embeddings file: %s of %s, added words: %s of %s" %
                                    (n_lines, n_expected, n_added, n_vocab))
                    line = line.strip()
                    toidx = line.find(" ")
                    word = line[0:toidx]
                    if filterset:
                        if word not in filterset:
                            n_added += 1
                            self.stoe[word] = self.embs4line(line, toidx, self.emb_dims)
                    else:
                        if word in self.stoi:
                            n_added += 1
                            self.stoe[word] = self.embs4line(line, toidx, self.emb_dims)
        elif emb_file.endswith(".vocab") or emb_file.endswith(".npy"):
            raise Exception("TODO: format .vocab/.npy not yet implemented!")
        elif emb_file.endswith(".gensim"):
            import gensim
            gensimmodel = gensim.models.KeyedVectors.load(emb_file, mmap='r')
            # now copy over only the embeddings we actually need
            # TODO: !!!!
            raise Exception(".gensim format for embeddings not yet implemented")
        else:
            raise Exception("Embeddings file must have one of the extensions: .txt, .txt.gz, .vocab, .npy")
        self.embeddings_loaded = True
        logger.info("Embeddings for \"%s\" loaded: %s, dims=%s" % (self.emb_id, n_added, self.emb_dims))
        #if self.stoe is not None and "the" in self.stoe:
        #    print("DEBUG: embeddings for the", self.stoe["the"], file=sys.stderr)

    def get_embeddings(self):
        """Return a numpy matrix of the embeddings in the order of the indices. If this is called
        before finish() an exception is raised"""
        self.check_finished("get_embeddings")
        return self.embeddings

    @staticmethod
    def rnd_vec(dims, strng=None, as_numpy=True):
        """Returns a random vector of the given dimensions where each dimension is from a gaussian(0,1)
        If str is None, the vector is dependent on the current numpy random state. If a string is given,
        then the random state is seeded with a number derived from the string first, so the random vector
        will always be the same for that string and number of dimensions."""
        if str:
            np.random.seed(hash(strng) % (2**32-1))
        vec = np.random.randn(dims).astype(np.float32)
        if as_numpy:
            return vec
        else:
            return list(vec)

    def zero_vec(self, as_numpy=True):
        if as_numpy:
            return np.zeros(self.emb_dims, np.float32)
        else:
            return list(np.zeros(self.emb_dims, np.float32))

    def add_counts(self, counts):
        """Incrementally add additional counts to the vocabulary. This can be done only before the finish
        method is called"""
        self.check_nonfinished("add_counts")
        self.freqs.update(counts)

    def set_emb_minfreq(self, min_freq=1):
        self.check_nonfinished("set_emb_minfreq")
        self.emb_minfreq = min_freq

    def set_max_size(self, max_size=None):
        self.check_nonfinished("set_max_size")
        self.max_size = max_size

    def set_emb_id(self, embid):
        self.check_nonfinished("set_emb_id")
        self.emb_id = embid

    def set_emb_file(self, file):
        self.check_nonfinished("set_emb_file")
        self.emb_file = file

    def set_emb_dims(self, dim):
        self.check_nonfinished("set_emb_dims")
        self.emb_dims = dim

    def finish(self, remove_counts=True, remove_embs=True):
        """Build the actual vocab instance, it can only be used properly to look-up things after calling
        this method, but no parameters can be changed nor counts added after this."""
        self.check_nonfinished("finish")

        # if the emb_train parameter was never set, try to come up with a sensible default here:
        # - if a file is specified, use the setting "no" for now,
        # - otherwise use "yes"
        if not self.emb_train:
            # we set our own default here: if a file is specified, then emb_train is no, otherwise
            # it is yes.
            if self.emb_file:
                self.emb_train = "no"
            else:
                self.emb_train = "yes"

        # make sure the padding "word" which may be included in the frequencies gets ignored.
        # we do this by removing the entry at this point, if it exists
        if self.have_oov:
            if self.oov_string in self.freqs:
                logger.debug("OOV symbol removed from frequencies, freq=%s, id=%s" %
                             (self.freqs[self.pad_string], self.emb_id))
                del self.freqs[self.oov_string]
        if self.have_pad:
            if self.pad_string in self.freqs:
                logger.debug("Pad symbol removed from frequencies, freq=%s, id=%s" %
                             (self.freqs[self.pad_string], self.emb_id))
                del self.freqs[self.pad_string]

        # go through the entries and put all the keys satisfying the emb_minfreq limit into a list
        # put all the words not satisfying the restriction in the filtered_words set
        filtered_words = set()
        self.itos = []
        print("Finishing vocab ", self.emb_id, "before filtering: ", len(self.freqs), file=sys.stderr)
        for s in self.freqs:
            if self.freqs[s] >= self.emb_minfreq:
                self.itos.append(s)
            else:
                filtered_words.add(s)
        # sort the keys by frequency, then alphabetically in reverse order
        # (so to achieve this sort, sort first alphabetically, then by frequency)
        self.itos = sorted(self.itos)
        print("Vocab", self.emb_id, "after minfreq filtering: ", len(self.itos), file=sys.stderr)
        self.itos = sorted(self.itos, reverse=True, key=lambda x: self.freqs[x])
        # add the additional symbols at the beginning, first and always at index 0, the pad symbol, except
        # when no_pad is True
        if self.no_special_indices:
            pass  # do nothing what we have is all we need
        elif self.pad_index_only:
            self.itos = [self.pad_string] + self.itos
        else:
            self.itos = [self.pad_string] + [self.oov_string] + self.itos
        # trim the itos according to max_size and add any trimmed words to the filtered_words set
        if self.max_size and len(self.itos) > self.max_size:
            for w in self.itos[self.max_size:]:
                filtered_words.add(w)
            self.itos = self.itos[:self.max_size]
        # now create the reverse map
        self.stoi = defaultdict(int)
        for i, s in enumerate(self.itos):
            self.stoi[s] = i
        self.n = len(self.itos)
        print("Vocab", self.emb_id, "final: ", self.n, file=sys.stderr)
        if self.emb_train == "onehot":
            # set the emb_dims to the number of values we have, but if we have a padding symbol,
            # do not include it in the dimensions
            if self.have_pad:
                self.emb_dims = self.n - 1
            else:
                self.emb_dims = self.n

        # print("DEBUG: initial itos for ",self.emb_id,"is",self.itos[0:20], file=sys.stderr)

        if not self.emb_file and not self.emb_dims:
            self.emb_dims = int(math.log2(self.n)**1.8)+1

        # if needed, load the embeddings: if the set we pass on is empty, only the embeddings in the vocab
        # are loaded, otherwise all embeddings not in the filter set are loaded
        if self.emb_file:
            self.load_embeddings(self.emb_file, filterset=filtered_words)
            # the embeddings loaded already have been filtered, but our own vocab may need
            # to get cleaned up now: if filtered_words is empty, then we want to keep only
            # those words in our vocab which also occur in the embeddings.
            # Otherwise (this is when we learn a mapping), we keep all our own vocab words which
            # do not occur in the embeddings, but we create some random embedding vectors for them
            if filtered_words:
                # ok, we loaded all embeddings except the filtered vocab words, so we first also
                # create the random embedding vectors for the words in our vocab not in the embeddings file
                for s in self.stoi:
                    if s not in self.stoe:
                        self.stoe[s] = self.rnd_vec(dims=self.emb_dims, strng=s)
                # we have also loaded embeddings for words which are not in our vocabulary, we need to add
                # those to our index. First create the set of words that need to get added
                words2add = set()
                for s in self.stoe:
                    if s not in self.stoi:
                        words2add.add(s)
                # now append those words at the end of the itos array and also add them to the stoi dict
                for s in words2add:
                    self.itos.append(s)
                    self.stoi[s] = self.n
                    self.n += 1
            else:
                # we have loaded only those embeddings which are in the vocab, but now we have some
                # vocab words left which are not in the embeddings: remove them!
                self.itos = [w for w in self.itos if w == self.pad_string or w == self.oov_string or w in self.stoe]
                self.stoi = defaultdict(int)
                for i, s in enumerate(self.itos):
                    self.stoi[s] = i
                self.n = len(self.itos)
            # now if necessary add the padding and oov vectors
            if self.have_oov:
                self.stoe[self.oov_string] = self.rnd_vec(dims=self.emb_dims, strng=self.oov_string)
            if self.have_pad:
                self.stoe[self.pad_string] = self.zero_vec()
            # now we should have an embedding vector for each word in stoi/itos, so we should now
            # create the actual embeddings matrix
            self.embeddings = np.zeros((self.n, self.emb_dims), np.float32)
            for s in self.stoi:
                idx = self.stoi[s]
                emb = self.stoe[s]
                self.embeddings[idx] = emb
        else:  # no emb file
            if self.emb_train == "onehot":
                self.embeddings = np.zeros((self.n, self.emb_dims), np.float32)
                fromindex = 0
                if self.have_pad:
                    fromindex = 1
                j = 0
                for i in range(fromindex, self.n):
                    self.embeddings[i, j] = 1.0
                    j += 1
            else:
                self.embeddings = np.random.randn(self.n, self.emb_dims).astype(np.float32)
                # override the padding vector with a zero vector if needed:
                if not self.no_special_indices:
                    self.embeddings[0] = np.zeros(self.emb_dims, np.float32)

        # print("DEBUG: itos new=", self.itos, file=sys.stderr)
        # print("DEBUG: stoi new=", self.stoi, file=sys.stderr)
        # print("DEBUG: stoe new=", self.stoe, file=sys.stderr)

        # if self.stoe is not None and "the" in self.stoe:
        #    print("DEBUG: embeddings for the", self.stoe["the"], file=sys.stderr)

        # cleanup what we do not need any more
        if remove_embs:
            self.stoe = None
        if remove_counts:
            self.freqs = None
        self.finished = True
        # print("DEBUG: final itos for ",self.emb_id,"is",self.itos[0:20], file=sys.stderr)

    def idx2string(self, idx):
        """Return the string for this index"""
        self.check_finished("idx2string")
        if idx >= len(self.itos):
            raise Exception("Vocab: index larger than vocabulary size")
        else:
            return self.itos[idx]

    def string2idx(self, string):
        self.check_finished("string2idx")
        if string in self.stoi:
            return self.stoi[string]  # NOTE: the pad string is in there!
        else:
            if self.have_oov:
                return self.stoi[self.oov_string]
            else:
                # not a proper word no oov character, for now throw an exception, this should probablly never happen
                raise Exception("String not found in vocab and do not have OOV symbol either: %s" % string)

    def string2emb(self, string):
        self.check_finished("string2emb")
        if self.embeddings is None:
            raise Exception("Cannot get embedding vector, no embeddings matrix")
        if string in self.stoi:
            return self.embeddings[self.stoi[string]]
        else:
            if self.have_oov:
                return self.embeddings[self.stoi[self.oov_string]]
            else:
                raise Exception("Cannot return embedding vector, string not found and no OOV symbol: %s" % string)

    def string2onehot(self, thestring):
        """return a one-hot vector for the string. If we have an oov index, return that for unknown words,
        otherwise raise and exception. If the string is the padding string, return an all zero vector.
        NOTE: this can be called even if the emb_train parameter was not equal to 'onehot' when creating the
        vocabulary. In that case, there may be an OOV symbol in the vocab and the onehot vector generated will
        contain it as its first dimension."""
        if not self.finished:
            raise Exception("Vocab %r has not been finished!" % self)
        vec = self.zero_onehotvec()
        if self.have_pad and thestring == self.pad_string:
            return vec
        if thestring in self.stoi:
            l = self.stoi[thestring]
        elif self.have_oov:
            l = self.stoi[self.oov_string]
        else:
            raise Exception("String not found in vocab and no OOV symbol: %s" % (thestring,))
        if self.have_pad:
            l -= 1
        vec[l] = 1.0
        return vec

    def zero_onehotvec(self):
        l = len(self.itos)
        if self.have_pad:
            l -= 1
        return [0.0] * l

    def onehot2string(self, vec):
        if not self.finished:
            raise Exception("Vocab has not been finished!")
        s = sum(vec)
        if self.have_pad and s == 0.0:
            return self.pad_string
        if s != 1.0:
            raise Exception("Not a proper one-hot vector: %s" % (vec,))
        idx = vec.index(1.0)
        if self.have_pad:
            idx += 1
        return self.itos[idx]

    def count(self, strng):
        """Return the count/frequency for the given word. NOTE: after finish() this will return 0 for any words
        that have been removed because of one of the filter criteria!!"""
        if self.freqs:
            c = self.freqs.get(strng)
            if c:
                return c
            else:
                return 0
        else:
            raise Exception("Cannot retrieve count, data has been removed")

    def size(self):
        """Return the total number of entries in the vocab, including any special symbols"""
        return len(self.itos)

    def __str__(self):
        return self.__repr__()+":nentries=%d" % len(self.stoi)

    def __repr__(self):
        tmp_entries = [self.itos[i] for i in range(min(len(self.itos),20))]
        return "Vocab(n=%d,emb_id=%r,emb_train=%r,emb_file=%r,emb_dims=%d,entries=%s)" % \
               (len(self.stoi), self.emb_id, self.emb_train, self.emb_file, self.emb_dims, tmp_entries)
