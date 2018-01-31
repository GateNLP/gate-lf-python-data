from __future__ import print_function
from gatelfdata.dataset import Dataset
from gatelfdata.features import Features
import unittest
import os
import sys
import logging

logger = logging.getLogger("gatelfdata")
logger.setLevel(logging.DEBUG)
streamhandler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)
filehandler = logging.FileHandler("test_api.log")
logger.addHandler(filehandler)

TESTDIR = os.path.join(os.path.dirname(__file__), '.')
DATADIR = os.path.join(TESTDIR, 'data')
print("DEBUG: datadir is ", TESTDIR, file=sys.stderr)

TESTFILE1 = os.path.join(DATADIR, "class-ionosphere.meta.json")
TESTFILE2 = os.path.join(DATADIR, "class-ngram-sp1.meta.json")
TESTFILE3 = os.path.join(DATADIR, "class-window-pos1.meta.json")
TESTFILE4 = os.path.join(DATADIR, "seq-pos1.meta.json")

class TestVocab1(unittest.TestCase):

    def test_vocab1(self):
        from gatelfdata.vocab import Vocab
        d1 = {"a": 12, "b": 13, "c": 1, "d": 2, "x": 12}
        v1 = Vocab(d1, add_symbols=["<<START>>"], max_size=5, min_freq=2)
        v1.finish()
        logger.info("\nTestVocab/test_vocab1: v1.itos=%r" % v1.itos)
        logger.info("\nTestVocab/test_vocab1: v1.stoi=%r" % v1.stoi)
        assert len(v1.itos) == 5
        assert len(v1.stoi) == 5
        assert "a" in v1.stoi
        assert v1.idx2string(2) == "b"
        assert v1.string2idx("a") == 3
        assert v1.string2idx("b") == 2
        assert v1.string2idx("<<START>>") == 1
        vec = v1.string2onehot("a")
        assert len(vec) == 5
        assert vec[0] == 0.0
        assert vec[1] == 0.0
        assert vec[2] == 0.0
        assert vec[3] == 1.0
        assert vec[4] == 0.0
        c = v1.count("d")
        assert c == 2


class Tests4Features1test1(unittest.TestCase):

    def test_t2(self):
        logger.info("Running Tests4Features1test1/test_t2")
        ds = Dataset(TESTFILE1)
        features = Features(ds.meta)
        s = features.size()
        assert s == 34
        it1 = iter(ds.instances_as_data())
        rec = next(it1)
        logger.info("TESTFILE1 rec1=%r" % rec)
        # we expect rec to be a pair: indep and dep
        indep, dep = rec
        # the indep part has as many values as there are features here
        assert len(indep) == 34
        # the dep part is the encoding for two nominal classes, we use
        # a one-hot encoding always for now, so this should be a vector
        # of length 2
        assert len(dep) == 2

    def test_t3(self):
        logger.info("Running Tests4Features1test1/test_t3")
        ds = Dataset(TESTFILE2)

        # check low level conversion methods first
        inst1 = [[['invincible', 'is', 'a', 'wonderful', 'movie', '.']], 'pos']
        (indep1, dep1) = inst1

        indep1_conv = ds.features(indep1)
        logger.info("Original  indep1=%r", indep1)
        logger.info("Converted indep1=%r", indep1_conv)
        ngram1 = indep1_conv[0]
        assert len(ngram1) == 6
        assert ngram1[0] == 3542
        assert ngram1[1] == 8
        it1 = iter(ds.instances_as_data())
        rec = next(it1)
        logger.info("TESTFILE2 rec1=%r", rec)
        (indep1_it, dep1_it) = rec
        ngram1_it = indep1_it[0]
        logger.info("TESTFILE2 ngram1_it=%r", ngram1_it)
        assert len(ngram1_it) == 6
        assert ngram1_it[0] == 3542
        assert ngram1_it[1] == 8
        assert len(dep1_it) == 2

    def test_t4(self):
        logger.info("Running Tests4Features1test1/test_t4")
        ds = Dataset(TESTFILE3)
        logger.info("TESTFILE3 attrs=%r", ds.meta.get("featureInfo").get("attributes"))
        features = Features(ds.meta)
        logger.info("TESTFILE3 features=%r", features)
        it1 = iter(ds.instances_as_data())
        rec = next(it1)
        logger.info("TESTFILE3 rec1=%r", rec)

        # we expect rec to be a pair: indep and dep
        # indep, dep = rec
        # the indep part has as many values as there are features here
        # assert len(indep) == 34
        # the dep part is the encoding for two nominal classes, we use
        # a one-hot encoding always for now, so this should be a vector
        # of length 2
        # assert len(dep) == 2

    def test_t5(self):
        logger.info("Running Tests4Features1test1/test_t4")
        ds = Dataset(TESTFILE4)
        it1 = iter(ds.instances_as_data())
        rec = next(it1)

        indep, dep = rec
        logger.info("TESTFILE4: indep=%r" % indep)
        logger.info("TESTFILE4: dep=%r" % dep)
        # the first row is a sequence of 3 elements, with 18 independent
        # features and one of 17 different targets
        assert len(indep) == 18
        # TODO!!!! Rethink what we should return for dep: this should
        # be a sequence of one-hot vectors, so the length here is
        # equal to the sequence length
        assert len(dep) == 3
        # check if the class is actually ADJ
        # print("DEBUG: target1=", , file=sys.stderr)




if __name__ == '__main__':
    unittest.main()
