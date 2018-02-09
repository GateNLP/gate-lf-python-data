from gatelfdata.dataset import Dataset
from gatelfdata.features import Vocabs
from gatelfdata.features import Features
import unittest
import os
import sys
import logging

logger = logging.getLogger("gatelfdata")
logger.setLevel(logging.ERROR)
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

    def test_t1(self):
        logger.info("Running Tests4Features1test1/test_t1")
        ds = Dataset(TESTFILE1)
        metafile = ds.metafile
        # print("debug: metafile=", metafile, file=sys.stderr)
        assert Dataset._modified4meta(metafile, name_part="train.val").endswith("train.val.json")

    def test_t2(self):
        logger.info("Running Tests4Features1test1/test_t2")
        ds = Dataset(TESTFILE1)
        features = Features(ds.meta)
        s = features.size()
        assert s == 34
        it1 = iter(ds.instances_converted(train=False, convert=True))
        rec = next(it1)
        logger.info("TESTFILE1 info=%r" % ds.get_info())
        logger.info("TESTFILE1 rec1=%r" % rec)
        # we expect rec to be a pair: indep and dep
        indep, dep = rec
        # the indep part has as many values as there are features here
        assert len(indep) == 34
        # the dep part is the encoding for two nominal classes, we use
        # a one-hot encoding always for now, so this should be a vector
        # of length 2
        # print("DEBUG: test_t2 dep=", dep, file=sys.stderr)
        assert len(dep) == 2
        Vocabs.init()

    def test_t3(self):
        logger.info("Running Tests4Features1test1/test_t3")
        ds = Dataset(TESTFILE2)

        it0 = iter(ds.instances_original())
        inst0 = next(it0)
        indep, dep = inst0
        # print("DEBUG: indep=", indep, file=sys.stderr)
        assert indep == [['invincible', 'is', 'a', 'wonderful', 'movie', '.']]
        assert dep == "pos"


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
        it1 = iter(ds.instances_converted(train=False, convert=True))
        rec = next(it1)
        logger.info("TESTFILE2 rec1=%r", rec)
        logger.info("TESTFILE2 info=%r" % ds.get_info())
        (indep1_it, dep1_it) = rec
        ngram1_it = indep1_it[0]
        logger.info("TESTFILE2 ngram1_it=%r", ngram1_it)
        assert len(ngram1_it) == 6
        assert ngram1_it[0] == 3542
        assert ngram1_it[1] == 8
        assert len(dep1_it) == 2
        Vocabs.init()

    def test_t4(self):
        logger.info("Running Tests4Features1test1/test_t4")
        ds = Dataset(TESTFILE3)
        logger.info("TESTFILE3 attrs=%r", ds.meta.get("featureInfo").get("attributes"))
        features = Features(ds.meta)
        logger.info("TESTFILE3 features=%r", features)
        it1 = iter(ds.instances_original())
        rec = next(it1)
        logger.info("TESTFILE3 rec1=%r", rec)
        logger.info("TESTFILE3 info=%r" % ds.get_info())

        # we expect rec to be a pair: indep and dep
        # indep, dep = rec
        # the indep part has as many values as there are features here
        # assert len(indep) == 34
        # the dep part is the encoding for two nominal classes, we use
        # a one-hot encoding always for now, so this should be a vector
        # of length 2
        # assert len(dep) == 2
        Vocabs.init()

    def test_t5(self):
        logger.info("Running Tests4Features1test1/test_t5")
        ds = Dataset(TESTFILE4)
        it1 = iter(ds.instances_converted(train=False, convert=True))
        rec = next(it1)

        indep, dep = rec
        logger.info("TESTFILE4: indep=%r" % indep)
        logger.info("TESTFILE4: dep=%r" % dep)
        logger.info("TESTFILE4 info=%r" % ds.get_info())
        # the first row is a sequence of 3 elements, with 18 independent
        # features and one of 17 different targets
        # so we should convert this into 18 features which each now should have 3 values
        # and 3 onehot vectors for the class
        assert len(indep) == 18
        assert len(dep) == 3
        # check if the class is actually ADJ for all three targets
        dep1 = dep[0]
        dep2 = dep[1]
        dep3 = dep[2]
        t11 =  ds.target.vocab.onehot2string(dep1)
        assert t11 == "ADJ"
        t12 = ds.target.vocab.onehot2string(dep2)
        assert t12 == "ADJ"
        t13 = ds.target.vocab.onehot2string(dep3)
        assert t13 == "ADJ"
        Vocabs.init()

    def test_t6(self):
        logger.info("Running Tests4Features1test1/test_t6")
        ds = Dataset(TESTFILE2)
        ds.split(convert=True, keep_orig=True, validation_size=3, random_seed=1)
        # check if getting the batches and validation sets works
        valset_orig = ds.validation_set_orig()
        # print("DEBUG: valset_orig=%s" % valset_orig, file=sys.stderr)
        assert len(valset_orig) == 3
        vorigi2 = valset_orig[1]
        assert vorigi2 == [[['a', 'very', 'well-made', ',', 'funny', 'and', 'entertaining', 'picture', '.']], 'pos']
        valset_conv = ds.validation_set_converted()
        # print("DEBUG: valset_conv=%s" % valset_conv, file=sys.stderr)
        assert len(valset_conv) == 3
        vconvi2 = valset_conv[1]

        assert vconvi2 == [[[4, 83, 1529, 3, 74, 5, 189, 174, 1]], [0.0, 1.0]]
        valset_conv_b = ds.validation_set_converted(as_batch=True)
        # print("DEBUG: valset_conv_b=%s" % (valset_conv_b,), file=sys.stderr)
        # we expect a tuple for indep and dep
        assert len(valset_conv_b) == 2
        indep1, dep1 = valset_conv_b
        # the indep part should now have lenth one because there is only one features
        assert len(indep1) == 1
        # there should be 3 values for that first feature
        assert len(indep1[0]) == 3
        # get a batch of original data
        bitb1 = ds.batches_original(train=True, batch_size=4, reshape=False)
        batch_orig1 = next(iter(bitb1))
        # print("DEBUG: batch_orig1=%s" % (batch_orig1,), file=sys.stderr)
        # if reshape was False, this is just a list of instances in original format
        assert len(batch_orig1) == 4
        assert batch_orig1[1] == [[['rife', 'with', 'nutty', 'cliches', 'and', 'far', 'too', 'much', 'dialogue', '.']], 'neg']
        bitb2 = ds.batches_original(train=True, batch_size=4, reshape=True)
        batch_orig2 = next(iter(bitb2))
        # print("DEBUG: batch_orig2=%s" % (batch_orig2,), file=sys.stderr)
        # if reshape was True, this is a tuple where the first element is the list of features
        assert len(batch_orig2) == 2
        featurelist1 = batch_orig2[0]
        feature1 = featurelist1[0]
        assert feature1[1] == ['rife', 'with', 'nutty', 'cliches', 'and', 'far', 'too', 'much', 'dialogue', '.']
        bconvb1 = ds.batches_converted(train=True, batch_size=4, reshape=False)
        batch_conv1 = next(iter(bconvb1))
        # print("DEBUG: batch_conv1=%s" % (batch_conv1,), file=sys.stderr)
        assert len(batch_conv1) == 4
        assert batch_conv1[1] ==[[[6693, 16, 6468, 543, 5, 167, 50, 58, 236, 1]], [1.0, 0.0]]
        bconvb2 = ds.batches_converted(train=True, batch_size=4, reshape=True)
        batch_conv2 = next(iter(bconvb2))
        # print("DEBUG: batch_conv2=%s" % (batch_conv2,), file=sys.stderr)
        assert len(batch_conv2) == 2
        featurelist1 = batch_conv2[0]
        feature1 = featurelist1[0]
        assert feature1[1] ==[6693, 16, 6468, 543, 5, 167, 50, 58, 236, 1]
        Vocabs.init()

    def test_t7(self):
        logger.info("Running Tests4Features1test1/test_t7")
        ds = Dataset(TESTFILE3)
        ds.split(convert=True, keep_orig=True, validation_size=3, random_seed=1)
        # check if getting the batches and validation sets works
        valset_orig = ds.validation_set_orig()
        # print("DEBUG: valset_orig=%s" % valset_orig, file=sys.stderr)
        assert len(valset_orig) == 3
        vorigi2 = valset_orig[1]
        assert vorigi2 == [['you', 'think', 'this', 'place', 'is', 'nice', 'VERB', 'DET', 'a', 'a', 'a', 'a', 'a', 'a',
                           '', 'nk', 'is', 'ce', '', 'ce', '', 'ink', '', 'ace', '', ''], 'NOUN']
        valset_conv = ds.validation_set_converted()
        # print("DEBUG: valset_conv=%s" % valset_conv, file=sys.stderr)
        assert len(valset_conv) == 3
        vconvi2 = valset_conv[1]
        assert vconvi2 == [[13, 157, 25, 104, 12, 319, 2, 6, 1, 1, 1, 1, 1, 1, 1, 151, 28, 14, 1, 14, 1, 215,
                             1, 101, 1, 1], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                             0.0, 0.0]]
        # assert vconvi2 == [[[4, 83, 1529, 3, 74, 5, 189, 174, 1]], [0.0, 1.0]]
        valset_conv_b = ds.validation_set_converted(as_batch=True)
        # print("DEBUG: valset_conv_b=%s" % (valset_conv_b,), file=sys.stderr)
        # we expect a tuple for indep and dep
        assert len(valset_conv_b) == 2
        indep1, dep1 = valset_conv_b
        # the indep part should now have lenth equal to the number of features
        assert len(indep1) == ds.nFeatures
        # there should be 3 values for that first feature
        assert len(indep1[0]) == 3
        # get a batch of original data
        bitb1 = ds.batches_original(train=True, batch_size=4, reshape=False)
        batch_orig1 = next(iter(bitb1))
        # print("DEBUG: batch_orig1=%s" % (batch_orig1,), file=sys.stderr)
        # if reshape was False, this is just a list of instances in original format
        assert len(batch_orig1) == 4
        assert batch_orig1[1] == [['Bill', 'Bradford', 'in', 'Credit', 'are', 'supposed', 'PROPN', 'ADP', 'Aa', 'Aa',
                                   'a', 'Aa', 'a', 'a', 'll', 'rd', '', 'it', '', 'ed', '', 'ord', '',
                                   'dit', '', 'sed'], 'NOUN']
        bitb2 = ds.batches_original(train=True, batch_size=4, reshape=True)
        batch_orig2 = next(iter(bitb2))
        # print("DEBUG: batch_orig2=%s" % (batch_orig2,), file=sys.stderr)
        # if reshape was True, this is a tuple where the first element is the list of features
        assert len(batch_orig2) == 2
        featurelist1 = batch_orig2[0]
        feature1 = featurelist1[0]
        assert feature1[1] == 'Bill'
        bconvb1 = ds.batches_converted(train=True, batch_size=4, reshape=False)
        batch_conv1 = next(iter(bconvb1))
        # print("DEBUG: batch_conv1=%s" % (batch_conv1,), file=sys.stderr)
        assert len(batch_conv1) == 4
        assert batch_conv1[1] == [[1210, 1495, 9, 796, 23, 3075, 7, 3, 2, 2, 1, 2, 1, 1, 20, 54, 1, 86, 1, 2, 1, 391,
                                   1, 300, 1, 77], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                    0.0, 0.0, 0.0, 0.0]]
        bconvb2 = ds.batches_converted(train=True, batch_size=4, reshape=True)
        batch_conv2 = next(iter(bconvb2))
        # print("DEBUG: batch_conv2=%s" % (batch_conv2,), file=sys.stderr)
        assert len(batch_conv2) == 2
        featurelist1 = batch_conv2[0]
        feature1 = featurelist1[0]
        assert feature1[1] == 1210
        Vocabs.init()

    def test_t8(self):
        logger.info("Running Tests4Features1test1/test_t8")
        ds = Dataset(TESTFILE3, reuse_files=True)
        # print("debug orig_train_file=", ds.orig_train_file, file=sys.stderr)


if __name__ == '__main__':
    unittest.main()
