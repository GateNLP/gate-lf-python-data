from gatelfdata.dataset import Dataset
from gatelfdata.vocabs import Vocabs
from gatelfdata.vocab import Vocab
from gatelfdata.features import Features
import unittest
import os
import sys
import logging
import numpy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)

# add file handler to gatelfdata and our own loggers
filehandler = logging.FileHandler("test_api.log")
logger1 = logging.getLogger("gatelfdata")
logger1.setLevel(logging.INFO)
logger1.addHandler(filehandler)
logger.addHandler(filehandler)

TESTDIR = os.path.join(os.path.dirname(__file__), '.')
DATADIR = os.path.join(TESTDIR, 'data')
print("DEBUG: datadir is ", TESTDIR, file=sys.stderr)

TESTFILE1 = os.path.join(DATADIR, "class-ionosphere.meta.json")
TESTFILE2 = os.path.join(DATADIR, "class-ngram-sp1.meta.json")
TESTFILE3 = os.path.join(DATADIR, "class-window-pos1.meta.json")
TESTFILE4 = os.path.join(DATADIR, "seq-pos1.meta.json")
TESTFILE5 = os.path.join(DATADIR, "seq-pos2.meta.json")
EMBFILE20_50TXT = os.path.join(DATADIR, "emb-mini20-25.txt")
EMBFILE20_50GZ = os.path.join(DATADIR, "emb-mini20-25.txt.gz")


# class Test4Debugging(unittest.TestCase):


class TestUtils(unittest.TestCase):

    def test_utils_1(self):
        l1 = [1, 2, 3]
        Dataset.pad_list_(l1, 10, pad_value=0)
        assert l1 == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
        l1 = [1, 2, 3]
        Dataset.pad_list_(l1, 10, pad_value=0, pad_left=True)
        assert l1 == [0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
        l2 = [[1, 2], [], [1, 2, 3, 4], [1], [2, 3]]
        Dataset.pad_matrix_(l2, pad_value=0)
        assert l2 == [[1, 2, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4], [1, 0, 0, 0], [2, 3, 0, 0]]


class TestFeatures1(unittest.TestCase):

    def test_features1_scaling1(self):
        ds = Dataset(TESTFILE1)
        feature3 = ds.features[3]
        val = feature3(0.5)
        assert val == 0.5
        # print("DEBUG: none / val for 0.5=%s" % val, file=sys.stderr)
        val = feature3(0.5, normalize="minmax")
        assert val == 0.75
        # print("DEBUG: minmax / val for 0.5=%s" % val, file=sys.stderr)
        val = feature3(0.5, normalize="meanvar")
        assert val > 2.3381 and val < 2.3382
        # print("DEBUG: meanvar / val for 0.5=%s" % val, file=sys.stderr)


class TestVocab1(unittest.TestCase):

    def test_vocab1(self):
        d1 = {"a": 12, "b": 13, "c": 1, "d": 2, "x": 12}
        v1 = Vocab(d1, max_size=6, emb_minfreq=2, emb_train="yes")
        v1.finish(remove_counts=False)
        # logger.info("\nTestVocab/test_vocab1: v1.itos=%r" % v1.itos)
        # logger.info("\nTestVocab/test_vocab1: v1.stoi=%r" % v1.stoi)
        assert len(v1.itos) == 6
        assert len(v1.stoi) == 6
        assert "a" in v1.stoi
        assert v1.idx2string(3) == "a"
        assert v1.string2idx("a") == 3
        assert v1.string2idx("b") == 2
        vec = v1.string2onehot("a")
        # logger.info("\nTestVocab/test_vocab1: onehot(a)=%r" % vec)
        assert len(vec) == 5
        assert vec[0] == 0.0
        assert vec[1] == 0.0
        assert vec[2] == 1.0
        assert vec[3] == 0.0
        c = v1.count("d")
        assert c == 2

    def test_vocab1b(self):
        d1 = {"a": 12, "b": 13, "c": 1, "d": 2, "x": 12}
        v1 = Vocab(d1, emb_train="onehot")
        v1.finish(remove_counts=False)
        logger.debug("\nTestVocab/test_vocab1b: v1.itos=%r" % v1.itos)
        logger.debug("\nTestVocab/test_vocab1b: v1.stoi=%r" % v1.stoi)
        emb = v1.string2emb("a")
        logger.debug("\nTestVocab/test_vocab1b: emb(a)=%r" % emb)
        assert numpy.array_equal(emb, numpy.array([0.0, 1.0, 0.0, 0.0, 0.0]))
        assert len(v1.itos) == 6
        assert len(v1.stoi) == 6
        assert "a" in v1.stoi
        assert v1.idx2string(2) == "a"
        assert v1.string2idx("a") == 2
        assert v1.string2idx("b") == 1
        vec = v1.string2onehot("a")
        logger.debug("\nTestVocab/test_vocab1b: onehot(a)=%r" % vec)
        assert len(vec) == 5
        assert vec[0] == 0.0
        assert vec[1] == 1.0
        assert vec[2] == 0.0
        assert vec[3] == 0.0

    def test_vocab1c(self):
        d1 = {"a": 12, "b": 13, "c": 1, "d": 2, "x": 12}
        v1 = Vocab(d1, emb_train="onehot", no_special_indices=True)
        v1.finish(remove_counts=False)
        logger.debug("\nTestVocab/test_vocab1c: v1.itos=%r" % v1.itos)
        logger.debug("\nTestVocab/test_vocab1c: v1.stoi=%r" % v1.stoi)
        emb =v1.string2emb("a")
        logger.debug("\nTestVocab/test_vocab1c: emb(a)=%r" % emb)
        assert numpy.array_equal(emb,numpy.array([0.0, 1.0, 0.0, 0.0, 0.0]))
        assert len(v1.itos) == 5
        assert len(v1.stoi) == 5
        assert "a" in v1.stoi
        assert v1.idx2string(1) == "a"
        assert v1.string2idx("a") == 1
        assert v1.string2idx("b") == 0
        vec = v1.string2onehot("a")
        logger.debug("\nTestVocab/test_vocab1c: onehot(a)=%r" % vec)
        assert len(vec) == 5
        assert vec[0] == 0.0
        assert vec[1] == 1.0
        assert vec[2] == 0.0
        assert vec[3] == 0.0


    def test_vocab2(self):
        # test using embedding file
        # but first some fake counts for the 20 words in there

        cnt1 = {'was': 20, 'as': 10, 'las': 12, 'mas': 1, 'please': 33, 'say': 40, 'sama': 1, 'always': 21, 'mais': 2,
                'because': 33, 'esta': 5, 'last': 11, 'thanks': 13, 'ass': 13, 'has': 55, 'pas': 1, 'said': 25,
                'bisa': 2, 'same': 13, 'days': 21}
        v1 = Vocab(cnt1, emb_train="yes", emb_file=EMBFILE20_50TXT)
        v1.finish()
        # this should contain all the entries from cnt1 plus the pad and OOV indices now
        assert len(v1.itos) == len(cnt1)+2
        # we should be able to get all the embedding vectors as one big matrix
        allembs = v1.get_embeddings()
        logger.debug("allembs=%s" % allembs)


class Tests4Batches(unittest.TestCase):

    def test_reshape_class1(self):
        # 3 instances, each having 2 features
        batch1 = [
            [[11, 12], 1],
            [[21, 22], 2],
            [[31, 32], 3]
        ]
        batch1_reshape = Dataset.reshape_batch_helper(batch1, n_features=2, is_sequence=False)
        # print("DEBUG: reshape_class_1: batch1_reshape=", batch1_reshape, file=sys.stderr)
        assert batch1_reshape == ([[11, 21, 31], [12, 22, 32]], [1, 2, 3])
        batch1_reshape_np = Dataset.reshape_batch_helper(batch1, as_numpy=True, n_features=2, is_sequence=False)
        # print("DEBUG: reshape_class_1: batch1_reshape_np=%r" % (batch1_reshape_np,), file=sys.stderr)
        ## NOTE: The numpy reshape returns floats for the classes, we should check if and when this is ok

    def test_reshape_class1a(self):
        # 3 instances, each having 2 features, this time indep only
        batch1 = [
            [11, 12],
            [21, 22],
            [31, 32],
        ]
        batch1_reshape = Dataset.reshape_batch_helper(batch1, n_features=2, is_sequence=False, indep_only=True)
        # print("DEBUG: reshape_class_1a: batch1_reshape=", batch1_reshape, file=sys.stderr)
        assert batch1_reshape == [[11, 21, 31], [12, 22, 32]]
        batch1_reshape_np = Dataset.reshape_batch_helper(batch1, as_numpy=True, n_features=2, is_sequence=False, indep_only=True)
        # print("DEBUG: reshape_class_1: batch1_reshape_np=%r" % (batch1_reshape_np,), file=sys.stderr)


    def test_reshape_class2(self):
        # 3 instances, each having 2 features, both features are sequences
        # feature value numbers indicate: instance, feature number, sequence element
        # Sequence lengths of first feature: 3, 2, 1,
        # Sequence lengths of second feature: 1, 4, 2
        batch1 = [
            [[[111, 112, 113], [121]], 1],
            [[[211, 212], [221, 222, 223, 224]], 2],
            [[[311], [321, 322]], 3]
        ]
        batch1_reshape = Dataset.reshape_batch_helper(batch1, n_features=2, is_sequence=False)
        # print("DEBUG: reshape_class_2: batch1_reshape=", batch1_reshape, file=sys.stderr)
        assert batch1_reshape == ([[[111, 112, 113], [211, 212, 0], [311, 0, 0]], [[121, 0, 0, 0], [221, 222, 223, 224], [321, 322, 0, 0]]], [1, 2, 3])
        batch1_reshape_np = Dataset.reshape_batch_helper(batch1, as_numpy=True, n_features=2, is_sequence=False)
        # print("DEBUG: reshape_class_2: batch1_reshape_np=%r" % (batch1_reshape_np,), file=sys.stderr)
        ## NOTE: The numpy reshape returns floats for the classes, we should check if and when this is ok!


    def test_reshape_seq1(self):
        # test reshaping sequence learning batches
        # a simple tiny batch of 3 sequences of feature vectors, each having 2 features and a target
        # the sequence lengths are 1,4,2
        batch1 = [
            [[[111, 112]], [-11]],
            [[[211, 212], [221, 222], [231, 232], [241, 242]], [-21, -22, -23, -23]],
            [[[311, 312], [321, 322]], [-31, -32]]
        ]
        # print("DEBUG: reshape_seq1: batch1=", batch1, file=sys.stderr)
        batch1_reshape = Dataset.reshape_batch_helper(batch1, feature_types=["index", "index"], is_sequence=True)
        # print("DEBUG: reshape_seq1: batch1_reshape=\n", batch1_reshape, file=sys.stderr)
        # print("DEBUG: expected: \n",
        #       [
        #           [[111, 0, 0, 0], [211, 221, 231, 241], [311, 321, 0, 0]],
        #           [[112, 0, 0, 0], [212, 222, 232, 242], [312, 322, 0, 0]]
        #       ],
        #       [
        #           [-11, -1, -1, -1], [-21, -22, -23, -23], [-31, -32, -1, -1]
        #       ], file=sys.stderr)

        assert batch1_reshape == ([
                  [[111, 0, 0, 0], [211, 221, 231, 241], [311, 321, 0, 0]],
                  [[112, 0, 0, 0], [212, 222, 232, 242], [312, 322, 0, 0]]
              ],
              [
                  [-11, -1, -1, -1], [-21, -22, -23, -23], [-31, -32, -1, -1]
              ])

class Tests4Dataset1test1(unittest.TestCase):

    def test_t1(self):
        # logger.info("Running Tests4Dataset1test1/test_t1")
        ds = Dataset(TESTFILE1)
        metafile = ds.metafile
        # print("debug: metafile=", metafile, file=sys.stderr)
        assert Dataset._modified4meta(metafile, name_part="train.val").endswith("train.val.json")


    def test_t1_1(self):
        # logger.info("Running Tests4Dataset1test1/test_t1_1")
        # test overriding the meta settings: override the "token" embeddings to have emb_dims=123
        # and emb_train=mapping and an emb_file
        ds = Dataset(TESTFILE3, config={"embs":"token:123:yes"})
        feats = ds.features
        # get the vocab of the first feature
        vocf1 = feats[0].vocab
        # print("DEBUG: emb_id=", vocf1.emb_id, "emb_dims=", vocf1.emb_dims, "emb_train=", vocf1.emb_train,
        #      "emb_file=", vocf1.emb_file, file=sys.stderr)
        assert vocf1.emb_id == "token"
        assert vocf1.emb_dims == 123

    def test_t1_2(selfs):
        # logger.info("Running Tests4Dataset1test1/test_t1_2")
        # check a simple sequence dataset with just one nominal feature without any embeddings definitions
        ds = Dataset(TESTFILE5)
        feats = ds.features
        vocf1 = feats[0].vocab
        # print("Vocab for feature 1 is ", vocf1, file=sys.stderr)

    def test_t2(self):
        # logger.info("Running Tests4Dataset1test1/test_t2")
        ds = Dataset(TESTFILE1)
        features = ds.features
        s = features.size()
        assert s == 34
        it1 = iter(ds.instances_converted(train=False, convert=True))
        rec = next(it1)
        logger.debug("TESTFILE1 info=%r" % ds.get_info())
        logger.debug("TESTFILE1 rec1=%r" % rec)
        # we expect rec to be a pair: indep and dep
        indep, dep = rec
        # print("DEBUG: rec=", rec, file=sys.stderr)
        # the indep part has as many values as there are features here
        assert len(indep) == 34
        # the dep part is the encoding for two nominal classes,
        assert dep == 0
        # if we would have converted the target as_onehot then we
        # would have gotten a vector instead:
        # assert len(dep) == 2

    def test_t3(self):
        # logger.info("Running Tests4Dataset1test1/test_t3")
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
        logger.debug("Original  indep1=%r", indep1)
        logger.debug("Converted indep1=%r", indep1_conv)
        ngram1 = indep1_conv[0]
        assert len(ngram1) == 6
        # print("DEBUG ngram1[0]=", ngram1[0], file=sys.stderr)
        assert ngram1[0] == 3543
        assert ngram1[1] == 9
        it1 = iter(ds.instances_converted(train=False, convert=True))
        rec = next(it1)
        logger.debug("TESTFILE2 rec1=%r", rec)
        logger.debug("TESTFILE2 info=%r" % ds.get_info())
        (indep1_it, dep1_it) = rec
        ngram1_it = indep1_it[0]
        logger.debug("TESTFILE2 dep_it=%r", dep1_it)
        # print("DEBUG dep1_it=", dep1_it, file=sys.stderr)
        assert len(ngram1_it) == 6
        assert ngram1_it[0] == 3543
        assert ngram1_it[1] == 9
        assert dep1_it == 1

    def test_t4(self):
        # logger.info("Running Tests4Dataset1test1/test_t4")
        ds = Dataset(TESTFILE3)
        logger.debug("TESTFILE3 attrs=%r", ds.meta.get("featureInfo").get("attributes"))
        # Features constructor finishes the vocab, so we need to re-initilize
        features = ds.features
        logger.debug("TESTFILE3 features=%r", features)
        it1 = iter(ds.instances_original())
        rec = next(it1)
        logger.debug("TESTFILE3 rec1=%r", rec)
        logger.debug("TESTFILE3 info=%r" % ds.get_info())

        # we expect rec to be a pair: indep and dep
        # indep, dep = rec
        # the indep part has as many values as there are features here
        # assert len(indep) == 34
        # the dep part is the encoding for two nominal classes, we use
        # a one-hot encoding always for now, so this should be a vector
        # of length 2
        # assert len(dep) == 2

    def test_t5(self):
        # logger.info("Running Tests4Dataset1test1/test_t5")
        ds = Dataset(TESTFILE4)
        it1 = iter(ds.instances_converted(train=False, convert=True))
        rec = next(it1)

        indep, dep = rec
        logger.debug("TESTFILE4: indep=%r" % indep)
        logger.debug("TESTFILE4: dep=%r" % dep)
        logger.debug("TESTFILE4 info=%r" % ds.get_info())
        # the first row is a sequence of 3 elements, with 18 independent
        # features and one of 17 different targets
        # so we should convert this into 18 features which each now should have 3 values
        # and 3 onehot vectors for the class

        assert len(dep) == 3
        assert len(indep) == 3   # 3 elements in the sequence
        assert len(indep[0]) == 18
        assert len(indep[1]) == 18
        assert len(indep[2]) == 18
        # check if the class is actually ADJ for all three targets
        dep1 = dep[0]
        dep2 = dep[1]
        dep3 = dep[2]
        t11 = ds.target.vocab.idx2string(dep1)
        assert t11 == "ADJ"
        t12 = ds.target.vocab.idx2string(dep2)
        assert t12 == "ADJ"
        t13 = ds.target.vocab.idx2string(dep3)
        assert t13 == "ADJ"
        # test getting batches in non-reshaped form
        bit1 = ds.batches_converted(train=False, convert=True, batch_size=2, reshape=False)
        biter1 = iter(bit1)
        batch1 = next(biter1)
        # print("DEBUG: TESTFILE4 batch/noreshape=%s" % (batch1,), file=sys.stderr)
        assert len(batch1) == 2
        # test getting batches in reshaped form
        bit2 = ds.batches_converted(train=False, convert=True, batch_size=2, reshape=True)
        biter2 = iter(bit2)
        batch2 = next(biter2)
        # print("DEBUG: TESTFILE4 batch/noreshape=%s" % (batch1,), file=sys.stderr)
        bindep, bdep = batch2
        assert len(bindep) == 18
        assert len(bdep) == 2
        assert len(bindep[0]) == 2

    def test_t6(self):
        # logger.info("Running Tests4Dataset1test1/test_t6")
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
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DEBUG: vconvi2=", vconvi2, file=sys.stderr)
        assert vconvi2 == [[[5, 84, 1530, 4, 75, 6, 190, 175, 2]], 1]
        valset_conv_b = ds.validation_set_converted(as_batch=True)
        # print("DEBUG: valset_conv_b=%s" % (valset_conv_b,), file=sys.stderr)
        # we expect a tuple for indep and dep
        assert len(valset_conv_b) == 2
        indep1, dep1 = valset_conv_b
        # the indep part should now have lenth one because there is only one feature
        assert len(indep1) == 1
        # there should be 3 values for that first feature
        # print("DEBUG: indep1[0]=%r" % (indep1[0]), file=sys.stderr)
        assert len(indep1[0]) == 3
        # get a batch of original data
        bitb1 = ds.batches_original(train=True, batch_size=4, reshape=False)
        batch_orig1 = next(iter(bitb1))
        # print("DEBUG: batch_orig1=%s" % (batch_orig1,), file=sys.stderr)
        # if reshape was False, this is just a list of instances in original format
        assert len(batch_orig1) == 4
        assert batch_orig1[1] == [[['rife', 'with', 'nutty', 'cliches', 'and', 'far', 'too', 'much', 'dialogue', '.']],
                                  'neg']
        bitb2 = ds.batches_original(train=True, batch_size=4, reshape=True)
        batch_orig2 = next(iter(bitb2))
        # print("DEBUG: batch_orig2=%s" % (batch_orig2,), file=sys.stderr)
        # if reshape was True, this is a tuple where the first element is the list of features
        assert len(batch_orig2) == 2
        featurelist1 = batch_orig2[0]
        feature1 = featurelist1[0]
        # print("DEBUG: feature1[1]=%s" % (feature1[1],), file=sys.stderr)
        assert feature1[1] == ['rife', 'with', 'nutty', 'cliches', 'and', 'far', 'too', 'much', 'dialogue', '.', '',
                               '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                               '', '', '']
        bconvb1 = ds.batches_converted(train=True, batch_size=4, reshape=False)
        batch_conv1 = next(iter(bconvb1))
        # print("DEBUG: batch_conv1=%s" % (batch_conv1,), file=sys.stderr)
        assert len(batch_conv1) == 4
        # print("DEBUG: batch_conv1[1]=%s" % (batch_conv1[1],), file=sys.stderr)
        assert batch_conv1[1] == [[[6694, 17, 6469, 544, 6, 168, 51, 59, 237, 2]], 0]
        bconvb2 = ds.batches_converted(train=True, batch_size=4, reshape=True)
        batch_conv2 = next(iter(bconvb2))
        # print("DEBUG: batch_conv2=%s" % (batch_conv2,), file=sys.stderr)
        assert len(batch_conv2) == 2
        featurelist1 = batch_conv2[0]
        feature1 = featurelist1[0]
        # print("DEBUG: feature1[1]=%s" % (feature1[1],), file=sys.stderr)
        assert feature1[1] == [6694, 17, 6469, 544, 6, 168, 51, 59, 237, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_t7(self):
        # logger.info("Running Tests4Dataset1test1/test_t7")
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
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DEBUG: vconvi2=", vconvi2, file=sys.stderr)
        assert vconvi2 == [[13, 157, 25, 104, 12, 319, 2, 5, 2, 2, 2, 2, 2, 2, 0, 151, 28, 14, 0, 14, 0, 215, 0, 101, 0, 0], 0]
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
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DEBUG: !!!batch_conv1[1]=%s" % (batch_conv1[1],), file=sys.stderr)
        assert len(batch_conv1) == 4
        # TODO: check why some indices changed between previously and now and if this is till correct!
        assert batch_conv1[1] == [[1210, 1495, 9, 796, 23, 3075, 6, 3, 3, 3, 2, 3, 2, 2, 20, 54, 0, 86, 0, 2, 0, 391, 0, 300, 0, 77], 0]
        bconvb2 = ds.batches_converted(train=True, batch_size=4, reshape=True)
        batch_conv2 = next(iter(bconvb2))
        # print("DEBUG: batch_conv2=%s" % (batch_conv2,), file=sys.stderr)
        assert len(batch_conv2) == 2
        featurelist1 = batch_conv2[0]
        feature1 = featurelist1[0]
        assert feature1[1] == 1210

    def test_t8(self):
        # logger.info("Running Tests4Dataset1test1/test_t8")
        ds = Dataset(TESTFILE3, reuse_files=True)
        # print("debug orig_train_file=", ds.orig_train_file, file=sys.stderr)
        num_idxs = ds.get_float_feature_idxs()
        # print(file=sys.stderr)
        # print("File", TESTFILE3, file=sys.stderr)
        # print("DEBUG float_idxs=", num_idxs, file=sys.stderr)
        nom_idxs = ds.get_index_feature_idxs()
        # print("DEBUG index_idxs=", nom_idxs, file=sys.stderr)
        ngr_idxs = ds.get_indexlist_feature_idxs()
        # print("DEBUG indexlist_idxs=", ngr_idxs, file=sys.stderr)

        ds = Dataset(TESTFILE4, reuse_files=True)
        # print("debug orig_train_file=", ds.orig_train_file, file=sys.stderr)
        num_idxs = ds.get_float_feature_idxs()
        # print(file=sys.stderr)
        # print("File", TESTFILE4, file=sys.stderr)
        # print("DEBUG float_idxs=", num_idxs, file=sys.stderr)
        nom_idxs = ds.get_index_feature_idxs()
        # print("DEBUG index_idxs=", nom_idxs, file=sys.stderr)
        ngr_idxs = ds.get_indexlist_feature_idxs()
        # print("DEBUG indexlist_idxs=", ngr_idxs, file=sys.stderr)

        ds = Dataset(TESTFILE2, reuse_files=True)
        # print("debug orig_train_file=", ds.orig_train_file, file=sys.stderr)
        num_idxs = ds.get_float_feature_idxs()
        # print(file=sys.stderr)
        # print("File", TESTFILE2, file=sys.stderr)
        # print("DEBUG float_idxs=", num_idxs, file=sys.stderr)
        nom_idxs = ds.get_index_feature_idxs()
        # print("DEBUG index_idxs=", nom_idxs, file=sys.stderr)
        ngr_idxs = ds.get_indexlist_feature_idxs()
        # print("DEBUG indexlist_idxs=", ngr_idxs, file=sys.stderr)

        ds = Dataset(TESTFILE1, reuse_files=True)
        # print("debug orig_train_file=", ds.orig_train_file, file=sys.stderr)
        num_idxs = ds.get_float_feature_idxs()
        # print(file=sys.stderr)
        # print("File", TESTFILE1, file=sys.stderr)
        # print("DEBUG float_idxs=", num_idxs, file=sys.stderr)
        nom_idxs = ds.get_index_feature_idxs()
        # print("DEBUG index_idxs=", nom_idxs, file=sys.stderr)
        ngr_idxs = ds.get_indexlist_feature_idxs()
        # print("DEBUG indexlist_idxs=", ngr_idxs, file=sys.stderr)

    def test_t9(self):
        # logger.info("Running Tests4Dataset1test1/test_t9")
        ds1 = Dataset(TESTFILE4, reuse_files=False,  targets_need_padding=False)
        ds1.target.set_as_onehot(True)
        batch_reshape = ds1.batches_converted(train=False, batch_size=4, reshape=True, convert=True)
        b1r = next(iter(batch_reshape))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DEBUG: the whole batch:\n",b1r,file=sys.stderr)
        targets = b1r[1]
        indeps = b1r[0]
        # ok, we want one element for each example in the batch
        assert len(targets) == 4
        # get the length of the first sequence from indep by looking at the number of elements
        # of the first feature of the first instance. Note that all sequences of feature values
        # and targets should be padded to the maximum sequence length!
        feature1 = indeps[0]
        len1 = len(feature1[0])
        # check the length of the target sequences
        assert len(targets[0]) == len1
        assert len(targets[1]) == len1
        assert len(targets[2]) == len1
        assert len(targets[3]) == len1
        # TODO: check why this is supposed to be one hot vectors and not indices here!!
        # for each target check that all entries are one-hot vectors of the same length
        for i in range(4):
            for j in range(len1):
                val = targets[i][j]
                assert isinstance(val, list)
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DEBUG: len val ",len(val), "val=",val, file=sys.stderr)
                assert len(val) == 17


if __name__ == '__main__':
    unittest.main()
