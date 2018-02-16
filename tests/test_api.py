from gatelfdata.dataset import Dataset
from gatelfdata.vocabs import Vocabs
from gatelfdata.vocab import Vocab
from gatelfdata.features import Features
import unittest
import os
import sys
import logging

logger = logging.getLogger("gatelfdata")
logger.setLevel(logging.WARN)
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
EMBFILE20_50TXT = os.path.join(DATADIR, "emb-mini20-25.txt")
EMBFILE20_50GZ = os.path.join(DATADIR, "emb-mini20-25.txt.gz")


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

class TestVocab1(unittest.TestCase):

    def test_vocab1(self):
        d1 = {"a": 12, "b": 13, "c": 1, "d": 2, "x": 12}
        v1 = Vocab(d1, add_symbols=["<<START>>"], max_size=6, min_freq=2, emb_train="yes")
        v1.finish()
        logger.info("\nTestVocab/test_vocab1: v1.itos=%r" % v1.itos)
        logger.info("\nTestVocab/test_vocab1: v1.stoi=%r" % v1.stoi)
        assert len(v1.itos) == 6
        assert len(v1.stoi) == 6
        assert "a" in v1.stoi
        assert v1.idx2string(3) == "b"
        assert v1.string2idx("a") == 4
        assert v1.string2idx("b") == 3
        assert v1.string2idx("<<START>>") == 2
        vec = v1.string2onehot("a")
        assert len(vec) == 6
        assert vec[0] == 0.0
        assert vec[1] == 0.0
        assert vec[2] == 0.0
        assert vec[3] == 0.0
        assert vec[4] == 1.0
        assert vec[5] == 0.0
        c = v1.count("d")
        assert c == 2

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
        # logger.info("allembs=%s" % allembs)

    def test_vocab2(self):
        # test using embedding file
        # but first some fake counts for the 20 words in there
        cnt1 = {'was': 20, 'as': 10, 'las': 12, 'mas': 1, 'please': 33, 'say': 40, 'sama': 1, 'always': 21, 'mais': 2,
                'because': 33, 'esta': 5, 'last': 11, 'thanks': 13, 'ass': 13, 'has': 55, 'pas': 1, 'said': 25,
                'bisa': 2, 'same': 13, 'days': 21}
        v1 = Vocab(cnt1, emb_train="yes", emb_file=EMBFILE20_50TXT, oov_vec_from="maxfreqavg", oov_vec_maxfreq=2)
        v1.finish()
        # this should contain all the entries from cnt1 plus the pad and OOV indices but minus the ones
        # that got removed because the frequency is <= 2
        logger.info("itos=%r" % v1.itos)
        assert len(v1.itos) == len(cnt1)+2-5
        # we should be able to get all the embedding vectors as one big matrix
        allembs = v1.get_embeddings()
        # logger.info("allembs shape=%s" % (allembs.shape,))
        assert allembs.shape[0] == len(cnt1)+2-5
        assert allembs.shape[1] == 25
        # logger.info("allembs=%s" % allembs)


class Tests4Batches(unittest.TestCase):

    def test_reshape1(self):
        # test reshaping sequence learning batches
        # a simple tiny batch of 3 sequences of feature vectors, each having 2 features and a target
        # the sequence lengths are 1,4,2
        batch1 = [
            [[[111, 112]], [11]],
            [[[211, 212], [221, 222], [231, 232], [241, 242]], [21, 22, 23, 23]],
            [[[311, 312], [321, 322]], [31, 32]]
        ]
        batch1_reshape = Dataset.reshape_batch_helper(batch1, nFeatures=2, nClasses=7, isSequence=True)
        print("DEBUG: batch1_reshape=", batch1_reshape, file=sys.stderr)

class Tests4Features1test1(unittest.TestCase):

    def test_t1(self):
        logger.info("Running Tests4Features1test1/test_t1")
        ds = Dataset(TESTFILE1)
        metafile = ds.metafile
        # print("debug: metafile=", metafile, file=sys.stderr)
        assert Dataset._modified4meta(metafile, name_part="train.val").endswith("train.val.json")


    def test_t1_1(self):
        logger.info("Running Tests4Features1test1/test_t1_1")
        # test overriding the meta settings: override the "token" embeddings to have emb_dims=123
        # and emb_train=mapping and an emb_file
        ds = Dataset(TESTFILE3, override_meta_embs={"emb_id": "token", "emb_dims": 123, "emb_train": "yes"})
        feats = ds.features
        # get the vocab of the first feature
        vocf1 = feats[0].vocab
        # print("DEBUG: emb_id=", vocf1.emb_id, "emb_dims=", vocf1.emb_dims, "emb_train=", vocf1.emb_train,
        #      "emb_file=", vocf1.emb_file, file=sys.stderr)
        assert vocf1.emb_id == "token"
        assert vocf1.emb_dims == 123



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
        assert dep == 0
        # if we would have converted the target as_onehot then we
        # would have gotten a vector instead:
        # assert len(dep) == 2

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
        # print("DEBUG ngram1[0]=", ngram1[0], file=sys.stderr)
        assert ngram1[0] == 3543
        assert ngram1[1] == 9
        it1 = iter(ds.instances_converted(train=False, convert=True))
        rec = next(it1)
        logger.info("TESTFILE2 rec1=%r", rec)
        logger.info("TESTFILE2 info=%r" % ds.get_info())
        (indep1_it, dep1_it) = rec
        ngram1_it = indep1_it[0]
        logger.info("TESTFILE2 ngram1_it=%r", ngram1_it)
        assert len(ngram1_it) == 6
        assert ngram1_it[0] == 3543
        assert ngram1_it[1] == 9
        assert dep1_it == 1
        # assert len(dep1_it) == 2

    def test_t4(self):
        logger.info("Running Tests4Features1test1/test_t4")
        ds = Dataset(TESTFILE3)
        logger.info("TESTFILE3 attrs=%r", ds.meta.get("featureInfo").get("attributes"))
        # Features constructor finishes the vocab, so we need to re-initilize
        Vocabs.init()
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
        t11 =  ds.target.vocab.idx2string(dep1)
        assert t11 == "ADJ"
        t12 = ds.target.vocab.idx2string(dep2)
        assert t12 == "ADJ"
        t13 = ds.target.vocab.idx2string(dep3)
        assert t13 == "ADJ"

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
        # print("DEBUG: vconvi2=", vconvi2, file=sys.stderr)
        # assert vconvi2 == [[[4, 83, 1529, 3, 74, 5, 189, 174, 1]], [0.0, 1.0]]
        assert vconvi2 == [[[5, 84, 1530, 4, 75, 6, 190, 175, 2]], 1]
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
        # assert batch_conv1[1] ==[[[6693, 16, 6468, 543, 5, 167, 50, 58, 236, 1]], [1.0, 0.0]]
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
        # print("DEBUG: vconvi2=", vconvi2, file=sys.stderr)
        # assert vconvi2 == [[14, 158, 26, 105, 13, 320, 3, 7, 2, 2, 2, 2, 2, 2, 0, 152, 29, 15, 0, 15, 0, 216,
        #                    0, 102, 0, 0], 0]
        assert vconvi2 == [[14, 158, 26, 105, 13, 320, 1, 5, 2, 2, 2, 2, 2, 2, 0, 152, 29, 15, 0, 15, 0, 216, 0, 102,
                            0, 0], 0]
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
        # print("DEBUG: !!!batch_conv1[1]=%s" % (batch_conv1[1],), file=sys.stderr)
        assert len(batch_conv1) == 4
        assert batch_conv1[1] == [[1211, 1496, 10, 797, 24, 3076, 6, 2, 3, 3, 2, 3, 2, 2, 21, 55, 0, 87, 0, 3, 0,
                                  392, 0, 301, 0, 78], 0]
        bconvb2 = ds.batches_converted(train=True, batch_size=4, reshape=True)
        batch_conv2 = next(iter(bconvb2))
        # print("DEBUG: batch_conv2=%s" % (batch_conv2,), file=sys.stderr)
        assert len(batch_conv2) == 2
        featurelist1 = batch_conv2[0]
        feature1 = featurelist1[0]
        assert feature1[1] == 1211

    def test_t8(self):
        logger.info("Running Tests4Features1test1/test_t8")
        ds = Dataset(TESTFILE3, reuse_files=True)
        # print("debug orig_train_file=", ds.orig_train_file, file=sys.stderr)
        num_idxs = ds.get_numeric_feature_idxs()
        print(file=sys.stderr)
        print("File", TESTFILE3, file=sys.stderr)
        print("DEBUG num_idxs=", num_idxs, file=sys.stderr)
        nom_idxs = ds.get_nominal_feature_idxs()
        print("DEBUG nom_idxs=", nom_idxs, file=sys.stderr)
        ngr_idxs = ds.get_ngram_feature_idxs()
        print("DEBUG ngr_idxs=", ngr_idxs, file=sys.stderr)

        ds = Dataset(TESTFILE4, reuse_files=True)
        # print("debug orig_train_file=", ds.orig_train_file, file=sys.stderr)
        num_idxs = ds.get_numeric_feature_idxs()
        print(file=sys.stderr)
        print("File", TESTFILE4, file=sys.stderr)
        print("DEBUG num_idxs=", num_idxs, file=sys.stderr)
        nom_idxs = ds.get_nominal_feature_idxs()
        print("DEBUG nom_idxs=", nom_idxs, file=sys.stderr)
        ngr_idxs = ds.get_ngram_feature_idxs()
        print("DEBUG ngr_idxs=", ngr_idxs, file=sys.stderr)

        ds = Dataset(TESTFILE2, reuse_files=True)
        # print("debug orig_train_file=", ds.orig_train_file, file=sys.stderr)
        num_idxs = ds.get_numeric_feature_idxs()
        print(file=sys.stderr)
        print("File", TESTFILE2, file=sys.stderr)
        print("DEBUG num_idxs=", num_idxs, file=sys.stderr)
        nom_idxs = ds.get_nominal_feature_idxs()
        print("DEBUG nom_idxs=", nom_idxs, file=sys.stderr)
        ngr_idxs = ds.get_ngram_feature_idxs()
        print("DEBUG ngr_idxs=", ngr_idxs, file=sys.stderr)

        ds = Dataset(TESTFILE1, reuse_files=True)
        # print("debug orig_train_file=", ds.orig_train_file, file=sys.stderr)
        num_idxs = ds.get_numeric_feature_idxs()
        print(file=sys.stderr)
        print("File", TESTFILE1, file=sys.stderr)
        print("DEBUG num_idxs=", num_idxs, file=sys.stderr)
        nom_idxs = ds.get_nominal_feature_idxs()
        print("DEBUG nom_idxs=", nom_idxs, file=sys.stderr)
        ngr_idxs = ds.get_ngram_feature_idxs()
        print("DEBUG ngr_idxs=", ngr_idxs, file=sys.stderr)


if __name__ == '__main__':
    unittest.main()
