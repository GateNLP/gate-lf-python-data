from __future__ import print_function
from gatelfdata.dataset import Dataset
from gatelfdata.features import Features
import unittest
import json
import os
import sys
from gatelfdata.p2comp import open

TESTDIR = os.path.join(os.path.dirname(__file__), '.')
DATADIR = os.path.join(TESTDIR, 'data')
print("DEBUG: datadir is ", TESTDIR, file=sys.stderr)

TESTFILE1 = os.path.join(DATADIR, "class-ionosphere.meta.json")
TESTFILE2 = os.path.join(DATADIR, "class-ngram-sp1.meta.json")
TESTFILE3 = os.path.join(DATADIR, "class-window-pos1.meta.json")
TESTFILE4 = os.path.join(DATADIR, "seq-pos1.meta.json")


class Tests4Features1(unittest.TestCase):
    """Tests for the ionosphere dataset"""
    meta = None
    # we really only need to load that once since all the tests will just read it

    def setUp(self):
        if not Tests4Features1.meta:
            print("\nTEST: loading meta file", TESTFILE1, file=sys.stderr)
            with open(TESTFILE1, "rt", encoding="utf-8") as inp:
                Tests4Features1.meta = json.load(inp)
        self.meta = Tests4Features1.meta

    def tearDown(self):
        pass


class Tests4Features1test1(Tests4Features1):

    def test_t1(self):
        # check if the number of features is correct
        features = Features(self.meta)
        print("Running Tests4Features1test1/test_t1, got ",features)
        s = features.size()
        assert s == 22


class TestFeatureNgram(Tests4Features1):
    pass


class TestFeatureAttributeList(Tests4Features1):
    pass


class TestDataset1(unittest.TestCase):
    def test_ds1(self):
        print("\nTEST TestDataset1/ds1: opening dataset", TESTFILE1, file=sys.stderr)
        ds = Dataset(TESTFILE1)


if __name__ == '__main__':
    unittest.main()
