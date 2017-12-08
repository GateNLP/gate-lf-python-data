from __future__ import print_function
from gatelfdata import Dataset, Features, FeatureNgram, FeatureAttribute, FeatureAttributeList
import unittest
import json
import os
import sys

TESTDIR = os.path.join(os.path.dirname(__file__), '.')
print("DEBUG: testdir is ", TESTDIR, file=sys.stderr)

TESTFILE1 = os.path.join(TESTDIR, "crvd1.meta.json")

class Tests4Features(unittest.TestCase):
    meta = None
    # we really only need to load that once since all the tests will just read it
    def setUp(self):
        if not Tests4Features.meta:
            print("\nTEST: loading the crvd1 meta file", file=sys.stderr)
            with open(TESTFILE1, "rt", encoding="utf-8") as inp:
                Tests4Features.meta = json.load(inp)
        self.meta = Tests4Features.meta

    def tearDown(selfself):
        pass

class TestFeatureAttribute(Tests4Features):
    def test_fa1(self):
        print("\nTEST: TestFeatureAttribute/fa1", file=sys.stderr)
        fa = FeatureAttribute()

class TestFeatureNgram(Tests4Features):
    pass

class TestFeatureAttributeList(Tests4Features):
    pass

class TestDataset1(unittest.TestCase):
    def test_ds1(self):
        print("\nTEST TestDataset1/ds1: opening dataset crvd1", file=sys.stderr)
        ds = Dataset(TESTFILE1)


if __name__ == '__main__':
    unittest.main()
