from __future__ import print_function
from gatelfdata import Dataset
from gatelfdata import Features
import os
import sys

ROOTDIR = os.path.join(os.path.dirname(__file__), '.')
TESTDIR = os.path.join(ROOTDIR, 'tests')
DATADIR = os.path.join(TESTDIR, 'data')

TESTFILE1 = os.path.join(DATADIR, "class-ionosphere.meta.json")
TESTFILE2 = os.path.join(DATADIR, "class-ngram-sp1.meta.json")
TESTFILE3 = os.path.join(DATADIR, "class-window-pos1.meta.json")
TESTFILE4 = os.path.join(DATADIR, "seq-pos1.meta.json")

if len(sys.argv) != 2:
   raise Exception("Need one parameter: meta file")

file = sys.argv[1]

ds = Dataset(file)

valset = ds.convert_to_file()
for b in ds.batches_converted(batch_size=20, as_numpy=False, pad_left=True):
  print("Batch: len=", len(b))
  print("Batch: data=", b)
