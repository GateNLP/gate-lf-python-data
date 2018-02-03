from __future__ import print_function
from gatelfdata import Dataset
import sys

if len(sys.argv) != 2:
   raise Exception("Need one parameter: meta file")

file = sys.argv[1]

ds = Dataset(file)

valset = ds.convert_to_file()
for b in ds.batches_converted(batch_size=20, as_numpy=False, pad_left=True):
  print("Batch: len=", len(b))
  print("Batch: data=", b)
