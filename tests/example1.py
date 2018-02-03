from __future__ import print_function
from gatelfdata import Dataset
import sys

if len(sys.argv) != 2:
   raise Exception("Need one parameter: meta file")

file = sys.argv[1]

ds = Dataset(file)
valset = ds.convert_to_file()
for instance in ds.instances_as_data():
  print("Instance: ",instance)
