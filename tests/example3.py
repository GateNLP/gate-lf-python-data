from __future__ import print_function
from gatelfdata import Dataset
import sys
import json

if len(sys.argv) != 2:
   raise Exception("Need one parameter: meta file")

file = sys.argv[1]

ds = Dataset(file)

valset = ds.convert_to_file()
it = iter(ds.instances_as_string())
for n in range(20):
    b = []
    print("BATCH: ",n)
    for i in range(2):
      print("INSTANCE: ",i)
      instance = next(it)
      print("Instance: ",instance)
      converted = ds.convert_instance(json.loads(instance))
      print("Converted: ",converted)
      b.append(converted)
    batch1=ds.reshape_batch(b)
    print("Size2 batch: ",batch1)
    print()
