#!/usr/bin/env python

# test loading a file
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from gatelfdata import Dataset

if len(sys.argv) != 2:
   raise Exception("Need one parameter: meta file")

file = sys.argv[1]

ds = Dataset(file)
