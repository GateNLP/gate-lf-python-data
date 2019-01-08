# gate-lf-python-data - A Library for using JSON data from the GATE LearningFramework in Python

This is a python library that deals with using training files created in "dense JSON format" 
by the GATE LearningFramework plugin. The library is meant to get used by other Python libraries
which actually implement or interface to learning algorithms, e.g. gate-lf-pytorch-json.

Main functionality:
* Use instances in the original format an converted to numeric-only format. The
  numeric only format represents nominal values as indices for a vocabulary or
  as one-hot vectors
* Generate vocabularies and handle pre-calculated embeddings for them
* Handle sequences of features
* Iterate over instances either in original format or in converted format
* Split the data into a validation set and a training set
* Iterate over batches of data, either in original format or in converted format
* Reshape batches into "features-first" format

## Requirements

Requirements:
* Python 3.5 or later - this does not work with python 2.x!
* Python package `numpy`

## Overview of the documentation:

* [Usage Examples](UsageExamples)
* [File Formats](FileFormats)
* [The Generated Python Documentation](pythondoc)
