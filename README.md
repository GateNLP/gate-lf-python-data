# Library for handling the (dense) data created by the GATE LearningFramework plugin

A library for simplifying reading and converting the JSON files created by the
GATE LearningFramework plugin.

Main functionality:
* Use instances in the original format an converted to numeric-only format. The
  numeric only format represents nominal values as indices for a vocabulary or
  as one-hot vectors
* Iterate over instances either in original format or in converted format
* Split the data into a validation set and a training set
* Iterate over batches of data, either in original format or in converted format
* Reshape batches into "features-first" format

## Usage examples

The following code shows how to get the validation set and how to iterate
over batches from the training set in converted format:
```
from gatelfdata import Dataset
ds = Dataset(the_met_file_path)
# split into 5% validation and 95% training data, store in converted format
ds.split(convert=True, keep_orig=False, validation_part=0.05)
# get the validation set in, reshaped in the same way as the batches
valset = ds.validation_set_converted(as_batch=True)
# iterate over batches of size 100
for batch in ds.batches_converted(train=True, batch_size=100)
    # do something with the batch, e.g. train a network
    pass
```
