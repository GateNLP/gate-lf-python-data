# Examples of using the gatelfdata library

## Split data and iterate over batches in converted format

The following code shows how to get the validation set and how to iterate
over batches from the training set in converted format:
```python
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
