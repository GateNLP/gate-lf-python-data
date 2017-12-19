from . dataset import Dataset
from . featurengram import FeatureNgram
from . feature import Feature

class Features(object):
    def __init__(self, meta):
        # initialisation consists of going through the meta info and
        # creating all the individual feature instances and storing them
        # in here in a list.
        # TODO: do we need to know here if we get a sequence of feature vectors?
        # NOTE: we should go through the actual features, not the attributes, so we do
        # not really need anything that represents an attributelist since this is
        # just a fixed number of simple attributes.
        # meta: either a string or the meta information already read in and parsed.
        if type(meta) == Dataset:
            self.meta = meta.meta
        elif type(meta) == str:
            self.meta = Dataset.load_meta(meta)
        elif type(meta) == dict:
            self.meta = meta
        else:
            raise Exception("Cannot interpret parameter as meta information")
        # now we have the meta, create the list of features
        self.features = []
        attrs = self.meta["featureInfo"]["attributes"]
        stats = self.meta["featureStats"]
        for f in self.meta["features"]:
            dt = f["datatype"]
            attrnr = f["attrid"]
            fname = f["name"]
            attrkind = f["kind"]
            # get a bit more info from the corresponding attribute metadata
            attrinfo = attrs[attrnr]
            fstats = stats["fname"]
            thefeature = Feature.make(fname, dt, attrinfo, fstats)
            self.features.append(thefeature)

    def __call__(self, valuelist):
        ## this will go through each input and run it through the stored feature
        ## instance, and the values will get put into the result list and returned
        ## Note that for ngram attributes, the "value" to put into the list is itself a list
        ## (of embedding indices).
        ## TODO: handle sequences of fvs by returning a list of lists.
        if len(valuelist) != len(self.features):
            raise Exception("Wrong number of values passed, expected",len(self.features),"got",len(valuelist))
        values = []
        for i in range(len(self.features)):
            values.append(self.features[i](valuelist[i]))
        return values

    def size(self):
        return len(self.features)

    def __str__(self):
        l = [f.__str__() for f in self.features]
        return "Features("+",".join(l)+")"