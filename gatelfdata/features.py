from . feature import Feature
import sys

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
        self.meta = meta
        self.isSequence = meta["isSequence"]
        if self.isSequence:
            self.seq_max = meta["sequLengths.max"]
            self.seq_avg = meta["sequLengths.mean"]
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
            fstats = stats[fname]
            thefeature = Feature.make(fname, dt, attrinfo, fstats)
            self.features.append(thefeature)

    def _convert_featurevec(self, valuelist):
        if len(valuelist) != len(self.features):
            raise Exception("Wrong number of values passed, expected", len(self.features), "got", len(valuelist))
        values = []
        for i in range(len(self.features)):
            values.append(self.features[i](valuelist[i]))
        return values

    def __call__(self, valuelist):
        ## For a feature vector:
        ## this will go through each input and run it through the stored feature
        ## instance, and the values will get put into the result list and returned
        ## Note that for ngram attributes, the "value" to put into the list is itself a list
        ## (of embedding indices).
        ## For a sequence of feature vectors: will return a list/vector
        ## for each feature where each element corresponds to a sequence element
        ## So the representation gets changed from a list of feature vectors
        ## of values to a list of values for each feature
        if self.isSequence:
            # for now we do this in an easy to understand but maybe slow way:
            # first go convert each of the feature vectors in the sequence
            # then convert the resulting list of lists
            seqofvecs = []
            for el in valuelist:
                vals4featurevec = self._convert_featurevec(el)
                seqofvecs.append(vals4featurevec)
            # now each element in sequofvecs should have as many elements
            # as there are features, just transpose that matrix
            return [l for l in map(list, zip(*seqofvecs))]
        else:
            values = self._convert_featurevec(valuelist)
            return values

    def size(self):
        return len(self.features)

    def __str__(self):
        l = [f.__str__() for f in self.features]
        return "Features("+",".join(l)+")"