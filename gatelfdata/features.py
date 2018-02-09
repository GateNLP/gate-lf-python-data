from . feature import Feature
import sys
import logging
from . vocabs import Vocabs


class Features(object):

    def __init__(self, meta):
        logger = logging.getLogger(__name__)
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
        # TODO: we need to handle the Vocab creation differently: instead of creating each
        # vocab as part of the make, we need to first collect the stats for all
        # all attrs that share the same vocab so that the final counts and entries for the
        # vocab are the union of all the individual stats!
        for f in self.meta["features"]:
            dt = f["datatype"]
            attrnr = f["attrid"]
            attrinfo = attrs[attrnr]
            attrcode = attrinfo.get("code")
            if dt == "nominal":
                Vocabs.setup_vocab(attrinfo, stats[f["name"]])
            Vocabs.finish()
        for f in self.meta["features"]:
            dt = f["datatype"]
            attrnr = f["attrid"]
            fname = f["name"]
            attrkind = f["kind"]
            # get a bit more info from the corresponding attribute metadata
            attrinfo = attrs[attrnr]
            fstats = stats[fname]
            thefeature = Feature.make(fname, dt, attrinfo, fstats)
            logger.debug("Features: appending feature=%r", thefeature)
            self.features.append(thefeature)

    def _convert_featurevec(self, valuelist, idxs=None):
        if not idxs and (len(valuelist) != len(self.features)):
            raise Exception("Wrong number of values passed, expected", len(self.features), "got", len(valuelist))
        if idxs and len(idxs) > len(valuelist):
            raise Exception("Wrong number of idxs passed, got", len(idxs), "but got values:", len(valuelist))
        if idxs and len(idxs) > len(self.features):
            raise Exception("Wrong number of idxs passed, got", len(idxs), "but got features:", len(self.features))
        if idxs:
            valueslist = [valuelist[i] for i in idxs]
            features = [self.features[i] for i in idxs]
        else:
            features = self.features
        values = []
        for i in range(len(features)):
            res = features[i](valuelist[i])
            values.append(res)
        return values

    def __iter__(self):
        return iter(self.features)

    def __call__(self, valuelist, idxs=None):
        # For a feature vector:
        # this will go through each input and run it through the stored feature
        # instance, and the values will get put into the result list and returned
        # Note that for ngram attributes, the "value" to put into the list is itself a list
        # (of embedding indices).
        # For a sequence of feature vectors: will return a list/vector
        # for each feature where each element corresponds to a sequence element
        # So the representation gets changed from a list of feature vectors
        # of values to a list of values for each feature
        if self.isSequence:
            # for now we do this in an easy to understand but maybe slow way:
            # first go convert each of the feature vectors in the sequence
            # then convert the resulting list of lists
            seqofvecs = []
            for el in valuelist:
                vals4featurevec = self._convert_featurevec(el, idxs=idxs)
                seqofvecs.append(vals4featurevec)
            # now each element in sequofvecs should have as many elements
            # as there are features, just transpose that matrix
            return [l for l in map(list, zip(*seqofvecs))]
        else:
            values = self._convert_featurevec(valuelist, idxs=idxs)
            return values

    def size(self):
        return len(self.features)

    def __repr__(self):
        l = [f.__repr__() for f in self.features]
        return "Features(features=%r)" % l

    def __str__(self):
        l = [f.__str__() for f in self.features]
        return "Features("+",".join(l)+")"

    def pretty_print(self, file=sys.stdout):
        print("Features:", file=file)
        for f in self.features:
            print("  ", f, file=file)
