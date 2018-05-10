import sys
import logging
from . featurenominal1ofk import FeatureNominal1ofk
from . featurenumeric import FeatureNumeric
from . featurenominalembs import FeatureNominalEmbs
from . featureboolean import FeatureBoolean
from . featurengram import FeatureNgram


class Features(object):


    def make_feature(self, fname, datatype, attribute, featurestats, vocabs):
        """Helper function to create a specific feature gets called as part of __init__"""
        kind = attribute["featureCode"]
        logger = logging.getLogger(__name__)
        logger.debug("Making feature for kind/name/type/attr: %r/%r/%r/%r", kind, fname, datatype, attribute)
        if kind == "N":
            # create an ngram feature, based on a simple feature of type nominal
            ret = FeatureNgram(fname, attribute, featurestats, vocabs.get_vocab(attribute))
        else:
            # create a simple feature of the correct type
            if datatype == "nominal":
                # create a nominal feature of the correct kind for either
                # embedding or one-hot coding
                # This is decided by the setting of the corresponding
                # embedding definition.
                emb_train = attribute["codeas"]
                if emb_train == "onehot":
                    ret = FeatureNominal1ofk(fname, attribute, featurestats)
                else:
                    ret = FeatureNominalEmbs(fname, attribute, featurestats, vocabs.get_vocab(attribute))
            elif datatype == "numeric":
                # simple numeric feature
                ret = FeatureNumeric(fname, attribute, featurestats)
            elif datatype == "boolean":
                # simple boolean feature
                ret = FeatureBoolean(fname, attribute, featurestats)
            else:
                raise Exception("Odd datatype: ", datatype)
        logger.debug("Returning: %r", ret)
        return ret


    def __init__(self, meta, vocabs):
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
        self.vocabs = vocabs
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
            # attrcode = attrinfo.get("code")
            if dt == "nominal":
                self.vocabs.setup_vocab(attrinfo, stats[f["name"]])
        self.vocabs.finish()
        for f in self.meta["features"]:
            dt = f["datatype"]
            attrnr = f["attrid"]
            fname = f["name"]
            # attrkind = f["kind"]
            # get a bit more info from the corresponding attribute metadata
            attrinfo = attrs[attrnr]
            fstats = stats[fname]
            thefeature = self.make_feature(fname, dt, attrinfo, fstats, self.vocabs)
            logger.debug("Features: appending feature=%r", thefeature)
            self.features.append(thefeature)

    def _convert_featurevec(self, valuelist, idxs=None, normalize=None):
        if not idxs and (len(valuelist) != len(self.features)):
            raise Exception("Wrong number of values passed, expected", len(self.features), "got", len(valuelist))
        if idxs and len(idxs) > len(valuelist):
            raise Exception("Wrong number of idxs passed, got", len(idxs), "but got values:", len(valuelist))
        if idxs and len(idxs) > len(self.features):
            raise Exception("Wrong number of idxs passed, got", len(idxs), "but got features:", len(self.features))
        if idxs:
            valuelist = [valuelist[i] for i in idxs]
            features = [self.features[i] for i in idxs]
        else:
            features = self.features
        values = []
        for i in range(len(features)):
            res = features[i](valuelist[i], normalize=normalize)
            values.append(res)
        return values

    def __iter__(self):
        return iter(self.features)

    def __getitem__(self, item):
        return self.features[item]

    def __call__(self, valuelist, idxs=None, normalize=None):
        # For a feature vector:
        # this will go through each input and run it through the stored feature
        # instance, and the values will get put into the result list and returned
        # Note that for ngram attributes, the "value" to put into the list is itself a list
        # (of embedding indices).
        # For a sequence of feature vectors: each feature vector gets converted
        # in the normal way, targets as well
        # NOTE: not sure yet how to handle nominals that are onehot encoded! In some cases
        # we want to instances in some we want the vectors .. see featurenominal1ofk
        if self.isSequence:
            out_indep = []
            for fv in valuelist:
                out_indep.append(self._convert_featurevec(fv, idxs=idxs))
            return out_indep
        else:
            values = self._convert_featurevec(valuelist, idxs=idxs)
            return values

    def __call__OLD(self, valuelist, idxs=None):
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
        fl = [f.__repr__() for f in self.features]
        return "Features(features=%r)" % fl

    def __str__(self):
        fl = [f.__str__() for f in self.features]
        return "Features("+",".join(fl)+")"

    def pretty_print(self, file=sys.stdout):
        print("Features:", file=file)
        for f in self.features:
            print("  ", f, file=file)
