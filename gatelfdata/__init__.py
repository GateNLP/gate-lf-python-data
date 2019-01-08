# https://stackoverflow.com/questions/10971033/backporting-python-3-openencoding-utf-8-to-python-2
__version__ = '0.2'
from gatelfdata.features import Features
from gatelfdata.featureboolean import FeatureBoolean
from gatelfdata.featurengram import FeatureNgram
from gatelfdata.featurenominalembs import FeatureNominalEmbs
from gatelfdata.featurenumeric import FeatureNumeric
from gatelfdata.dataset import Dataset
from gatelfdata.target import Target
from gatelfdata.targetnominal import TargetNominal
from gatelfdata.vocab import Vocab
