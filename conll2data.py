#!/usr/bin/env python

# Simple program to convert CONLL format files into
# a pair of meta/data files that can be processed by this library
# and the libraries that use this library.
# Currently the only conversion method implemented is to convert
# this into a sequence tagging problem, where each instance is a
# sequence of labeled elements.

import sys
import logging
import argparse
from collections import Counter
import re
import json
import statistics
import os
import datetime

meta_info = {
    "featureNames": ["Token┆string╬A"],
    "isSequence": True,
    "featureStats": {
        "Token┆string╬A": {
            "isString": True,
            "min": 0,
            "stringCounts": None,  # This gets updated, word-frequency dict
            "max": 0,
            "variance": 0,
            "mean": 0,
            "n": None   # This gets updated, number of words in total
        }
    },
    "features":
        [{"attrid": 0, "datatype": "nominal",
          "kind": "A", "name": "Token┆string╬A"}],
    "savedOn": None, # conversion date
    "sequLengths.mean": None, # average length of sequences
    "sequLengths.min": None, # !!!
    "sequLengths.max": None, # !!!
    "sequLengths.variance": None, # !!!
    "targetStats": {
        "variance": 0,
        "min": 0,
        "max": 0,
        "n": None, # Number of occurrences of targets in total (not # classes)
        "stringCounts": None, # target-freq dict
        "isString": True
    },
    "dataFile": None, # path to the data file for the meta file
    "linesWritten": None, # number of instances in the data file
    "featureInfo": {
        "attributes": [
            {
                "emb_file": "",
                "codeas": "one_of_k",
                "emb_id": "",
                "annType": "Token",
                "missingValueValue": "",
                "missingValueTreatment": "keep",
                "featureCode": "A",
                "listsep": "",
                "withinType": "Sentence",
                "emb_train": "yes",
                "datatype": "nominal",
                "code": "A",
                "alphabet": "null",
                "emb_dims": 0,
                "name": "",
                "feature": "string",
                "featureId": 0,
                "featureName4Value": ""
            }
        ],
        "globalScalingMethod": "NONE"
    }
}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

repl = r'[0-9]'


parser = argparse.ArgumentParser()
parser.add_argument("--lc", type=str2bool, default=True, help="If the tokens should get lower-cased")
parser.add_argument("--zero", type=str2bool, default=True, help="If all the digits should get mapped to '0'")
parser.add_argument("--col", type=int, default=3, help="Columnt of the target in the input file, 0-based")
parser.add_argument("file", type=str, help="Input file")
parser.add_argument("outprefix", type=str, help="Path prefix of the output files")

args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)

logger.debug("Input file is %s" % args.file)
logger.debug("Output path prefix is %s" % args.outprefix)

outdata_name = args.outprefix + ".data.json"
outmeta_name = args.outprefix + ".meta.json"
have_docstring = False
tokens = []
targets = []
# the following statistics need to get collected
stringCounts_token = Counter()   # token->frequency
n_token = 0     # total number of words

stringCounts_target = Counter()  # target->frequency
n_target = 0    # total number of targets
seqLengths = []  # to calculate min, max, mean, variance

def count_elements(token, target):
    global n_token, n_target, stringCounts_token, stringCounts_target
    stringCounts_token[token] += 1
    stringCounts_target[target] += 1
    n_token += 1
    n_target += 1


def normalize_token(token):
    if args.lc:
        token = token.lower()
    if args.zero:
        token = re.sub(repl, "0", token)
    return token


def out_instance(tokens, targets, outs):
    # also update the sequence lengths
    instance = [
        [[token] for token in tokens],
        targets
    ]
    global seqLengths
    seqLengths.append(len(tokens))
    print(json.dumps(instance), file=outs)

linesWritten = 0  # number of instances
with open(outdata_name, "tw", encoding="utf-8") as outds:
    with open(args.file, encoding="utf-8") as ins:
        for line in ins:
            line = line.rstrip()
            if line.startswith("-DOCSTART-"):
                have_docstring = True
                continue
            if not line:
                if not have_docstring:
                    out_instance(tokens, targets, outds)
                    linesWritten += 1
                    tokens = []
                    targets = []
                have_docstring = False
            else:
                fields = line.split()
                token = fields[0]
                target = fields[args.col]
                token = normalize_token(token)
                tokens.append(token)
                targets.append(target)
                count_elements(token, target)
    # this should not be necessary but just to be safe:
    if tokens:
        # output the instance
        # increment linesWritten
        # gather statistics
        pass

# calculate final stats, set remaining fields, then output meta file
meta_info["featureStats"]["Token┆string╬A"]["stringCounts"] = stringCounts_token
meta_info["featureStats"]["Token┆string╬A"]["n"] = n_token
meta_info["targetStats"]["stringCounts"] = stringCounts_target
meta_info["targetStats"]["n"] = n_target
meta_info["sequLengths.mean"] = sum(seqLengths)/len(seqLengths)
meta_info["sequLengths.min"]  =  min(seqLengths)
meta_info["sequLengths.max"] = max(seqLengths)
meta_info["sequLengths.variance"] = statistics.pvariance(seqLengths)
meta_info["linesWritten"] = linesWritten
meta_info["dataFile"] = os.path.abspath(outdata_name)
meta_info["savedOn"] = datetime.datetime.today().strftime('%Y-%m-%d')
with open(outmeta_name, "tw", encoding="utf-8") as outms:
    json.dump(meta_info, outms)
print("Number of different tokens:", len(stringCounts_token), file=sys.stderr)
print("Number of different targets:", len(stringCounts_target), file=sys.stderr)
print("Number of tokens/targets: ", n_token, "/", n_target, file=sys.stderr)
print("Number of instances/sequences:", linesWritten, file=sys.stderr)
# DONE, output some infos ...
