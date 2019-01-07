#!/usr/bin/env python
"""
Simple utility program which will read in a metafile and write the embeddings needed
for a vocabulary to a new file.
"""

import argparse
import os
import sys
filepath = os.path.dirname(__file__)
rootpath = os.path.join(filepath, os.pardir)
sys.path.append(rootpath)
from gatelfdata.dataset import Dataset
import gzip

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--embid", default="", type=str, help="Embedding id for which to write the file (use default)")
    parser.add_argument("--minfreq", type=int, default=1, help="Minimum frequency setting for the vocabulary (1)")
    parser.add_argument("metafile", help="Path to metafile (REQUIRED)")
    parser.add_argument("embfile", help="Path to input embeddings file to use")
    parser.add_argument("outfile", help="Path to output file")
    args = parser.parse_args()

    embconfig = "{}::no:{}:{}".format(args.embid, args.minfreq, args.embfile)
    ds = Dataset(args.metafile, config={"embs": embconfig, "remove_embs": False})

    vocab = ds.vocabs.vocabs[args.embid]

    print("Got embeddings:", len(vocab.stoe))

    myopen = open
    if args.outfile.endswith(".gz"):
        myopen = gzip.open
    dims = len(next(iter(vocab.stoe.items()))[1])
    n = len(vocab.stoe)
    if vocab.pad_string in vocab.stoe:
        n = n - 1
    if vocab.oov_string in vocab.stoe:
        n = n - 1
    print("Embeddings shape: {}/{}".format(n, dims))
    with myopen(args.outfile, "wt", encoding="utf8") as outs:
        for word, emb in vocab.stoe.items():
            if word != vocab.pad_string and word != vocab.oov_string:
                outs.write(word)
                for val in emb:
                    outs.write(" ")
                    outs.write(str(val))
                outs.write("\n")






