import os
import gensim
import numpy as np
import psutil
import sys
import logging
import gzip
import re
# from fastnumbers import fast_float
from memory_profiler import profile
from collections import defaultdict
from memory_profiler import memory_usage
parent_dir = os.path.dirname(__file__)
parent_dir_mod = parent_dir[:-11] # subtract the source file name
sys.path.append(parent_dir_mod)
from gatelfdata import Dataset

TESTDIR = os.path.join(os.path.dirname(__file__), '.')
DATADIR = os.path.join(TESTDIR, 'data')
print("DEBUG: datadir is ", TESTDIR, file=sys.stderr)

TESTFILE1 = os.path.join(DATADIR, "class-ionosphere.meta.json")
TESTFILE2 = os.path.join(DATADIR, "class-ngram-sp1.meta.json")
TESTFILE3 = os.path.join(DATADIR, "class-window-pos1.meta.json")
TESTFILE4 = os.path.join(DATADIR, "seq-pos1.meta.json")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler = logging.StreamHandler()
logger.addHandler(streamhandler)
# Code to figure out best way to load pre-calculated embeddings fast an efficiently
# We need to load embeddings in a way where only those embeddings are loaded which
# occur also in our own vocabulary.

process = psutil.Process(os.getpid())
mem_rss0, mem_vms0 = process.memory_info()[0:2]
memusage0 = memory_usage(-1, interval=1, timeout=1)[0]
print("Before loading dataset: Memory RSS/VMS=%s/%s // %s" % (mem_rss0, mem_vms0, memusage0))
print("System memory", psutil.virtual_memory())

ds = Dataset(TESTFILE3)
mem_rss1, mem_vms1 = process.memory_info()[0:2]
memusage1 = memory_usage(-1, interval=1, timeout=1)[0]
print("After loading dataset: Memory RSS/VMS // Mem=%s/%s // %s" % (mem_rss1, mem_vms1, memusage1))
print("After loading dataset: diffs  RSS/VMS // Mem=%s/%s // %s" % (mem_rss1-mem_rss0, mem_vms1-mem_vms0, memusage1-memusage0))
print("Dataset vocabs: ", ds.vocabs)
vtoken = ds.vocabs.get_vocab("token")
print("Token vocab: ", vtoken)

# test our simple approach to loading embeddings. We expect a link or copy of glove.6B.50d.txt.gz in the tests/data
# directory for this
# emb_file = "tests/data/glove.6B.50d.txt.gz"
# emb_file = "tests/data/wiki.en.vec"
emb_file = sys.argv[1]

stoi = vtoken.stoi
stoe = defaultdict()
n_vocab = len(stoi)
n_added = 0
n_lines = 0
if emb_file.endswith(".txt") or emb_file.endswith(".vec") or emb_file.endswith(".txt.gz"):
    if emb_file.endswith(".txt.gz"):
        reader = gzip.open
    else:
        reader = open
    logger.info("Loading embeddings for %s from %s" % ("token", emb_file))
    n_expected = 0
    with reader(emb_file, 'rt', encoding="utf-8") as infile:
        for line in infile:
            if n_added == n_vocab:
                print("INFO: already have all embeddings needed, stopping reading")
            if n_lines == 0:
                m = re.match(r'^\s*([0-9]+)\s+([0-9]+)\s*$', line)
                if m:
                    n_expected = int(m.group(1))
                    dims = int(m.group(2))
                    continue
            n_lines += 1
            if n_lines % 10000 == 0:
                logger.debug("Read lines from embeddings file: %s of %s, added=%s of %s" % (n_lines, n_expected, n_added, n_vocab))
            # line = line.strip()
            # fields = re.split(r' +', line)
            #fields = line.split()
            #word = fields[0]
            toidx = line.find(" ")
            word = line[0:toidx]
            if word in stoi:
                # embs = np.array([fast_float(e, raise_on_invalid=True) for e in fields[1:]])
                # embs = map(fast_float, fields[1:])
                # embs = [fast_float(e, raise_on_invalid=True) for e in fields[1:]]
                embs = []
                for i in range(dims):
                    fromidx = toidx+1
                    toidx = line.find(" ", fromidx)
                    if toidx < 0:
                        toidx = len(line)
                #    # print("Parsing: from=",fromidx,"to=",toidx,"str=", line[fromidx:toidx])
                    # embs.append(fast_float(line[fromidx:toidx]))
                    embs.append(float(line[fromidx:toidx]))
                #embs = [fast_float(e, raise_on_invalid=True) for e in line.split()[1:]]
                #if embs != embs2:
                #    print("ERROR, not equals, embs=",embs,"embs2=",embs2)
                #    sys.exit(1)
                n_added += 1
                stoe[word] = embs
        # update the emb_dims setting from the last embedding we read, if any
else:
    raise Exception("Embeddings file must have one of the extensions: .txt, .txt.gz, .vocab, .npy")
logger.info("Embeddings loaded, total=%s, added=%s" % (n_lines, n_added))
mem_rss2, mem_vms2 = process.memory_info()[0:2]
memusage2 = memory_usage(-1, interval=1, timeout=1)[0]
print("After loading embs: Memory RSS/VMS // Mem=%s/%s // %s" % (mem_rss2, mem_vms2, memusage2))
print("After loading embs: diffs  RSS/VMS // Mem=%s/%s // %s" % (mem_rss2-mem_rss1, mem_vms2-mem_vms1, memusage2-memusage1))
# print("Embedding for ','=", stoe[","])