#!/bin/bash 

## this is how each of the input data files was sample down to at most 1000 lines
tmpfile=/tmp/sample.sh.$$
echo "12345" > $tmpfile
cat | shuf | head -1000 
