#!/bin/bash 

## this is how each of the input data files was sample down to at most 1000 lines
cat | shuf | head -1000 
