#!/bin/bash

export PYTHONPATH=`pwd`
rm -rf ./doc-sphinx
rm -rf ./doc-html
mkdir doc-sphinx
mkdir doc-html
# cp sphinx-config/index.rst doc-sphinx
sphinx-apidoc -e -f -H "GATE LF Python Data (gatelfdata)" -A "Johann Petrak" -V "0.1" --ext-autodoc --ext-githubpages -o doc-sphinx gatelfdata
mv doc-sphinx/modules.rst doc-sphinx/index.rst
sphinx-build -b html -c sphinx-config doc-sphinx doc-html 
cp -r doc-html/* docs/pythondoc
