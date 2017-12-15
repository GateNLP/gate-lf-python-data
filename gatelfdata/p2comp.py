# https://stackoverflow.com/questions/10971033/backporting-python-3-openencoding-utf-8-to-python-2
from __future__ import print_function
import sys
if sys.version_info[0] > 2:
    # py3k
    print("DEBUG: running p2comp, have python 3", file=sys.stderr)
    import io
    def open(file, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, closefd=True, opener=None):
        return io.open(file, mode=mode, encoding=encoding,
                    errors=errors, buffering=buffering, newline=newline, closefd=closefd, opener=opener)
else:
    # py2
    print("DEBUG: running p2comp, have python 2", file=sys.stderr)
    import codecs
    import warnings
    def open(file, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, closefd=True, opener=None):
        if newline is not None:
            warnings.warn('newline is not supported in py2')
        if not closefd:
            warnings.warn('closefd is not supported in py2')
        if opener is not None:
            warnings.warn('opener is not supported in py2')
        return codecs.open(filename=file, mode=mode, encoding=encoding,
                    errors=errors, buffering=buffering)
