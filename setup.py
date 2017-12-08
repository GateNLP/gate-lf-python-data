import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return open(os.path.join(here, fname)).read()
readme = read('README.md')

setup(
    name="gitlfdata",
    version="0.1",
    description=("Library to handle the dense json data generated by the GATE Learning Framework"),
    author="Johann Petrak",
    #author_email = "notyet@somewhere.com",
    #url = "http://packages.python.org/an_example_pypi_project",
    license="Apache 2.0",
    #keywords = "example documentation tutorial",
    packages=['gatelfdata'],
    long_description=readme,
    py_modules=['gatelfdata'],
    # scripts=['some.py'],
    # entry_points = {'console_scripts': ['some=some:main']},
    tests_require = ['nose'],
    test_suite = 'nose.collector',
    classifiers=[],
)
