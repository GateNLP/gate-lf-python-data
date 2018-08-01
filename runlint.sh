#!/bin/bash

pylint -d c0301 gatelfdata | tee pylint.out.txt
