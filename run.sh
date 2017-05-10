#!/usr/bin/env sh
#DIR=src/
DIR=./
export PYTHONPATH=$PYTHONPATH:"${DIR}libsvm-3.22/python/"
python $1
