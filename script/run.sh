#!/bin/bash

export PYTHONPATH=../data
\time --verbose mpirun -n 4 python3.6 train.py -datadir ../data -outputdir ../model -logdir log/
