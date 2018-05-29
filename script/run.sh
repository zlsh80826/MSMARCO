#!/bin/bash

export PYTHONPATH=../data
python3.6 train.py -datadir ../data -outputdir ../model -logdir log/
