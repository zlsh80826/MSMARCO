#!/bin/bash
python3.6 elmo.py
allennlp elmo --all --batch-size 10000 --cuda-device 0 vocabs.txt elmo_embedding.bin
