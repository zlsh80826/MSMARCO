#!/bin/bash
python3.6 elmo.py
allennlp elmo --all --batch-size 32768 vocabs.txt elmo_embedding.bin
