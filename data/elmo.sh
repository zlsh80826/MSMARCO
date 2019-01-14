#!/bin/bash
python3.6 elmo.py
allennlp elmo --all --batch-size 12000 --cuda-device 0 --use-sentence-keys vocabs.txt elmo_embedding.bin 
