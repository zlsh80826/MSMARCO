#!/bin/bash

set -xe
export PYTHONPATH=../data

cd ../script
echo 'Start to test ...'
python3.6 train.py --model ../model/pm.model --test ../data/test.tsv
cp ../model/pm.model_out.json ../output.json

# echo 'Start to evaluate ...'
cd ../Evaluation
sh run.sh sample_test_data/dev_as_references.json ../output.json
