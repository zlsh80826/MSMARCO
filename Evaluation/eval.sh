#!/bin/bash

if [ $# -ne 1 ]; then
	echo 'Usage: sh eval.sh <version>'
else
    if [ $1 == 'v1' ]; then
        set -xe
        export PYTHONPATH=../data
        cd ../script
        echo 'Start to test ...'
        \time --verbose python3.6 train.py --model ../model/pm.model --test ../data/test.tsv
        cp ../model/pm.model_out.json ../output.json

        echo 'Start to evaluate ...'
        cd ../Evaluation
        sh run.sh answer/dev_as_references.json ../output.json
    elif [ $1 == 'v2' ]; then 
        set -xe
        export PYTHONPATH=../data
        cd ../script
        echo 'Start to test ...'
        \time --verbose python3.6 train.py --model ../model/pm.model --test ../data/test.tsv
        cp ../model/pm.model_out.json ../output.json

        echo 'Start to evaluate ...'
        cd ../Evaluation
        sh run.sh answer/v2_dev_noans.json ../output.json
    else
        echo '<version> can be only "v1" or "v2"'
    fi
fi

