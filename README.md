# MSMARCO with S-NET extraction
* A CNTK(Microsoft deep learning toolkit) implementation of [S-NET: FROM ANSWER EXTRACTION TO ANSWER
GENERATION FOR MACHINE READING COMPREHENSION](https://arxiv.org/pdf/1706.04815.pdf) with some modifications. 
* This project is designed for the [MSMARCO](http://www.msmarco.org/) dataset
* Code structure is based on [CNTK BIDAF Example](https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow/msmarco)

## Requirements

There are some required library for training, if you have any problem, please contact me. 
Here are the suggested version.

### General
* cuda-9.0
* openmpi-1.10
* python3.6

### Python
* Please refer requirements.txt

## Usage

### Preprocess
Download MSMARCO dataset, GloVe embedding.
```
cd data
python3.6 download.py v1
```

Convert raw data to tsv format
```
python3.6 convert_msmarco.py
```

Convert tsv format to ctf(CNTK input) format and build vocabs dictionary
```
python3.6 tsv2ctf.py
```

### Train
``` 
cd ../script
mkdir log
sh run.sh
```

### Performance

#### Paper
||rouge-l|bleu_1|
|---|---|---|
|S-Net (Extraction)|41.45|44.08|
|S-Net (Extraction, Ensemble)|42.92|44.97|

#### This implementation
||rouge-l|bleu_1|
|---|---|---|
|MSMARCO v1 w/o elmo|38.43 | 39.14|
|MSMARCO v1 w/  elmo|working|working|
|MSMARCO v2 w/o elmo|working|working|
|MSMARCO v2 w   elmo|working|working|

## TODO
- [ ] Multi-threads preprocessing 
- [ ] Elmo-Embedding
- [X] Evaluation script
- [ ] MSMARCO v2 support
- [ ] Reasonable metrics
