# MSMARCO with S-NET extraction
* A CNTK(Microsoft deep learning toolkit) implementation of [S-NET: FROM ANSWER EXTRACTION TO ANSWER
GENERATION FOR MACHINE READING COMPREHENSION](https://arxiv.org/pdf/1706.04815.pdf) with some modifications. 
* This project is designed for the [MSMARCO](http://www.msmarco.org/) dataset
* Code structure is based on [CNTK BIDAF Example](https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow/msmarco)
* Support MSMARCO V1 and V2!

## Requirements

Here are some required libraries for training.

### General
* python3.6
* cuda-9.0 (CNTK required)
* openmpi-1.10 (CNTK required)
* gcc >= 6 (CNTK required)

### Python
* Please refer requirements.txt

## Usage 

### Preprocess

#### MSMARCO V1
Download MSMARCO v1 dataset, GloVe embedding.

```Bash
cd data
python3.6 download.py v1
```

Convert raw data to tsv format

```Bash
python3.6 convert_msmarco.py --threads=`nproc` 
```

Convert tsv format to ctf(CNTK input) format and build vocabs dictionary

```Bash
python3.6 tsv2ctf.py
```

Generate elmo embedding

```Bash
sh elmo.sh
```

#### MSMARCO V2

Download MSMARCO v2 dataset, GloVe embedding.

```Bash
cd data
python3.6 download.py v2
```

Convert raw data to tsv format

```Bash
python3.6 convert_msmarco.py --threads=`nproc` --ratio=0.1
```

Convert tsv format to ctf(CNTK input) format and build vocabs dictionary

```Bash
python3.6 tsv2ctf.py
```

Generate elmo embedding

```Bash
sh elmo.sh
```

### Train (Same for V1 and V2)

```Bash
cd ../script
mkdir log
sh run.sh
```

### Evaluate develop dataset

#### MSMARCO V1

```Bash
cd Evaluation
sh eval.sh v1
```

#### MSMARCO v2

```Bash
cd Evaluation
sh eval.sh v2
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
|MSMARCO v1 w/  elmo|39.42 | 39.47|
|MSMARCO v2 w/  elmo|42.10 | 42.46|

## TODO
- [X] Multi-threads preprocessing 
- [X] Elmo-Embedding
- [X] Evaluation script
- [X] MSMARCO v2 support
- [ ] Reasonable metrics
