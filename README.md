# MSMARCO with S-NET Extraction (Extraction-net)
* A CNTK(Microsoft deep learning toolkit) implementation of [S-NET: FROM ANSR EXTRACTION TO ANSWER
GENERATION FOR MACHINE READING COMPREHENSION](https://arxiv.org/pdf/1706.04815.pdf) extraction part with some modifications. 
* This project is designed for the [MSMARCO](http://www.msmarco.org/) dataset
* Code structure is based on [CNTK BIDAF Example](https://github.com/Microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow/msmarco)
* Support MSMARCO V1 and V2!

## Requirements

Here are some required libraries for training and evaluations.

### General
* python3.6
* cuda-9.0 (CNTK required)
* openmpi-1.10 (CNTK required)
* gcc >= 6 (CNTK required)

### Python
* Please refer requirements.txt

## Evaluate with pretrained model

This repo provides pretrained model and pre-processed validation dataset for testing the performance

Please download [pretrained model](https://drive.google.com/open?id=1P9mfJtaFxSSOhshZNmqsjXKS9oN5KEVy) and 
[pre-processed data](https://drive.google.com/file/d/1aNpxea4r42VrJzPpAg2GMkasTuvT3xkU/view?usp=sharing) and put them on
the ``MSMARCO/data`` and ``MSMARCO`` root directory respectively, then decompress them at the right places. 

The code structure should be like

```Bash
MSMARCO
├── data
│   ├── elmo_embedding.bin
│   ├── test.tsv
│   ├── vocabs.pkl
│   ├── data.tar.gz
│   └── ... others
├── model
│   ├── pm.model
│   ├── pm.model.ckp
│   └── pm.model_out.json
└── ... others
```

After decompressing, 

```Bash
cd Evaluation
sh eval.sh
```

then you should get the generated answer and rough-l score.

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
python3.6 convert_msmarco.py v1 --threads=`nproc` 
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
python3.6 convert_msmarco.py v2 --threads=`nproc`
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
|MSMARCO v2 w/  elmo|43.66 | 44.44|

## TODO
- [X] Multi-threads preprocessing 
- [X] Elmo-Embedding
- [X] Evaluation script
- [X] MSMARCO v2 support
- [ ] Reasonable metrics
