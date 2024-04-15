# Chinese-NER
Chinese Named Entity Recognition (BiLSTM-crf model enhanced by word segmentation).

## Update
**Warning**: When the training loss drops but the evaluation metrics are 0, please check whether the  learning rate is suitable for the Bi-LSTM model.

## Environment

* python>=3.6.4
* pytorch==1.5.0
* seqeval==0.0.12
* tqdm ==  4.62.3

## Model

* BiLSTM + CRF


## Data Overview

Training Set|Validation set|Test set (sentence samples)
|:-:|:-:|:-:|
28,161|2,531|3,205|

### Data Format

Each line contains a character and its label, separated by "\t" or space. Each sentence is followed by a blank line.

```
中	B-LOC
国	I-LOC
很	O
大	O

句	O
子	O
结	O
束	O
是	O
空	O
行	O
```

We use the ROLING 2022 dataset (https://github.com/NCUEE-NLPLab/ROCLING-2022-ST-CHNER). Download and unzip it in `data/roling/`, 
(1) run 'parse_ROLING.py' to get all the original files (train, dev, test)

(2) run ‘seg_corpus.py’under the '/seg_corpus/' directory to prepare domain lexicons 

(3) run the following command to generate four dictionaries for mapping lexicons in each set of data

```
python process_roling.py --sen_file ./data/embedding/ROLING_embed_char.txt --dict_file ./data/word2id_char.json

python process_roling.py --sen_file ./data/embedding/ROLING_embed_jieba1.txt --dict_file ./data/word2id_jieba1.json

python process_roling.py --sen_file ./data/embedding/ROLING_embed_jieba2.txt --dict_file ./data/word2id_jieba2.json

python process_roling.py --sen_file ./data/embedding/ROLING_embed_jieba3.txt --dict_file ./data/word2id_jieba3.json

```
(4) Modify the `labels` in `main.py` according to your dataset:

```
labels = ['B-BODY', 'O', 'I-BODY', 'B-CHEM', 'I-CHEM', 'B-SYMP', 'I-SYMP', 'B-SUPP', 'I-SUPP', 'B-DISE', 'I-DISE', 'B-TREAT', 'I-TREAT', 'B-TIME', 'I-TIME', 'B-DRUG', 'I-DRUG', 'B-EXAM', 'I-EXAM', 'B-INST', 'I-INST']
```

## Usage
### **Train**
```

# run bilstm+crf
python main.py --model bilstm --crf --mode train 

```
### **Test**

```
python main.py --model bilstm --crf --mode test

```

## BiLSTM + CRF model results on the ROLING2022 CHNER datasets are as followed:


**Jieba Base (epoch 5)**

| precision | recall | f1-score | support |
| :-: | :-: | :-: | :-: |
| B-BODY | 0.819 | 0.659 | 0.731 | 5315 |
| B-CHEM | 0.837 | 0.537 | 0.654 | 1718 |
| B-DISE | 0.885 | 0.589 | 0.707 | 2609 |
| B-DRUG | 0.787 | 0.393 | 0.524 | 481 |
| B-EXAM | 0.623 | 0.464 | 0.532 | 207 |
| B-INST | 0.700 | 0.224 | 0.339 | 250 |
| B-SUPP | 0.703 | 0.530 | 0.604 | 183 |
| B-SYMP | 0.771 | 0.549 | 0.642 | 1944 |
| B-TIME | 0.807 | 0.487 | 0.608 | 197 |
| B-TREAT | 0.797 | 0.585 | 0.675 | 468 |
| I-BODY | 0.822 | 0.657 | 0.730 | 8254 |
| I-CHEM | 0.831 | 0.570 | 0.676 | 3851 |
| I-DISE | 0.932 | 0.627 | 0.750 | 7571 |
| I-DRUG | 0.824 | 0.430 | 0.565 | 1599 |
| I-EXAM | 0.840 | 0.437 | 0.575 | 733 |
| I-INST | 0.676 | 0.192 | 0.300 | 629 |
| I-SUPP | 0.771 | 0.588 | 0.667 | 551 |
| I-SYMP | 0.821 | 0.488 | 0.612 | 2878 |
| I-TIME | 0.756 | 0.409 | 0.531 | 408 |
| I-TREAT | 0.840 | 0.602 | 0.701 | 1251 |
| O | 0.845 | 0.982 | 0.909 | 77019 |

| **accuracy** | 0.844 | 118116 |
| **macro avg** | 0.795 | 0.524 | 0.621 | 118116 |
| **weighted avg** | 0.843 | 0.844 | 0.831 | 118116 |



**Baseline (epoch 3)**

|label| precision| recall | f1-score| support
| :-:     | :-:     | :-:     | :-:     | :-:     |
| B-BODY | 0.776 | 0.716 | 0.745 | 5315 |
| B-CHEM | 0.789 | 0.544 | 0.644 | 1718 |
| B-DISE | 0.851 | 0.676 | 0.754 | 2609 |
| B-DRUG | 0.811 | 0.455 | 0.583 | 481 |
| B-EXAM | 0.595 | 0.440 | 0.506 | 207 |
| B-INST | 0.821 | 0.092 | 0.165 | 250 |
| B-SUPP | 0.756 | 0.661 | 0.706 | 183 |
| B-SYMP | 0.774 | 0.532 | 0.630 | 1944 |
| B-TIME | 0.908 | 0.350 | 0.505 | 197 |
| B-TREAT | 0.808 | 0.476 | 0.599 | 468 |
| I-BODY | 0.826 | 0.721 | 0.770 | 8254 |
| I-CHEM | 0.856 | 0.617 | 0.717 | 3851 |
| I-DISE | 0.917 | 0.728 | 0.812 | 7571 |
| I-DRUG | 0.897 | 0.532 | 0.667 | 1599 |
| I-EXAM | 0.857 | 0.458 | 0.597 | 733 |
| I-INST | 0.883 | 0.084 | 0.154 | 629 |
| I-SUPP | 0.790 | 0.677 | 0.729 | 551 |
| I-SYMP | 0.835 | 0.446 | 0.582 | 2878 |
| I-TIME | 0.901 | 0.267 | 0.412 | 408 |
| I-TREAT | 0.936 | 0.490 | 0.643 | 1251 |
| O | 0.866 | 0.984 | 0.922 | 77019 |

| **accuracy** | 0.860 | 118116 |
| **macro avg** | 0.831 | 0.521 | 0.612 | 118116 |
| **weighted avg** | 0.859 | 0.860 | 0.847 | 118116 |

## References

* **pytorch CRF**: https://github.com/kmkurn/pytorch-crf
* **Chinese-NER**:  https://github.com/xiaofei05/Chinese-NER.git
