import os,json,collections, sys, io, csv, gensim
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

os.chdir(r'C:\Users\anaverageone\htlt_env\CLTL\Chinese-NER-master')
os.getcwd()

# define all local directories of the files paths
ROLING_corpus = 'seg_corpus\ROLING\ROLING_word_corpus.txt'
OPEN_corpus = 'seg_corpus\OPEN_corpus.txt'
FUSION_corpus = 'seg_corpus\ROLING\FUSION_corpus.txt'

train_file, dev_file, test_file = 'data/roling/train.csv', 'data/roling/dev.csv','data/roling/ROCLING22_CHNER_truth.txt'
train, dev, test = 'data/roling/train.txt', 'data/roling/dev.txt','data/roling/test.txt'

train_jieba1, dev_jieba1, test_jieba1 = 'data/roling/train_jieba1.txt', 'data/roling/dev_jieba1.txt', 'data/roling/test_jieba1.txt'
train_jieba2, dev_jieba2, test_jieba2 = 'data/roling/train_jieba2.txt', 'data/roling/dev_jieba2.txt', 'data/roling/test_jieba2.txt'
train_jieba3, dev_jieba3, test_jieba3 = 'data/roling/train_jieba3.txt', 'data/roling/dev_jieba3.txt', 'data/roling/test_jieba3.txt'

ROLING_embed_char = 'embedding/ROLING_embed_char.txt'
ROLING_embed_jieba1,ROLING_embed_jieba2,ROLING_embed_jieba3 = 'embedding/ROLING_embed_jieba1.txt','embedding/ROLING_embed_jieba2.txt','embedding/ROLING_embed_jieba3.txt'

dev_char_e3_output,char_e3_dev_outcome = 'output_char_e3/dev_output/bilstm_crf.txt','output_char_e3/dev_output/char_dev_outcome.txt'
dev_char_e5_output,char_e5_dev_outcome = 'output_char_e5/dev_output/bilstm_crf.txt','output_char_e5/dev_output/char_dev_outcome.txt'
dev_char_e7_output,char_e7_dev_outcome = 'output_char_e7/dev_output/bilstm_crf.txt','output_char_e7/dev_output/char_dev_outcome.txt'

dev_jieba1_e3_output,jieba1_e3_dev_outcome = 'output_jieba1_e3/dev_output/bilstm_crf.txt','output_jieba1_e3/dev_output/jieba1_dev_e3_outcome.txt'
dev_jieba1_e5_output,jieba1_e5_dev_outcome = 'output_jieba1_e5/dev_output/bilstm_crf.txt','output_jieba1_e5/dev_output/jieba1_dev_e5_outcome.txt'
dev_jieba1_e7_output,jieba1_e7_dev_outcome = 'output_jieba1_e7/dev_output/bilstm_crf.txt','output_jieba1_e7/dev_output/jieba1_dev_e7_outcome.txt'

dev_jieba2_e3_output,jieba2_e3_dev_outcome = 'output_jieba2_e3/dev_output/bilstm_crf.txt','output_jieba2_e3/dev_output/jieba2_dev_e3_outcome.txt'
dev_jieba2_e5_output,jieba2_e5_dev_outcome = 'output_jieba2_e5/dev_output/bilstm_crf.txt','output_jieba2_e5/dev_output/jieba2_dev_e5_outcome.txt'
dev_jieba2_e7_output,jieba2_e7_dev_outcome = 'output_jieba2_e7/dev_output/bilstm_crf.txt','output_jieba2_e7/dev_output/jieba2_dev_e7_outcome.txt'

dev_jieba3_e3_output,jieba3_e3_dev_outcome = 'output_jieba3_e3/dev_output/bilstm_crf.txt','output_jieba3_e3/dev_output/jieba3_dev_e3_outcome.txt'
dev_jieba3_e5_output,jieba3_e5_dev_outcome = 'output_jieba3_e5/dev_output/bilstm_crf.txt','output_jieba3_e5/dev_output/jieba3_dev_e5_outcome.txt'
dev_jieba3_e7_output,jieba3_e7_dev_outcome = 'output_jieba3_e7/dev_output/bilstm_crf.txt','output_jieba3_e7/dev_output/jieba3_dev_e7_outcome.txt'


test_char_output, char_test_outcome = 'output_char/bilstm_crf.txt','output_char/char_test_outcome.txt'
test_jieba3_output, jieba3_test_outcome = 'output_jieba3/bilstm_crf.txt','output_jieba3/jieba3_test_outcome.txt'

# define the functions for evaluation
def get_char_files(file1, Column1,Column2, file2):
    with open(file2, "w", encoding="utf-8") as f2:
        df1 = pd.read_csv(file1, encoding='utf-8')
        for index, row in df1.iterrows():
            lista = eval(row[Column1])
            listb = eval(row[Column2])
            for word, tag in zip(lista, listb):
                f2.write(f"{word}\t{tag}\n")
            f2.write("\n")

def get_test_file(file1, file2):
    modified_sentences = []
    with open(file1, "r", encoding="utf-8") as f1:
        sents = f1.read().strip().split("\n \n")
   
    with open(file2, "w", encoding="utf-8") as f2:
        for sent in sents:
            word_labels = sent.strip().split('\n')
            for wl in word_labels:
                if wl.strip():
                    w, l = wl.split(' ')
                    f2.write(f'{w}\t{l}\n')
            f2.write('\n')
def create_conf_matrix(true_list,new_pred_list, outfile):
    label_list=['B-EXAM', 'B-BODY', 'B-DISE', 'I-DISE', 'B-SYMP', 'B-TREAT', 'B-CHEM', 'I-CHEM', 'I-SYMP', 'B-TIME', 'B-SUPP', 'I-BODY', 'I-TREAT', 'B-INST', 'B-DRUG', 'I-DRUG','I-TIME', 'I-INST', 'I-SUPP', 'I-EXAM','O']
    data = confusion_matrix(true_list, new_pred_list,labels=label_list)
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + label_list)
        for i in range(len(data)):
            writer.writerow([label_list[i]] + data[i].tolist())
            
            
            
            
# calling the funcitons to get different files         
get_char_files(dev_file,'character','character_label',dev_char)
get_char_files(train_file,'character','character_label',train_char)
get_test_file(test_file, test)

# get all results of different epoches for Jieba1 dev dataset
convert_word_output(dev_jieba1,dev_jieba1_e3_output,dev,jieba1_dev_e3_outcome)
get_results(jieba1_dev_outcome,'output_jieba1_e3/dev_output/jieba1_e3_dev_classification.txt','output_jieba1_e3/dev_output/jieba1_e3_dev_confusion.csv')
convert_word_output(dev_jieba1,dev_jieba1_e5_output,dev,jieba1_dev_e5_outcome)
get_results(jieba1_dev_outcome,'output_jieba1_e5/dev_output/jieba1_e5_dev_classification.txt','output_jieba1_e5/dev_output/jieba1_e5_dev_confusion.csv')
convert_word_output(dev_jieba1,dev_jieba1_e7_output,dev,jieba1_dev_e7_outcome)
get_results(jieba1_dev_outcome,'output_jieba1_e7/dev_output/jieba1_e7_dev_classification.txt','output_jieba1_e7/dev_output/jieba1_e7_dev_confusion.csv')

# get all results of different epoches for Jieba2 dev dataset
convert_word_output(dev_jieba2,dev_jieba2_e3_output,dev,jieba2_dev_e3_outcome)
get_results(jieba2_dev_outcome,'output_jieba2_e3/dev_output/jieba2_e3_dev_classification.txt','output_jieba2_e3/dev_output/jieba2_e3_dev_confusion.csv')
convert_word_output(dev_jieba2,dev_jieba2_e5_output,dev,jieba2_dev_e5_outcome)
get_results(jieba2_dev_outcome,'output_jieba2_e5/dev_output/jieba2_e5_dev_classification.txt','output_jieba2_e5/dev_output/jieba2_e5_dev_confusion.csv')
convert_word_output(dev_jieba2,dev_jieba2_e7_output,dev,jieba2_dev_e7_outcome)
get_results(jieba2_dev_outcome,'output_jieba2_e7/dev_output/jieba2_e7_dev_classification.txt','output_jieba2_e7/dev_output/jieba2_e7_dev_confusion.csv')

# get all results of different epoches for Jieba3 dev dataset
convert_word_output(dev_jieba3,dev_jieba3_e3_output,dev,jieba3_dev_e3_outcome)
get_results(jieba3_dev_outcome,'output_jieba3_e3/dev_output/jieba3_e3_dev_classification.txt','output_jieba3_e3/dev_output/jieba3_e3_dev_confusion.csv')
convert_word_output(dev_jieba3,dev_jieba3_e5_output,dev,jieba3_dev_e5_outcome)
get_results(jieba3_dev_outcome,'output_jieba3_e5/dev_output/jieba3_e5_dev_classification.txt','output_jieba3_e5/dev_output/jieba3_e5_dev_confusion.csv')
convert_word_output(dev_jieba3,dev_jieba3_e7_output,dev,jieba3_dev_e7_outcome)
get_results(jieba3_dev_outcome,'output_jieba3_e7/dev_output/jieba3_e7_dev_classification.txt','output_jieba3_e7/dev_output/jieba3_e7_dev_confusion.csv')

# get the test reuslts of jieba1_e5 model
convert_word_output(test_jieba3,test_jieba3_output,test,jieba3_outcome)
get_results(jieba3_outcome,'output_jieba3/jieba3_classification.txt','output_jieba3/jieba3_confusion.csv')

# get the test reuslts of char_e3 model
convert_char_output(test,test_char_output,char_test_outcome)
get_results(char_test_outcome,'output_char/char_classification.txt','output_char/char_confusion.csv')