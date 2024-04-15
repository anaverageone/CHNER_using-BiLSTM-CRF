import os,json,collections, sys, io, csv, gensim
import openpyxl, pprint, string, jieba
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
os.chdir(r'C:\Users\anaverageone\htlt_env\CLTL\Chinese-NER-master')
os.getcwd()

ROLING_corpus = 'seg_corpus\ROLING\ROLING_word_corpus.txt'
OPEN_corpus = 'seg_corpus\OPEN_corpus.txt'
FUSION_corpus = 'seg_corpus\ROLING\FUSION_corpus.txt'

train_file, dev_file = 'data/roling/train.csv', 'data/roling/dev.csv'

train_char, dev_char, test_char = 'data/roling/train.txt', 'data/roling/dev.txt','data/roling/test.txt'

train_jieba1, dev_jieba1, test_jieba1 = 'data/roling/train_jieba1.txt', 'data/roling/dev_jieba1.txt', 'data/roling/test_jieba1.txt'
train_jieba2, dev_jieba2, test_jieba2 = 'data/roling/train_jieba2.txt', 'data/roling/dev_jieba2.txt', 'data/roling/test_jieba2.txt'
train_jieba3, dev_jieba3, test_jieba3 = 'data/roling/train_jieba3.txt', 'data/roling/dev_jieba3.txt', 'data/roling/test_jieba3.txt'

ROLING_embed_char = 'embedding/ROLING_embed_char.txt'
ROLING_embed_jieba1,ROLING_embed_jieba2,ROLING_embed_jieba3 = 'embedding/ROLING_embed_jieba1.txt','embedding/ROLING_embed_jieba2.txt','embedding/ROLING_embed_jieba3.txt'

# base jieba
def w2t(infile, column1, column2):
    df = pd.read_csv(infile)
    list1 = [json.loads(l) for l in df[column1].tolist()]
    list2 = [json.loads(l) for l in df[column2].tolist()]
    return list1, list2

def jieba_wc(char_list):
    wc_list = []
    for sublist in char_list:
        seg_list = (' '.join(jieba.cut(''.join(sublist), HMM=True))).split() 
        word_index = {}
        start_index = 0
        for word in seg_list:
            word_index[start_index] = word
            start_index += len(word)
        wc_list.append(word_index)
    return wc_list
    
def jieba_wc_txt(infile, outfile,column1, column2):
    list1, list2 = w2t(infile, column1, column2)
    wc_list = jieba_wc(list1)
    with open(outfile, 'w', encoding='utf-8') as out_file:
        for i, sub_dict in enumerate(wc_list):
            for k in sub_dict.keys():
                start_tag = list2[i][k]
                word = sub_dict[k]
                out_file.write(f"{word}\t{start_tag}\n")
            out_file.write(f"\n")
# TEST FILE

def read_test(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        sents = [w.replace('\n', '') for w in text.split('\n \n')]
    return sents
            
def jieba_test(infile, outfile):
    sents = read_test(infile)
    new_sents = []
    with tqdm(total=len(sents), desc="Segmenting Sentences") as pbar:
        for sent in sents:
            seg_list = (' '.join(jieba.cut(sent, HMM=True))).split()
            new_sents.append(seg_list)
            pbar.update(1)
    with open(outfile, 'w', encoding='utf-8') as out_file:
        for sublist in new_sents:
            for subword in sublist:
                out_file.write(f"{subword}\t{'O'}\n")
            out_file.write(f"\n")
            
def ROLNIG_embed(file_1, file_2, Column1, txt_file):
    df_1 = pd.read_csv(file_1, encoding='utf-8')
    df_2 = pd.read_csv(file_2, encoding='utf-8')
    df_merged = df_1.append(df_2, ignore_index=True)
    # Iterate over rows and write to the text file
    with open(txt_file, 'w', encoding='utf-8') as file:
        for index, row in df_merged.iterrows():
            list_a = eval(row[Column1])
            s = ' '.join(str(x) for x in list_a)
            file.write(f"{s}\n") 
            
def jieba_embed(infile1, infile2, outfile):
    with open(infile1, 'r', encoding='utf-8') as in_file1, open(infile2, 'r', encoding='utf-8') as in_file2:
        sentlist = [] 
        text1, text2 = in_file1.read().split('\n\n'), in_file2.read().split('\n\n')
        text = text1 + text2
        
        with open(outfile, 'w', encoding='utf-8') as out_file:  # Open the output file once outside the loop
            for sent in text:
                sublist = sent.split('\n')
                new_list = []
                for item in sublist:
                    word = item.split(' ')[0]
                    new_list.append(word)
                s = ' '.join(str(x) for x in new_list)
                out_file.write(f"{s}\n")
                
                
# Basic (default dict) jieba
jieba_test(test_file, test_jieba1)

# ROLING enhanced jieba
jieba.load_userdict(ROLING_corpus)
jieba_test(test_file, test_jieba2)

# FUSION_corpus enhanced jieba
jieba.load_userdict(FUSION_corpus)
jieba_test(test_file, test_jieba3)

# create new train-dev-test sets of data (test set gola labels are all assigned as initial value 'O')
jieba_wc_txt(dev_file, dev_jieba1,'character','character_label')
jieba_wc_txt(train_file,train_jieba1,'character','character_label')
jieba_wc_txt(dev_file,dev_jieba2,'character','character_label')
jieba_wc_txt(train_file,train_jieba2,'character','character_label')
jieba_wc_txt(dev_file,dev_jieba3,'character','character_label')
jieba_wc_txt(train_file,train_jieba3,'character','character_label')

# prepare the lexicons for generating word embedings (dictionaries) 
ROLNIG_embed(train_file,dev_file,'character',ROLING_embed_char)
jieba_embed(train_jieba1,dev_jieba1, ROLING_embed_jieba1)
jieba_embed(train_jieba2,dev_jieba2, ROLING_embed_jieba2)
jieba_embed(train_jieba2,dev_jieba2, ROLING_embed_jieba3)