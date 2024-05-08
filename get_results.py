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


test_char_output, char_test_outcome = 'output_char_e3/bilstm_crf.txt','output_char_e3/char_test_outcome.txt'
test_jieba3_output, jieba3_test_outcome = 'output_jieba3_e5/bilstm_crf.txt','output_jieba3_e5/jieba3_outcome.txt'

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

def get_gold_predict(file1):
    gold_labels = []
    prediction_labels = []
    with open(file1, "r", encoding="utf-8") as f1:
        lines = f1.readlines()
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 3:  
            gold_labels.append(parts[1])  
            prediction_labels.append(parts[2])
    return gold_labels, prediction_labels
            
# def plot_confusion_matrix(confusion_matrix, class_names, errors_only=False, figsize=(21, 21), fontsize=9):
      
#     # Instantiate Figure
#     fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#     # plt.subplots_adjust(wspace=0.5)
    
#     # Confusion Matrix - Class Counts
#     df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)    
#     heatmap = sns.heatmap(df_cm, ax=ax, cmap='Blues', fmt='d', annot=True, annot_kws={"size": fontsize+4},
#                           linewidths=2, linecolor='black', cbar=False)   

#     ax.tick_params(axis='x', labelrotation=0, labelsize=fontsize, labelcolor='black')
#     ax.tick_params(axis='y', labelrotation=0, labelsize=fontsize, labelcolor='black')
#     ax.set_xlabel('PREDICTED CLASS', fontsize=fontsize, color='black')
#     ax.set_ylabel('TRUE CLASS', fontsize=fontsize, color='black')
#     ax.set_title('Confusion Matrix - Class Counts', pad=21, fontsize=fontsize, color='black')    

#     for text in ax.texts:
#         if text.get_text() == '0':
#             text.set_color(color='white')

# A Function that plots the heatmaps (un-normalized) for the predicted results compared with gold labels (https://github.com/jkmackie/confusion_matrix_visualization.git)
def plot_confusion_matrix(confusion_matrix, class_names, errors_only=False, figsize=(15, 13), fontsize=9):
      
    # Instantiate Figure
    fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    # Confusion Matrix - Class Counts
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)    
    heatmap = sns.heatmap(df_cm, ax=ax, cmap='Blues', fmt='d', annot=True, annot_kws={"size": fontsize+4},
                          linewidths=2, linecolor='black', cbar=False)   

    ax.tick_params(axis='x', labelrotation=45, labelsize=fontsize, labelcolor='black')
    ax.tick_params(axis='y', labelrotation=45, labelsize=fontsize, labelcolor='black')
    ax.set_xlabel('PREDICTED CLASS', fontsize=fontsize, color='black')
    ax.set_ylabel('TRUE CLASS', fontsize=fontsize, color='black')
    ax.set_title('Confusion Matrix - Class Counts', pad=21, fontsize=fontsize, color='black')    

    for text in ax.texts:
        if text.get_text() == '0':
            text.set_color(color='white')
            
# A Function that plots the heatmaps (normalized) for the predicted results compared with gold labels (https://github.com/jkmackie/confusion_matrix_visualization.git)
def plot_confusion_matrix_norm(confusion_matrix, class_names, errors_only=False, figsize = (15,13), fontsize=9):     
    #Instantiate Figure
    fig, (ax1) = plt.subplots(nrows=1, figsize=figsize)
    plt.subplots_adjust(wspace = 1)
    
    #Show errors only by filling diagonal with zeroes.
    # if errors_only:
    #     np.fill_diagonal(confusion_matrix, 0)        
        
    # ax1 - Normalized Confusion Matrix    
    #Normalize by dividing (M X M) matrix by (M X 1) matrix.  (M X 1) is row totals.
    conf_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:,np.newaxis]
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  #fix any nans caused by zero row total
    df_cm_norm = pd.DataFrame(conf_matrix_norm, index=class_names, columns=class_names)
    heatmap = sns.heatmap(df_cm_norm, ax=ax1, cmap='Blues', fmt='.3f', annot=True, annot_kws={"size": fontsize},
              linewidths=2, linecolor='black', cbar=False)
    
    ax1.tick_params(axis='x', labelrotation=0, labelsize=fontsize, labelcolor='black')
    ax1.tick_params(axis='y', labelrotation=0, labelsize=fontsize, labelcolor='black')
    ax1.set_ylim(ax1.get_xlim()[0], ax1.get_xlim()[1])  #Fix messed up ylim
    ax1.set_xlabel('PREDICTED CLASS', fontsize=fontsize, color='black')
    ax1.set_ylabel('TRUE CLASS', fontsize=fontsize, color='black')
    ax1.set_title('Confusion Matrix - Normalized', pad=21, fontsize=fontsize, color='black')
      
  
    for text in ax1.texts:
        if text.get_text() == '0.000':
            text.set_color(color='white')            
            
# calling the funcitons to get different files         
get_char_files(dev_file,'character','character_label',dev_char)
get_char_files(train_file,'character','character_label',train_char)
get_test_file(test_file, test)

# get all results of (epoches=3,5,7) of Baseline system on dev dataset
convert_char_output(dev,dev_char_e3_output,char_e3_dev_outcome)
get_results(char_e3_dev_outcome,'output_char_e3/dev_output/char_dev_classification.txt','output_char_e3/dev_output/char_dev_confusion.csv')
convert_char_output(dev,dev_char_e5_output,char_e5_dev_outcome)
get_results(char_e5_dev_outcome,'output_char_e5/dev_output/char_dev_classification.txt','output_char_e5/dev_output/char_dev_confusion.csv')
convert_char_output(dev,dev_char_e7_output,char_e7_dev_outcome)
get_results(char_e7_dev_outcome,'output_char_e7/dev_output/char_dev_classification.txt','output_char_e7/dev_output/char_dev_confusion.csv')

# get all results of (epoches=3,5,7) of Jieba1 system on dev dataset
convert_word_output(dev_jieba1,dev_jieba1_e3_output,dev,jieba1_dev_e3_outcome)
get_results(jieba1_dev_outcome,'output_jieba1_e3/dev_output/jieba1_e3_dev_classification.txt','output_jieba1_e3/dev_output/jieba1_e3_dev_confusion.csv')
convert_word_output(dev_jieba1,dev_jieba1_e5_output,dev,jieba1_dev_e5_outcome)
get_results(jieba1_dev_outcome,'output_jieba1_e5/dev_output/jieba1_e5_dev_classification.txt','output_jieba1_e5/dev_output/jieba1_e5_dev_confusion.csv')
convert_word_output(dev_jieba1,dev_jieba1_e7_output,dev,jieba1_dev_e7_outcome)
get_results(jieba1_dev_outcome,'output_jieba1_e7/dev_output/jieba1_e7_dev_classification.txt','output_jieba1_e7/dev_output/jieba1_e7_dev_confusion.csv')

# get all results of (epoches=3,5,7) of Jieba2 system on dev dataset
convert_word_output(dev_jieba2,dev_jieba2_e3_output,dev,jieba2_dev_e3_outcome)
get_results(jieba2_dev_outcome,'output_jieba2_e3/dev_output/jieba2_e3_dev_classification.txt','output_jieba2_e3/dev_output/jieba2_e3_dev_confusion.csv')
convert_word_output(dev_jieba2,dev_jieba2_e5_output,dev,jieba2_dev_e5_outcome)
get_results(jieba2_dev_outcome,'output_jieba2_e5/dev_output/jieba2_e5_dev_classification.txt','output_jieba2_e5/dev_output/jieba2_e5_dev_confusion.csv')
convert_word_output(dev_jieba2,dev_jieba2_e7_output,dev,jieba2_dev_e7_outcome)
get_results(jieba2_dev_outcome,'output_jieba2_e7/dev_output/jieba2_e7_dev_classification.txt','output_jieba2_e7/dev_output/jieba2_e7_dev_confusion.csv')

# get all results of (epoches=3,5,7) of Jieba3 system on dev dataset
convert_word_output(dev_jieba3,dev_jieba3_e3_output,dev,jieba3_dev_e3_outcome)
get_results(jieba3_dev_outcome,'output_jieba3_e3/dev_output/jieba3_e3_dev_classification.txt','output_jieba3_e3/dev_output/jieba3_e3_dev_confusion.csv')
convert_word_output(dev_jieba3,dev_jieba3_e5_output,dev,jieba3_dev_e5_outcome)
get_results(jieba3_dev_outcome,'output_jieba3_e5/dev_output/jieba3_e5_dev_classification.txt','output_jieba3_e5/dev_output/jieba3_e5_dev_confusion.csv')
convert_word_output(dev_jieba3,dev_jieba3_e7_output,dev,jieba3_dev_e7_outcome)
get_results(jieba3_dev_outcome,'output_jieba3_e7/dev_output/jieba3_e7_dev_classification.txt','output_jieba3_e7/dev_output/jieba3_e7_dev_confusion.csv')

# get the test reuslts of jieba1 system (Epoch = 5)
convert_word_output(test_jieba3,test_jieba3_output,test,jieba3_outcome)
get_results(jieba3_outcome,'output_jieba3/jieba3_classification.txt','output_jieba3/jieba3_confusion.csv')

# get the test reuslts of char system (Epoch = 3)
convert_char_output(test,test_char_output,char_test_outcome)
get_results(char_test_outcome,'output_char_e3/char_classification.txt','output_char_e3/char_confusion.csv')

# get the heatmap (un-normalized and normalized) for Baseline system
baseline_gold_labels, baseline_prediction_labels = get_gold_predict(char_test_outcome)
baseline_labels = [
    'O', 'I-DISE', 'I-BODY', 'B-DISE', 'B-BODY', 'I-SUPP', 'I-CHEM', 'B-SUPP',
    'I-DRUG', 'B-CHEM', 'I-TREAT', 'B-SYMP', 'B-TREAT', 'I-EXAM','B-DRUG', 'I-SYMP',
    'B-EXAM', 'B-TIME', 'I-TIME', 'B-INST', 'I-INST'
]

categdf=pd.DataFrame({
                 'y_true': baseline_gold_labels,
                 'y_pred': baseline_prediction_labels})
categdf['correct'] = categdf['y_true']==categdf['y_pred']

cm=confusion_matrix(categdf['y_true'], categdf['y_pred'], labels=baseline_labels)
plot_confusion_matrix(confusion_matrix=cm, class_names=baseline_labels, errors_only=False, fontsize=10)

# get the heatmap (un-normalized and normalized) for Jieba Full system
jieba3_gold_labels, jieba3_prediction_labels = get_gold_predict(jieba3_test_outcome)
jieba3_labels = ['O', 'B-BODY', 'I-SUPP', 'I-BODY', 'B-SUPP', 'B-SYMP', 'B-CHEM', 'I-SYMP', 'B-TIME', 'I-CHEM', 'B-DISE', 'I-DISE', 'B-TREAT', 'I-TIME', 'I-EXAM', 'B-EXAM', 'I-TREAT', 'B-DRUG', 'I-DRUG', 'B-INST', 'I-INST']


jieba3_categdf=pd.DataFrame({
                 'y_true': jieba3_gold_labels,
                 'y_pred': jieba3_prediction_labels})
jieba3_categdf['correct'] = jieba3_categdf['y_true']==jieba3_categdf['y_pred']

jieba3_cm=confusion_matrix(jieba3_categdf['y_true'], jieba3_categdf['y_pred'], labels=jieba3_labels)
# un-normalized heatmap
plot_confusion_matrix(confusion_matrix=jieba3_cm, class_names=jieba3_labels, errors_only=False, fontsize=10)
#normalized heatmap
