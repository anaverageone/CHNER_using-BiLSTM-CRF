import os,json,collections, csv
import openpyxl, pprint
import pandas as pd
from collections import Counter
os.chdir(r'C:\Users\anaverageone\htlt_env\CLTL\AITutorial-2022-ChineseNER-main')
os.getcwd()

chatbot_corpus = 'seg_corpus\chatbot_corpus.txt'

CBLUE_corpus = 'seg_corpus\CBLUE_corpus.txt'
CHIP_CDN = 'seg_corpus\CBLUE-main\CHIP-CDN\CHIP_CDN.txt'
CMeEE_V2 = 'seg_corpus\CBLUE-main\CMeEE-V2\CMeEE_V2.txt'
CMeIE_V2 = 'seg_corpus\CBLUE-main\CMeIE-V2\CMeIE_V2.txt'
IMCS_V2 = 'seg_corpus\CBLUE-main\IMCS-V2-NER\IMCS_V2.txt'

THUOCL_corpus = 'seg_corpus\THUOCL_corpus.txt'

ROLING_corpus = 'seg_corpus\ROLING\ROLING_word_corpus.txt'
OPEN_corpus = 'seg_corpus\OPEN_corpus.txt'
FUSION_corpus = 'seg_corpus\ROLING\FUSION_corpus.txt'

def chatbot_ner(directory,outfile):
    ner_list = []
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                ner = text.split('\n')
                ner_list += ner
    with open(outfile, 'w', encoding='utf-8') as file:
        for item in ner_list:
            file.write(f"{item}\n")

chatbot_ner('seg_corpus\chatbot-base-on-Knowledge-Graph-master\data_dictionary',chatbot_corpus)
def CHIP_CDN_ner(directory,outfile):
    ner_list = []
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                for item in data:
                    doc_str = item['normalized_result']
                    for x in doc_str.split('##'):
                        ner_list.append(x)
    with open(outfile, 'w', encoding='utf-8') as file:
        for item in ner_list:
            file.write(f"{item}\n")
CHIP_CDN_ner('seg_corpus\CBLUE-main\CHIP-CDN',CHIP_CDN)
def IMCS_ner(directory, outfile):
    ner_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            for k,v in data.items():
                diagnosis = v['diagnosis']
                ner_list.append(diagnosis)
                implicit_info = v['implicit_info']['Symptom']
                for k, v in implicit_info.items():
                    ner_list.append(k)
    with open(outfile, 'w', encoding='utf-8') as file:
        for item in ner_list:
            file.write(f"{item}\n")
            
IMCS_ner('seg_corpus\CBLUE-main\IMCS-V2-NER', IMCS_V2)
def THUOCL_ner(infile, outfile):
    ner_dict = {}
    with open(infile, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line) == 2:
                k, v = line[0], line[1]
                if v.isdigit():  # Check if v contains only digits
                    v = int(v)
                    if k not in ner_dict:
                        ner_dict[k] = v
                    else:
                        ner_dict[k] += v
            else:
                print(f"Ignoring line: {line}")
                    
    with open(outfile, 'w', encoding='utf-8') as file:
        for key, value in ner_dict.items():
            file.write(f"{key}\n")
            
THUOCL_ner(THUOCL_medical, THUOCL_nofreq)
def CMeEE_ner(directory, outfile):
    ner_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as file:
                json_data = json.loads(file.read())
                for line in json_data:
                    if line['entities']:
                        for item in line['entities']:
                            ner_list.append(item['entity'])
                    else:
                        continue
    with open(outfile, 'w', encoding='utf-8') as file:
        for item in ner_list:
            file.write(f"{item}\n")

CMeEE_ner('seg_corpus\CBLUE-main\CMeEE-V2', CMeEE_V2)

def CMeIE_ner(directory, outfile):
    ner_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.jsonl'):
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    line_data = json.loads(line.strip())
                    if line_data['spo_list']:
                        subject = line_data["spo_list"][0]["subject"] 
                        ner_list.append(subject) 
                    else:
                        continue
    with open(outfile, 'w', encoding='utf-8') as file:
        for item in ner_list:
            file.write(f"{item}\n")
CMeIE_ner('seg_corpus\CBLUE-main\CMeIE-V2', CMeIE_V2)

def CMeIE_ner(directory, outfile):
    ner_list = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.jsonl'):
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    line_data = json.loads(line.strip())
                    if line_data['spo_list']:
                        subject = line_data["spo_list"][0]["subject"] 
                        ner_list.append(subject) 
                    else:
                        continue
    with open(outfile, 'w', encoding='utf-8') as file:
        for item in ner_list:
            file.write(f"{item}\n")
CMeIE_ner('seg_corpus\CBLUE-main\CMeIE-V2', CMeIE_V2)

def THUOCL_ner(infile, outfile):
    ner_list = []
    with open(infile, 'r', encoding='utf-8') as file:
        for line in file:
            ner_list.append(line.split('\t')[0])
    with open(outfile, 'w', encoding='utf-8') as file:
        for item in ner_list:
            file.write(f"{item}\n")
THUOCL_ner('seg_corpus\THUOCL-master\data\THUOCL_medical.txt', THUOCL_corpus)
def ROLING_ner(infile, outfile):
    ner_list = []
    with open(infile, 'r', encoding='utf-8') as file:
        for line in file:
            ner_list.append(line.split(' ')[0])
    with open(outfile, 'w', encoding='utf-8') as file:
        for item in ner_list:
            file.write(f"{item}\n")
ROLING_ner('data/train_word.txt', ROLING_corpus)

def combine_all(directory, outfile):
    result_dict = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    item = line.strip()
                    if item not in result_dict:
                        result_dict[item] = 1
                    else:
                        continue  
    with open(outfile, 'w', encoding='utf-8') as file:
        for item in result_dict.keys():
            file.write(f"{item}\n")
combine_all('seg_corpus\CBLUE-main',CBLUE_corpus)
combine_all('seg_corpus',OPEN_corpus)
combine_all('seg_corpus\ROLING',FUSION_corpus)