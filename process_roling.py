import json
import argparse

def read_file(filename):
    with open(filename, "r", encoding="utf8") as f:
        data = []
        for line in f:
            if "target" in filename:
                line = line.replace("_", "-")
            data.append(line.strip().split())
    return data



def get_dict(sents, filter_word_num):
    word_count = {}
    for sent in sents:
        for word in sent:
            word_count[word] = word_count.get(word, 0) + 1
    
    # 过滤低频词
    word2id = {
        "[PAD]": 0, 
        "[UNK]": 1
    }
    for word, count in word_count.items():
        if count >= filter_word_num:
            word2id[word] = len(word2id)
    
    print("Total %d tokens, filter count<%d tokens, save %d tokens."%(len(word_count)+2, filter_word_num, len(word2id)))

    return word2id, word_count    

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    # task setting
    parser.add_argument('--sen_file', type=str, default='data/embedding/ROLING_embed_jieba3.txt')
    parser.add_argument('--dict_file', type=str, default='data/word2id_jieba3.json')
    parser.add_argument('--crf', action='store_true')
    
    args = parser.parse_args()

    sen_file = args.sen_file
    dict_file = args.dict_file
    sens = read_file(sen_file)
    # get dicts
    word2id, _ = get_dict(sens, filter_word_num=3)
    with open(dict_file, "w", encoding="utf-8") as f:
        json.dump(word2id, f, ensure_ascii=False)