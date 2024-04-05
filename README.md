# CHNER_using-BiLSTM-CRF
CHNER_using BiLSTM-CRF

## Steps 1 Directories

+ `Predict` : test predictions
+ `data` : input data
+ `package` : model
+ `saved_model` : trained models with parameters and scores
+ `embedding`：w2v embeddings 

## Steps 2 Evironment initlization
+ conda create --name toturial python=3.7
+ pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
+ pip install pandas==1.3.5
+ pip install tqdm==4.62.3
+ pip install gensim==4.2.0

## Steps 3 Pre-training word embeddings (w2v)
### Data Format
Use the data for training embeddings. Separate characters with spaces as shown below:：
```
当 希 望 工 程 救 助 的 百 万 儿 童 成 长 起 来 ， 科 教 兴 国 蔚 然 成 风 时 ， 今 天 有 收 藏 价 值 的 书 你 没 买 ， 明 日 就 叫 你 悔 不 当 初 ！ 
藏 书 本 来 就 是 所 有 传 统 收 藏 门 类 中 的 第 一 大 户 ， 只 是 我 们 结 束 温 饱 的 时 间 太 短 而 已 。 
因 有 关 日 寇 在 京 掠 夺 文 物 详 情 ， 藏 界 较 为 重 视 ， 也 是 我 们 收 藏 北 史 料 中 的 要 件 之 一 。 
```

### Train
```
python word2vec_train.py
```

## Steps 4 training BiLSTM-CRF  
### Data Format
Use the BIO tagging format. Separate characters and labels with spaces. Separate sentences with blank lines as shown below:
#### Train and Evaluation file
```
淮 B-LOC
科 O
技 O
集 O
市 O
， O
还 O
吸 O
引 O
了 O
联 B-ORG
合 I-ORG
国 I-ORG
工 I-ORG
业 I-ORG
发 I-ORG
展 I-ORG
组 I-ORG
织 I-ORG
中 I-ORG
国 I-ORG
投 I-ORG
资 I-ORG
促 I-ORG
进 I-ORG
处 I-ORG
等 O
国 O
内 O
外 O
十 O
多 O
家 O
知 O
名 O
投 O
资 O
商 O
。 O

他 O
们 O
不 O
仅 O
购 O
买 O
技 O
术 O
， O
而 O
且 O
引 O
进 O
科 O
技 O
人 O
才 O
。 O
```
#### Test file
```
沙
巴
航
空
服
务
中
心

什
么
是
格
鲁
吉
亚
统
一
共
产
党
```
### Parameter

--model: Train or Test
--save_model_name: Name of saved model parameters (.pt format)
--predict_name: Name of saved prediction file (.txt format)
--load_model_name: Name of model to load
--Train_data_path: Path to training data
--Eval_data_path: Path to evaluation data
--Test_data_path: Path to testing data
--Epoch: Number of training epochs
--lr: Learning rate
--batch_size: Batch size
--lstm_hidden_dim: LSTM hidden dimension
--lstm_dropout_rate: LSTM dropout rate
--seed: Random seed
--gpu: Which GPU to use
--embedding: File path of word embeddings
--dimension: Dimension of word embeddings

## Usage

### Train
```
python main.py --mode Train --save_model_name ROLING_char_e3.pt  --Epoch 3 --gpu 1 --Train_data_path data/train_char.txt  --Eval_data_path data/dev_char.txt  --embedding embedding/ROLING_w2v_char.txt

```

### Test
```
python main.py --mode Test --load_model_name tutorial --predict_name predict.txt --gpu 0
```
## Steps 5 Evaluation
First, execute turn_to_eval.py to generate eval.txt, then proceed to execute conlleval.py to get the result score.txt.
```
python turn_to_eval.py --truth truth.txt --prediction predict.txt 
python conlleval.py < eval.txt 

```
