import pandas as pd


label_cols_3 = ['toxic', 'severe_toxic', 'obscene']

train_df=pd.read_csv('data/train.csv')
EMBEDDING_FILE='F:\\glove.840B.300d.txt'
full_text=[str(i) for i in train_df['comment_text'].values]
y_train_3=train_df[label_cols_3].values.astype('int8')

label_cols_4 = ['toxic', 'severe_toxic', 'obscene','threat']
y_train_4=train_df[label_cols_4].values.astype('int8')
import numpy as np
def word_idx_map(text):
    tokens=[]
    for c in text:
        tokens+=c.split()
    tokens+=['<pad>']
    tokens+=['<unk>']
    # for idx,word in enumerate(tokens):
    #     print(idx,word)
    word_to_idx={word:idx for idx,word in enumerate(tokens)}
    return tokens,word_to_idx


def get_embedding(EMB_FILE,EMB_DIM,word_dict):
    print('reading word embedding data...')
    vec = []
    word2id = {}
    #import the word vec
    f = open(EMB_FILE,encoding='utf8')
    # info = f.readline()
    # print ('word vec info:',info)
    n=0
    while True:
        if n>=20000:
            break
        content = f.readline()
        if content == '':
            break
        try:
            content = content.strip().split()
            if content[0] in word_dict:
                word2id[content[0]] = len(word2id)
                content = content[1:]
                content = [float(i) for i in content]
                vec.append(content)
        except:
            print(content)
        n=n+1
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    print(len(word2id))

    dim = 300
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)
    return vec,word2id
def tokenize(text,word2idx,maxlen):
    tokens=[]
    for c in text:
        token=[word2idx[w] if w in word2idx else word2idx['UNK'] for w in c.split()]
        # print(token)
        if len(token)>maxlen:
            token=token[0:maxlen]
        else:
            token+=[0]*(maxlen-len(token))
        tokens.append(np.array(token))
    return np.array(tokens).astype('int32')
_,word_to_idx=word_idx_map(full_text)
embed_mat,word_dict=get_embedding(EMBEDDING_FILE,300,word_to_idx)


num_sample=len(list(full_text))
y_train=list(y_train_3)

train_text=full_text[0:100]
train_label=y_train_3[0:100]

test_text=full_text[100:110]
test_label=list(y_train_4)[100:110]

MAX_LENGTH=30
train_token=tokenize(train_text,word_dict,MAX_LENGTH)
test_token=tokenize(test_text,word_dict,MAX_LENGTH)

print(test_text[0])
train_labels=[]
for x in train_label:
    train_labels.append(np.array(x))
train_labels=np.array(train_labels)

test_labels=[]
n=0
for x in test_label:
    test_labels.append(np.array(x))
test_labels=np.array(test_labels)    

np.save('kbpdata/vec.npy', embed_mat)
np.save('kbpdata/train_token.npy', train_token)
np.save('kbpdata/test_token.npy', test_token)

np.save('kbpdata/train_labels.npy', train_labels)
np.save('kbpdata/test_labels.npy', test_labels)