import numpy as np
import csv
import pandas as pd
import random
import numpy.random as nr
import sys
import h5py
import math
import MeCab
import re
import nltk
from gensim import corpora
from gensim import corpora, matutils
from keras.optimizers import SGD
nltk.download('stopwords')

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenize

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

data=pd.read_csv('/Users/mugikura/Downloads/Data.csv', encoding = "ISO-8859-1",header=None)

Sentence_before=data[1].values.tolist()
Value_before=data[0].values.tolist()

Sentence_learn=Sentence_before[0:10000]
Sentence_learn.extend(Sentence_before[len(Sentence_before)-10001:len(Sentence_before)-1])
Value_learn=Value_before[0:10000]
Value_learn.extend(Value_before[len(Value_before)-10001:len(Value_before)-1])

for i in range(len(Sentence_learn)):
    Sentence_learn[i]=Sentence_learn[i].lower()
    Sentence_learn[i]= Sentence_learn[i].split()
    Sentence_learn[i] = [s for s in Sentence_learn[i] if not s.startswith('@')]

from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))
Sentence_learn_check=[]
print(type(en_stops))

for i in range(len(Sentence_learn)):
    insert=[]
    for j in range(len(Sentence_learn[i])):
        word=Sentence_learn[i][j]
        if  word not  in en_stops:
            insert.append(word)
        
    Sentence_learn_check.append(insert)

dictionary = corpora.Dictionary(Sentence_learn_check)
dictionary.filter_extremes(no_below=20, no_above=0.3)
dict_inside=list(dictionary.token2id.keys())
dict_set= set(dict_inside)
dict_num=len(dictionary.token2id)
dictionary.save_as_text('Language_Processing,txt')

dense_learn=[]
for i in range(len(Sentence_learn)):
    tmp = dictionary.doc2bow(Sentence_learn[i])
    dense_learn.append(list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0]))

Value_learn=[]
Value_learn=Value_before[0:10000]
Value_learn.extend(Value_before[len(Value_before)-10001:len(Value_before)-1])

Value_Teach=[]

for i in range(len(Value_learn)):
    if Value_learn[i]==0:
        Value_Teach.append([1.0,0.0])
    else:
        Value_Teach.append([0.0,1.0])
        
Sentence_Learn_Final=[]
train_dataset=[]
for i in range(len(Sentence_learn_check)):
    insert=[]
    for j in range(len(Sentence_learn_check[i])):
        word=Sentence_learn_check[i][j]
        if word  in dict_set:
            insert.append(word)
    Sentence_Learn_Final.append(insert)
    
train_dataset=Sentence_Learn_Final

max=0
for i in range(len(train_dataset)):
    s=len(train_dataset[i])
    if max<s:
        max=s
        
size=max

model_emb=10

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(dict_num, model_emb),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(size)),
    tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None),
    tf.keras.layers.Dense(size, activation='relu'),
    keras.layers.Dropout(0.5, noise_shape=None, seed=None),
    tf.keras.layers.Dense(2,activation='softmax')
])
model.summary()

for i in range(len(train_dataset)):
    tasu=size-len(train_dataset[i])
    
    if(tasu<0):
        train_dataset[i].pop()
    tasu=size-len(train_dataset[i])
    
    for j in range(tasu):
        train_dataset[i].append(0)

inputData_test = np.array(train_dataset)
inputValue_test= np.array(Value_Teach)
print(inputValue_test)

learning_rate=0.01

model.compile(loss='mean_squared_error')

epoch=3
batch_size = 100

history=model.fit(inputData_test, inputValue_test, epochs=3, batch_size=200)