#!/usr/bin/env python
# coding: utf-8

# In[204]:


import tensorflow as tf
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
import neologdn
nltk.download('stopwords')
nltk.download('punkt')


# In[205]:


from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


# In[206]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
tf.enable_v2_behavior()

print(tf.__version__)


# In[207]:


imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data()


# In[208]:


INDEX_FROM=3   # word index offset
word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id[" "] = 0
word_to_id[" "] = 1 # START
word_to_id["?"] = 2 # UNKNOWN

# Make ID to Word dictionary
id_to_word = {value:key for key,value in word_to_id.items()}

def restore_original_text(index_no):
    return (' '.join(id_to_word[id] for id in x_train[index_no] ))
def restore_original_text_test(index_no):
    return (' '.join(id_to_word[id] for id in x_test[index_no] ))


# In[209]:


Value_learn=[]
Value_before=[]
Sentence_learn=[]
Sentence_before=[]
Sentence_test=[]
Sentence_before_test=[]
Value_test=[]
Value_before_test=[]
N=10000

for i in range(len(x_train)):
    Sentence_learn.append(restore_original_text(i))
    Value_learn.append(y_train[i])
Sentence_before=Sentence_learn[0:N]
Sentence_learn=[]
Sentence_learn=Sentence_before
Sentence_before=[]
Value_before=Value_learn[0:N]
Value_learn=[]
Value_learn=Value_before
Value_before=[]

for i in range(len(x_test)):
    Sentence_test.append(restore_original_text_test(i))
    Value_test.append(y_test[i].item())
Sentence_before_test=Sentence_test[0:N]
Sentence_test=[]
Sentence_test=Sentence_before_test
Sentence_before_test=[]
Value_before_test=Value_test[0:N]
Value_test=[]
Value_test=Value_before_test
Value_before_test=[]


# In[210]:


Sentence_learn_before=[]
for i in range(len(Sentence_learn)):
    Sentence=[]
    Sentence_learn[i]=neologdn.normalize(Sentence_learn[i])
    Sentence_learn[i]= re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', Sentence_learn[i])
    Sentence_learn[i]= re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', Sentence_learn[i])
    Sentence_learn[i]= re.sub(r'@/[\w/:%#\$&\?\(\)~\.=\+\-]+', '', Sentence_learn[i])
    Sentence_learn[i]= re.sub(r'@[\w/:%#\$&\?\(\)~\.=\+\-]+', '', Sentence_learn[i])
    Sentence_learn[i] = re.sub(r'[!-/:-@[-`{-~]', r' ', Sentence_learn[i])
    Sentence_learn[i]=Sentence_learn[i].lower()
    Sentence=nltk.word_tokenize(Sentence_learn[i])
    Sentence_learn_before.append(Sentence)
Sentence_learn=[]
Sentence_learn=Sentence_learn_before
Sentence_learn_before=[]

Sentence_test_before=[]
for i in range(len(Sentence_test)):
    Sentence=[]
    Sentence_test[i]=neologdn.normalize(Sentence_test[i])
    Sentence_test[i]= re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', Sentence_test[i])
    Sentence_test[i]= re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', Sentence_test[i])
    Sentence_test[i]= re.sub(r'@/[\w/:%#\$&\?\(\)~\.=\+\-]+', '', Sentence_test[i])
    Sentence_test[i]= re.sub(r'@[\w/:%#\$&\?\(\)~\.=\+\-]+', '', Sentence_test[i])
    Sentence_test[i] = re.sub(r'[!-/:-@[-`{-~]', r' ', Sentence_test[i])
    Sentence_test[i]=Sentence_test[i].lower()
    Sentence=nltk.word_tokenize(Sentence_test[i])
    Sentence_test_before.append(Sentence)
Sentence_test=[]
Sentence_test=Sentence_test_before
Sentence_test_before=[]


# In[211]:


from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))
Sentence_learn_check=[]
Sentence_test_check=[]

for i in range(len(Sentence_learn)):
    insert=[]
    for j in range(len(Sentence_learn[i])):
        word=Sentence_learn[i][j]
        if  word not  in en_stops:
            insert.append(word)
        
    Sentence_learn_check.append(insert)
    
for i in range(len(Sentence_test)):
    insert=[]
    for j in range(len(Sentence_test[i])):
        word=Sentence_test[i][j]
        if  word not  in en_stops:
            insert.append(word)
        
    Sentence_test_check.append(insert)


# In[212]:


dictionary = corpora.Dictionary(Sentence_learn_check)


# In[213]:


print(len(dictionary.token2id))
dictionary.filter_extremes(no_below=50,no_above=1)  #no_aboveは出現文書数/全文書数≤指定値」になるような語のみを保持。no_belowで最低出現文書数を指定
dict_inside=list(dictionary.token2id.keys())
dict_set= set(dict_inside)
dict_num=len(dictionary.token2id)
print(len(dictionary.token2id))


# In[214]:


dictionary.save_as_text('Language_Processing,txt')


# In[215]:


Value_Teach=[]
Value_test_Teach=[]

#0のときがnegative,0のときがpositive
for i in range(len(Value_learn)):
    if Value_learn[i]==0:
        Value_Teach.append(0)
    else:
        Value_Teach.append(1)
        
for i in range(len(Value_test)):
    if Value_test[i]==0:
        Value_test_Teach.append(0)
    else:
        Value_test_Teach.append(1)
        


# In[216]:


Sentence_Learn_Final=[]
train_dataset=[]
for i in range(len(Sentence_learn_check)):
    insert=[]
    for j in range(len(Sentence_learn_check[i])):
        word=Sentence_learn_check[i][j]
        if word  in dict_set:
            insert.append(word)
    Sentence_Learn_Final.append(insert)
    
for i in range(len(Sentence_Learn_Final)):
    insert=[]
    for j in range(len(Sentence_Learn_Final[i])):
        word=Sentence_Learn_Final[i][j]
        insert.append(dictionary.token2id[word])
    train_dataset.append(insert)


Sentence_Test_Final=[]
test_dataset=[]
for i in range(len(Sentence_test_check)):
    insert=[]
    for j in range(len(Sentence_test_check[i])):
        word=Sentence_test_check[i][j]
        if word  in dict_set:
            insert.append(word)
    Sentence_Test_Final.append(insert)
    
for i in range(len(Sentence_Test_Final)):
    insert=[]
    for j in range(len(Sentence_Test_Final[i])):
        word=Sentence_Test_Final[i][j]
        insert.append(dictionary.token2id[word])
    test_dataset.append(insert)


# In[217]:


max=0
for i in range(len(train_dataset)):
    s=len(train_dataset[i])
    if max<s:
        max=s
        
max=50
size=max
print(max)


# In[218]:


model_emb=10

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(dict_num, model_emb),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(size)),
    tf.keras.layers.Dense(size, activation='relu'),
    tf.keras.layers.Dropout(0.4, noise_shape=None, seed=None),
    tf.keras.layers.Dense(2,activation='softmax')
])
model.summary()


# In[219]:


for i in range(len(train_dataset)):
    tasu=size-len(train_dataset[i])
    
    if(tasu<0):
        train_dataset[i].pop()
        for j in range((-1)*tasu):
            train_dataset[i].pop()
    tasu=size-len(train_dataset[i])
    
    for j in range(tasu):
        train_dataset[i].append(0)
        
for i in range(len(test_dataset)):
    tasu=size-len(test_dataset[i])
    
    if(tasu<0):
        for j in range((-1)*tasu):
            test_dataset[i].pop()
    tasu=size-len(test_dataset[i])
    
    for j in range(tasu):
        test_dataset[i].append(0)


# In[220]:


inputData_train = np.array(train_dataset)
inputValue_train= np.array(Value_Teach)

inputData_test = np.array(test_dataset)
inputValue_test= np.array(Value_test_Teach)

print(type(inputData_train[0]))
for i in range(len(inputData_train)):
    if(len(train_dataset[i])!=max):
        print(len(train_dataset[i]))
    #if(len(train_data))
print(type(inputValue_train[0]))

num=len(test_dataset[0])
for i in range(len(test_dataset)):
    if(len(test_dataset[i])!=num):
        print(num)
        print(len(test_dataset[i]))


# In[221]:


learning_rate=0.01

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[222]:


epoch=3
batch_size = 100

history=model.fit(inputData_train, inputValue_train, epochs=20, batch_size=200)


# In[223]:


history.history


# In[224]:


test_loss,test_acc=model.evaluate(inputData_test,inputValue_test,verbose=0)


# In[225]:


print(test_acc)


# In[ ]:




