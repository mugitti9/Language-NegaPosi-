#!/usr/bin/env python
# coding: utf-8

# In[460]:


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


# In[461]:


from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


# In[462]:


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()


# In[463]:


data=pd.read_csv('training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1",header=None)


# In[464]:


Sentence_before=data[5].values.tolist()
Value_before=data[0].values.tolist()


# In[465]:


learn_length=500
test_length=500
Sentence_learn=Sentence_before[0:learn_length]
Sentence_learn.extend(Sentence_before[len(Sentence_before)-learn_length-1:len(Sentence_before)-1])
print(Sentence_learn[0])
Value_learn=Value_before[0:learn_length]
Value_learn.extend(Value_before[len(Value_before)-learn_length-1:len(Value_before)-1])
print(type(Value_learn[0]))
Sentence_test=Sentence_before[learn_length:test_length+learn_length]
Sentence_test.extend(Sentence_before[len(Sentence_before)-learn_length-test_length-1:len(Sentence_before)-learn_length-1])
Value_test=Value_before[learn_length:learn_length+test_length]
Value_test.extend(Value_before[len(Value_before)-learn_length-test_length-1:len(Value_before)-learn_length-1])


# In[466]:


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


# In[467]:


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


# In[468]:


dictionary = corpora.Dictionary(Sentence_learn_check)


# In[469]:


print(len(dictionary.token2id))
dictionary.filter_extremes(no_below=10,no_above=1)  #no_aboveは出現文書数/全文書数≤指定値」になるような語のみを保持。no_belowで最低出現文書数を指定
dict_inside=list(dictionary.token2id.keys())
dict_set= set(dict_inside)
dict_num=len(dictionary.token2id)
print(len(dictionary.token2id))


# In[470]:


dictionary.save_as_text('Language_Processing,txt')


# In[471]:


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
        


# In[472]:


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


# In[473]:


max=0
for i in range(len(train_dataset)):
    s=len(train_dataset[i])
    if max<s:
        max=s
        
size=max


# In[474]:


model_emb=10

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(dict_num, model_emb),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(size)),
    tf.keras.layers.Dense(size, activation='relu'),
    tf.keras.layers.Dropout(0.4, noise_shape=None, seed=None),
    tf.keras.layers.Dense(2,activation='softmax')
])
model.summary()


# In[475]:


for i in range(len(train_dataset)):
    tasu=size-len(train_dataset[i])
    
    if(tasu<0):
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


# In[476]:


inputData_train = np.array(train_dataset)
inputValue_train= np.array(Value_Teach)

inputData_test = np.array(test_dataset)
inputValue_test= np.array(Value_test_Teach)

print(len(Value_Teach))

num=len(test_dataset[0])
for i in range(len(test_dataset)):
    if(len(test_dataset[i])!=num):
        print(num)
        print(len(test_dataset[i]))


# In[477]:


learning_rate=0.01

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[478]:


epoch=3
batch_size = 100

history=model.fit(inputData_train, inputValue_train, epochs=20, batch_size=200)


# In[479]:


history.history


# In[480]:


test_loss,test_acc=model.evaluate(inputData_test,inputValue_test,verbose=0)


# In[481]:


print(test_acc)


# In[ ]:




