# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:46:55 2022

@author: Nur Izyan Binti Kamarudin
"""

import re
import json
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense,Dropout,LSTM,Embedding,Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

#%%
class ExploratoryDataAnalysis():
    
    def __init__(self):
        pass
    
    def remove_symbol(self,data):
        for index, words in enumerate(data):
            data[index] = re.sub(r"[^\w]", " ", words)
        return data
    
    def text_tokenize(self,data,token_save_path,
                           num_words,oov_token,prnt=False):
        
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        
        token_json = tokenizer.to_json()
        
        with open(token_save_path,'w') as json_file:
            json.dump(token_json, json_file)
            
        word_index = tokenizer.word_index
        
        if prnt == True:
            print(dict(list(word_index.items())[0:10]))
            
        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def text_pad_sequences(self,data,maxlen):
        return pad_sequences(data,maxlen=maxlen,padding='post',
                             truncating='post')
    
class DataPreprocessing():
    
    def __init__(self):
        pass
    
    def one_hot_encoder(self, data,path):  
        enc = OneHotEncoder(sparse=False) 
        data = enc.fit_transform(np.expand_dims(data,axis=-1))
        pickle.dump(enc, open(path, 'wb'))
        return data
    
class ModelCreation():
    def __init__(self):
        pass
    
    def lstm_model(self,num_words,nb_categories,embedding_output,
                   nodes,dropout):
        model = Sequential()
        model.add(Embedding(num_words, embedding_output)) 
        model.add(Bidirectional(LSTM(nodes, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))  
        model.summary()
        return model
    
class ModelEvaluation():
    def __init__(self):
        pass
    
    def report_metrics(self,y_true,y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))