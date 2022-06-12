import os
import re
import json
import numpy as np
import pickle
import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Dropout,LSTM,Embedding,Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from tensorflow.keras.callbacks import TensorBoard
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
    
    def split_data(self,data_1,data_2):
        X_train, X_test, y_train, y_test = train_test_split(data_1, 
                                                            data_2, 
                                                            test_size=0.3, 
                                                            random_state=123)
        X_train = np.expand_dims(X_train,-1)
        X_test = np.expand_dims(X_test,-1)
        return X_train, X_test, y_train, y_test
    
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
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', 
                      metrics='acc')
        return model
    
    def train_model(self,path,model,data_1,data_2,data_3,data_4,epochs):
        log = os.path.join(path,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tensorboard_callback = TensorBoard(log_dir=log, histogram_freq=1)
        return model.fit(data_1, data_2, epochs=epochs, 
                         validation_data=(data_3,data_4),
                         callbacks=tensorboard_callback)
    
class ModelEvaluation():
    def __init__(self):
        pass
    
    def predict_model(self,model,data_1,data_2,size_1):
        predicted = np.empty([len(data_1), size_1])

        for index, test in enumerate(data_1):
            predicted[index,:] = model.predict(np.expand_dims(test, axis=0))

        y_true = np.argmax(data_2, axis=1)
        y_pred = np.argmax(predicted, axis=1)
        return y_true, y_pred
    
    def report_metrics(self,y_true,y_pred):
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))