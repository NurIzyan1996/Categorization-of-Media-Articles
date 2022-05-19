# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:55:01 2022
Building a deep learning model to categorize unseen
articles into 5 categories namely Sport, Tech, Business, Entertainment and
Politics. 
@author: Nur Izyan Binti Kamarudin
"""

import os
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from modules import ExploratoryDataAnalysis,DataPreprocessing,ModelCreation,ModelEvaluation
#%% PATHS
URL = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
TOKENIZER_PATH = os.path.join(os.getcwd(), 'saved_model','tokenizer_data.json')
OHE_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model','ohe.pkl')
LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
#%% STEP 1: Data Loading
df = pd.read_csv(URL)
category = df['category']
text = df['text']

#%% STEP 2: Data Inspection

# a) checking any misspell and repeating words
print(category.unique())

# b) checking any HTML tags inside text data
text[100]
# Observation: there are many HTML tags in the text shown.

# c) checking the upper case character inside text
for index,words in enumerate(text):
    result = any(i.isupper() for i in text[index])
    
print(result == True)
# observation, all wrods in text data contains lower case character

#%% STEP 3: Data Cleaning
eda = ExploratoryDataAnalysis()
text = eda.remove_symbol(text) # to remove tags

#%% STEP 4:  Data vectorization

# a) vectorize the text data using Tokenization approach
text = eda.text_tokenize(text, TOKENIZER_PATH, num_words=10000, 
                         oov_token='<OOV>', prnt=True)

# b) padding the tokenized data
temp = ([np.shape(i) for i in text])
np.mean(temp) # get the mean for maxlen
# Observation: the mean shape is 393.86, therefore, the maxlen is 400
text = eda.text_pad_sequences(text, maxlen=400)

#%% STEP 5: Preprocessing
 
# a) encode 'category' using One Hot Encoder approach
data_pre = DataPreprocessing()
category = data_pre.one_hot_encoder(category, OHE_SAVE_PATH)

#%% STEP 6: Builing DL Model

# a) split train & test data
# X-text, Y-category
X_train, X_test, y_train, y_test = train_test_split(text, category, 
                                                    test_size=0.3, 
                                                    random_state=123)

# b) expand dimension into 3D array
X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)

# c) LSTM Model

mc = ModelCreation()

num_words = 10000
nb_categories = np.shape(category)[1]
model = mc.lstm_model(num_words, nb_categories, embedding_output=64,
                   nodes=64,dropout=0.2)

plot_model(model)

model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics='acc')

log_files = os.path.join(LOG_PATH, 
                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)

hist = model.fit(X_train, y_train, epochs=10,
                 validation_data=(X_test,y_test), 
                 callbacks=tensorboard_callback)

#%% STEP 7: Model Evaluation

predicted_y = np.empty([len(X_test), nb_categories])

for index, test in enumerate(X_test):
    predicted_y[index,:] = model.predict(np.expand_dims(test, axis=0))

#%% STEP 8: Model analysis
y_pred = np.argmax(predicted_y, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()
me.report_metrics(y_true,y_pred)

#%%
# STEP 9: Model Deployment
model.save(MODEL_SAVE_PATH)















