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
mc = ModelCreation()
X_train, X_test, y_train, y_test = mc.split_data(text, category)

# b) LSTM Model
num_words = 10000
nb_categories = np.shape(category)[1]
model = mc.lstm_model(num_words, nb_categories, embedding_output=64,
                      nodes=64,dropout=0.2)
plot_model(model)

# c) train the model
mc.train_model(LOG_PATH,model,X_train,y_train,X_test,y_test,epochs=10)

#%% STEP 7: Model Evaluation

me = ModelEvaluation()
y_true, y_pred = me.predict_model(model,X_test,y_test,nb_categories)
me.report_metrics(y_true,y_pred)

#%%
# STEP 8: Model Deployment
model.save(MODEL_SAVE_PATH)















