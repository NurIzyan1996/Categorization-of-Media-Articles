# Categorization of Media Articles
Text documents are essential as they are one of the richest sources of data for businesses. Text documents often contain crucial information which might shape
the market trends or influence the investment flows. Therefore, companies often hire analysts to monitor the trend via articles posted online, tweets on social media
platforms such as Twitter or articles from newspaper. However, some companies may wish to only focus on articles related to technologies and politics. Thus,
filtering of the articles into different categories is required. 

In this repository, I have created an LSTM deep learning model to ccategorize unseen articles into 5 categories namely Sport, Tech, Business, Entertainment and
Politics.

# Description
This repository contains 2 python files (train.py, modules.py).

train.py contains the codes to build a deep learning model and train on the dataset.

module.py contains the codes where there are class and functions to be used in train.py.

#How run Tensorboard

1. Clone this repository and use the model.h5, ohe.pkl and tokenizer_data.json(inside saved_model folder) to deploy on your dataset.
2. Run tensorboard at the end of training to see how well the model perform via Anaconda prompt. Activate the correct environment.
3. Type "tensorboard --logdir "the log path"
4. Paste the local network link into your browser and it will automatically redirected to tensorboard local host and done! Tensorboard is now can be analyzed.

# The Architecture of Model
![The Architecture of Model](model_architecture.PNG)

# The Performance of model
![The Performance of model](model_performance.PNG)

# Tensorboard screenshot from my browser
![Tensorboard](tensorboard.PNG)

# Discussion

# Discussion
Based on the assignment given, we are required to create a deep learning model with accuracy 70%. I have succcessfully produce an LSTM model that produces 95% accuracy.

In my opinion, the 

My intake from the training dataset is we would need more customers' information to be added in the data so that the model can learn the pattern of the data very efficiently. Moreover, other approachs such as adding more layers, increasing number of nodes and epochs can be done to gain higher accuracy.

Throughout of doing this process, I have spent hours on cleaning the data only. I have the difficulty at imposing the scaling models in three python files. Hence, the process of removing the NaN values did not go smoothly.

In conclusion, the performance of the deep learning model that i build is poorly accurate which will cause the prediction of any new data will be unaccurate. In the future, this will cause a company fail to design a marketing strategy to target the most profitable segments.
