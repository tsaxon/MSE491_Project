#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Import libraries and tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from scipy.io import wavfile as wav

from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.optimizers import Adam
#from keras.utils import to_categorical


# In[8]:


# Load the dataset
df = pd.read_csv('UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv')
labels = list(df['class'].unique())


# In[9]:


# Look at example from each class
files = dict()
for i in range(len(labels)):
    tmp = data[data['class'] == labels[i]][:1].reset_index()
    path = 'UrbanSound8K/UrbanSound8K/audio/fold{}/{}'.format(tmp['fold'][0], tmp['slice_file_name'][0])
    files[labels[i]] = path
fig = plt.figure(figsize=(15,15))# Log graphic of waveforms to Comet
experiment.log_image('class_examples.png')
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i, label in enumerate(labels):
    fn = files[label]
    fig.add_subplot(5, 2, i+1)
    plt.title(label)
    data, sample_rate = librosa.load(fn)
    librosa.display.waveplot(data, sr= sample_rate)
plt.savefig('class_examples.png')


# In[ ]:




