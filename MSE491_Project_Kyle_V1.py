#!/usr/bin/env python
# coding: utf-8

# # You might have to change some file paths

# In[1]:


# Import libraries
from comet_ml import Experiment
import IPython.display as ipd
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import struct
from scipy.io import wavfile as wav
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Create experiment
experiment = Experiment(api_key="EK1EL2E5mW0pKn8G0MQpLgFhJ", project_name="MSE491_Project")


# In[3]:


# Function to read wav files
def read_file_properties(filename):
        wave_file = open(filename,"rb")
        riff = wave_file.read(12)
        fmt = wave_file.read(36)
        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]
        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I",sample_rate_string)[0]        
        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H",bit_depth_string)[0]
        return (num_channels, sample_rate, bit_depth)


# In[4]:


# Load dataset and get labels
df = pd.read_csv('UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv')
labels = list(df['class'].unique())


# In[5]:


# Look at example from each class
files = dict()
for i in range(len(labels)):
    tmp = df[df['class'] == labels[i]][:1].reset_index()
    path = 'UrbanSound8K/UrbanSound8K/audio/fold{}/{}'.format(tmp['fold'][0], tmp['slice_file_name'][0])
    files[labels[i]] = path
# Plot examples
fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i, label in enumerate(labels):
    fn = files[label]
    fig.add_subplot(5, 2, i+1)
    plt.title(label)
    data, sample_rate = librosa.load(fn)
    librosa.display.waveplot(data, sr= sample_rate)
plt.savefig('class_examples.png')
# Log image to comet
experiment.log_image('class_examples.png')


# In[6]:


# Log wav files to comet for debugging
for label in labels:
    fn = files[label]
    experiment.log_audio(fn, metadata = {'name': label})
audiodata = []
for index, row in df.iterrows():
    fn = 'UrbanSound8K/UrbanSound8K/audio/fold{}/{}'.format(row['fold'], row['slice_file_name'])
    data = read_file_properties(fn)
    audiodata.append(data)


# In[7]:


# Convert to pd
audiodf = pd.DataFrame(audiodata, columns=['num_channels', 'sample_rate', 'bit_depth'])


# In[8]:


# Get mfccs
fn = 'UrbanSound8K/UrbanSound8K/audio/fold1/191431-9-0-66.wav'
librosa_audio, librosa_sample_rate = librosa.load(fn)
scipy_sample_rate, scipy_audio = wav.read(fn)
mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc = 40)
# Plot mfccs
plt.figure(figsize=(8,8))
librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')
plt.savefig('MFCCs.png')
# Log image to comet
experiment.log_image('MFCCs.png')


# In[9]:


# Function to extract features
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)    
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None  
    return mfccsscaled


# In[10]:


# Extract features 
metadata = df
features = []
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath('UrbanSound8K/UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    class_label = row["class"]
    data = extract_features(file_name)
    features.append([data, class_label])


# In[11]:


# Convert to pd
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])


# In[12]:


# Convert features and labels to np
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
# Encode labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))


# # Everything above here is purely getting the data, code is very slightly modified from the doc.
# # Below here is original code of making the NN

# In[13]:


# Train-test split

