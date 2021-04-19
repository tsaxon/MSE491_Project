# Import libraries and tool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav

from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical

# from comet_ml import Experiment

# experiment = Experiment(api_key="API_KEY",
#                         project_name="urbansound8k")

data = pd.read_csv('D:/PythonML/UrbanSound8K/metadata/UrbanSound8K.csv')
labels = list(data['class'].unique())

files = dict()
for i in range(len(labels)):
    tmp = data[data['class'] == labels[i]][:1].reset_index()
    path = 'UrbanSound8K/UrbanSound8K/audio/fold{}/{}'.format(tmp['fold'][0], tmp['slice_file_name'][0])
    files[labels[i]] = path