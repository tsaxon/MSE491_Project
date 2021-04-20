#%% Imports

from comet_ml import Experiment


experiment = Experiment(api_key="API_KEY",
                        project_name="urbansound8k")

import IPython.display as ipd
import numpy as np
import pandas as pd
import librosa
from librosa import display
import matplotlib.pyplot as plt

from scipy.io import wavfile as wav
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical

#%% littterally doesnt work



FILEPATH = 'D:/PythonML/UrbanSound8K/metadata/UrbanSound8K.csv'

data = pd.read_csv(FILEPATH)

labels = list(data['class'].unique())

files = dict()

FILEPATH2 = 'D:/PythonML/UrbanSound8K/audio/fold{}/{}'

for i in range(len(labels)):
    tmp = data[data['class'] == labels[i]][:1].reset_index()
    path = FILEPATH2.format(tmp['fold'][0], tmp['slice_file_name'][0])
    
    files[labels[i]] = path
    
fig = plt.figure(figsize=(15,15))# Log graphic of waveforms to Comet

#%% 
experiment.log_image('class_examples.png')
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i, label in enumerate(labels):
    fn = files[label]
    fig.add_subplot(5, 2, i+1)
    plt.title(label)
    data, sample_rate = librosa.load(fn)
    librosa.display.waveplot(data, sr= sample_rate)
plt.savefig('class_examples.png')  
    
