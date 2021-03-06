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

from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix, plot_confusion_matrix
import time

#%% Progress bar : 
    # use in for loops as such:
    # for i in tqdm(range(69))
from tqdm import tqdm

# In[2]:


# Create experiment
experiment = Experiment(
    api_key="hfHGUgNto54Kw0GpWvAkNj7wH",
    project_name="mse491-urbansound8k-tim-s",
    workspace="tsaxon",
)


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
path1 = 'D:/PythonML/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv'
df = pd.read_csv(path1)
labels = list(df['class'].unique())


# In[5]:


# Look at example from each class
files = dict()
for i in tqdm(range(len(labels))):
    tmp = df[df['class'] == labels[i]][:1].reset_index()
    path = 'D:/PythonML/UrbanSound8K/UrbanSound8K/audio/fold{}/{}'.format(tmp['fold'][0], tmp['slice_file_name'][0])
    files[labels[i]] = path
# Plot examples
fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(hspace=0.4, wspace=0.4)


for i, label in tqdm(enumerate(labels)):
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
print('\nLog wav files to comet for debugging\n')
for label in tqdm(labels):
    fn = files[label]
    experiment.log_audio(fn, metadata = {'name': label})
audiodata = []
for index, row in tqdm(df.iterrows()):
    fn = 'D:/PythonML/UrbanSound8K/UrbanSound8K/audio/fold{}/{}'.format(row['fold'], row['slice_file_name'])
    data = read_file_properties(fn)
    audiodata.append(data)


# In[7]:


# Convert to pd
audiodf = pd.DataFrame(audiodata, columns=['num_channels', 'sample_rate', 'bit_depth'])


# In[8]:


# Get mfccs
fn = 'D:/PythonML/UrbanSound8K/UrbanSound8K/audio/fold1/191431-9-0-66.wav'
librosa_audio, librosa_sample_rate = librosa.load(fn)
scipy_sample_rate, scipy_audio = wav.read(fn)
mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc = 40)
# Plot mfccs
plt.figure(figsize=(8,8))
librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')
plt.title('mfccs')
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
        print("Error encountered while parsing file: ", file_name)
        return None  
    return mfccsscaled


# In[10]:


# Extract features
print('\nExtract features\n')
metadata = df
features = []
for index, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath('D:/PythonML/UrbanSound8K/UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
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

print('\ndone\n')
# # Everything above here is purely getting the data, code is very slightly modified from the doc.
# # Below here is original code of making the NN

# In[13]:
# Train-test split
print('\nSplitting Train / Test Data\n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, yy, test_size=0.2, random_state=127)

print('Xtrain and Xtest sizes are: ', Xtrain.shape, Xtest.shape)
print('ytrain and ytest sizes are: ', ytrain.shape, ytest.shape)
 #%%
print('\nA peak at featuresdf:\n', featuresdf.head())

#%% Feedforward Model

num_labels = yy.shape[1]

model = keras.Sequential(
    [
     keras.layers.Input(shape=(Xtrain.shape[1])),
     keras.layers.Dense(256, activation="relu", name="layer1", ),
     keras.layers.Dropout(0.5),
     keras.layers.Dense(256, activation="relu", name="layer2", ),
     keras.layers.Dropout(0.5),
     # keras.layers.Dense(256, activation="relu", name="layer3", ),
     # keras.layers.Dropout(0.5),
     keras.layers.Dense(num_labels, activation="softmax", name="layer4", ),
     ])
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.summary()
score = model.evaluate(Xtest,ytest, verbose=1)
accuracy = 100*score[1]

print("\nPre-training accuracy: %.4f%%" % accuracy)


#%% How our example trains

print('\nMULTI-LABEL: FeedForwardNN Fitting:\n')
start = time.time()
num_epochs = 50
num_batch_size = 32

model.fit(Xtrain, ytrain, 
          batch_size = 32, 
          epochs = 50, 
          validation_data=(Xtest, ytest), 
          verbose=1)

# Evaluating the model on the test and train datasets

score = model.evaluate(Xtest, ytest, verbose=0)
end = time.time()
print("Testing Accuracy: {0:.2%}".format(score[1]))
score = model.evaluate(Xtrain, ytrain, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

#%% MLP: Multi Layer Perceptron : Uses Backpropagation
from sklearn.neural_network import MLPClassifier
start = time.time()

mlp = MLPClassifier(
     hidden_layer_sizes = (40,256,10),
     activation = 'relu',
     solver = 'adam',
     learning_rate = 'constant',
     batch_size = 32,
     random_state= 69,
    )

mlp.fit(Xtrain, ytrain)

score = model.evaluate(Xtest, ytest, verbose=0)
end = time.time()
print("MLP Testing Accuracy: {0:.2%}".format(score[1]))
score = model.evaluate(Xtrain, ytrain, verbose=0)
print("MLP Training Accuracy: {0:.2%}".format(score[1]))
print('\n\nDone in %.2f seconds.\n\n' % (end-start))
#%%
ypred = mlp.predict(Xtest)

print(confusion_matrix(ytest.argmax(axis=1), ypred.argmax(axis=1)))
#%% SVM : SUPPORT VECTOR MACHINES


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

start = time.time()
print('\nSINGLE LABEL: SVM: Linear SVC: One-vs-Rest\n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=127)
# One-vs-Rest
clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    LinearSVC(random_state=69, tol=1e-5))

clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('Linear SVM test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))
# One-vs-One svm.SVC()

#%% 
from sklearn.svm import SVC
start = time.time()
print('\nSINGLE (multi?) LABEL: SVM: SVC: One-vs-One\n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=69)

# One-vs-Rest
clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    SVC(kernel='rbf',random_state=69, tol=1e-5,verbose=0))

clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('SVC test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))
# One-vs-One svm.SVC()



#%%
# from sklearn.neighbors import KNeighborsClassifier

#%% 
from sklearn.svm import NuSVC
start = time.time()
print('\nSINGLE LABEL: SVM: NuSVC: One-vs-One\n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=69)
# One-vs-Rest
clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    NuSVC(kernel='rbf',random_state=69, tol=1e-5,verbose=0))

clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('SVC test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

#%% 

from sklearn.tree import DecisionTreeClassifier
start = time.time()
print('\nMULTI-LABEL: Decision Tree One-vs-One\n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, yy, test_size=0.2, random_state=69)
# One-vs-Rest
clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    DecisionTreeClassifier(random_state=69))

clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('SVC test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

#%% 

from sklearn.tree import DecisionTreeClassifier
start = time.time()
print('\nMULTI-LABEL: Decision Tree One-vs-One\n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, yy, test_size=0.2, random_state=69)

# One-vs-Rest

# clf = make_pipeline(StandardScaler(),
#                     # KMeans(n_clusters=50,random_state=69),
#                     DecisionTreeClassifier(random_state=69))
# clf = DecisionTreeClassifier(random_state=69)

# # Above is unecessary, gives same results as line below
clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('SVC test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

# y_pred = clf.predict(Xtest);
# print(multilabel_confusion_matrix(ytest, y_pred))
# print(classification_report(ytest, y_pred))
#%%
from sklearn.utils.multiclass import type_of_target
print(type_of_target(yy))

print('\ndone\n')


experiment.end()