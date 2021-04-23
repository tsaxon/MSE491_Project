#!/usr/bin/env python
# coding: utf-8

# # You might have to change some file paths

# In[1]:


# Import libraries
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

#Confusion Matrix Heat Map
import seaborn as sn
#%% Progress bar : 
    # use in for loops as such:
    # for i in tqdm(range(69))
from tqdm import tqdm



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
path1 = 'C:/Users/kylea/Downloads/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv'
df = pd.read_csv(path1)
labels = list(df['class'].unique())


# In[5]:


# Look at example from each class
files = dict()
for i in tqdm(range(len(labels))):
    tmp = df[df['class'] == labels[i]][:1].reset_index()
    path = 'C:/Users/kylea/Downloads/UrbanSound8K/UrbanSound8K/audio/fold{}/{}'.format(tmp['fold'][0], tmp['slice_file_name'][0])
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


# In[6]:


# Log wav files
print('\nLog wav files\n')
for label in tqdm(labels):
    fn = files[label]
audiodata = []
for index, row in tqdm(df.iterrows()):
    fn = 'C:/Users/kylea/Downloads/UrbanSound8K/UrbanSound8K/audio/fold{}/{}'.format(row['fold'], row['slice_file_name'])
    data = read_file_properties(fn)
    audiodata.append(data)


# In[7]:


# Convert to pd
audiodf = pd.DataFrame(audiodata, columns=['num_channels', 'sample_rate', 'bit_depth'])


# In[8]:


# Get mfccs
fn = 'C:/Users/kylea/Downloads/UrbanSound8K/UrbanSound8K/audio/fold1/191431-9-0-66.wav'
librosa_audio, librosa_sample_rate = librosa.load(fn)
scipy_sample_rate, scipy_audio = wav.read(fn)
mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc = 40)
# Plot mfccs
plt.figure(figsize=(8,8))
librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')
plt.title('mfccs')
plt.savefig('MFCCs.png')



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
    file_name = os.path.join(os.path.abspath('C:/Users/kylea/Downloads/UrbanSound8K/UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
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
# Encode labels : turn into an [x, n_labels] array, filled with 0's or 1's
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

layernodes1 = 256
layernodes2 = 256
model = keras.Sequential(
    [
     keras.layers.Input(shape=(Xtrain.shape[1])),
     keras.layers.Dense(layernodes1, activation="relu", name="layer1", ),
     keras.layers.Dropout(0.5),
     keras.layers.Dense(layernodes2, activation="relu", name="layer2", ),
     keras.layers.Dropout(0.5),
     # keras.layers.Dense(256, activation="relu", name="layer3", ),
     # keras.layers.Dropout(0.5),
     # keras.layers.Dense(256, activation="relu", name="layer4", ),
     # keras.layers.Dense(256, activation="relu", name="layer3", ),
     # keras.layers.Dropout(0.5),
     keras.layers.Dense(num_labels, activation="softmax", name="last_layer", ),
     ])
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.summary()
score = model.evaluate(Xtest,ytest, verbose=1)
accuracy = 100*score[1]

print("\nPre-training accuracy: %.4f%%" % accuracy)


print('\nFeedForwardNN Fitting: Encoded labels\n')
start = time.time()
batch_size = 32
epochs = 100
model.fit(Xtrain, ytrain, 
          batch_size = batch_size, 
          epochs = epochs,
          validation_data=(Xtest, ytest), 
          verbose=1)

# Evaluating the model on the test and train datasets

score = model.evaluate(Xtest, ytest, verbose=0)
end = time.time()
print("Sequential NN Testing Accuracy: {0:.2%}".format(score[1]))
score = model.evaluate(Xtrain, ytrain, verbose=0)
print("Sequential NN Training Accuracy: {0:.2%}".format(score[1]))

print('\n Batch size = %.0f, Epochs = %.0f, #Nodes = %.0f,%.0f' %(batch_size,epochs,layernodes1,layernodes2))
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

ypred = model.predict(Xtest)
cfm = confusion_matrix(ytest.argmax(axis=1), ypred.argmax(axis=1), normalize='pred')
plt.figure(figsize = (10,7))
plt.title('Sequential NN Decision Matrix')
sn.heatmap(cfm)

#%% MLP: Multi Layer Perceptron : Uses Backpropagation
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, yy, test_size=0.2, random_state=127)

start = time.time()

mlp = MLPClassifier(
     hidden_layer_sizes = (256,256,10),
     activation = 'relu',
     solver = 'adam',
     learning_rate = 'adaptive', # 'adaptive', 
     batch_size = 32,
     random_state= 69,
    )

mlp.fit(Xtrain, ytrain)

score = mlp.score(Xtest, ytest)
end = time.time()

print("MLP Testing Accuracy: {0:.2%}".format(score))

score = model.evaluate(Xtrain, ytrain, verbose=0)
print("MLP Training Accuracy: {0:.2%}".format(score[1]))

print('\n\nDone in %.2f seconds.\n\n' % (end-start))

ypred = mlp.predict(Xtest)
cfm = confusion_matrix(ytest.argmax(axis=1), ypred.argmax(axis=1), normalize='pred')
plt.figure(figsize = (10,7))
plt.title('Multi Layer Perceptron Decision Matrix')
sn.heatmap(cfm)

#%% SVM : SUPPORT VECTOR MACHINES


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Don't use "One Hot Encoding" : 
start = time.time()
print('\nSVM: Linear SVC: One-vs-Rest : Labels not encoded\n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=127)

# One-vs-Rest
clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    LinearSVC(random_state=69, tol=1e-5, max_iter = 1000))

clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('Linear SVC test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))
# One-vs-One svm.SVC()

ypred = clf.predict(Xtest)
cfm = confusion_matrix(ytest, ypred, normalize='true')
plt.figure(figsize = (10,7))
plt.title('Linear SVC Confusion Matrix')
sn.heatmap(cfm)

#%% 
from sklearn.svm import SVC
start = time.time()
print('\nSVM: SVC: One-vs-One : labels not encoded \n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=69)

# One-vs-Rest
clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    SVC(kernel='rbf',random_state=69, tol=1e-5,verbose=0))
# clf =  SVC(kernel='rbf',random_state=69, tol=1e-5,verbose=0)
clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('SVC test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

ypred = clf.predict(Xtest)
cfm = confusion_matrix(ytest, ypred, normalize='true')
plt.figure(figsize = (10,7))
plt.title('SVC Confusion Matrix')
sn.heatmap(cfm)

#%%
# from sklearn.neighbors import KNeighborsClassifier

#%% 
from sklearn.svm import NuSVC
start = time.time()
print('\nSVM: NuSVC: One-vs-One : labels not encoded \n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=69)
# One-vs-Rest
clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    NuSVC(kernel='rbf',random_state=69, tol=1e-5,verbose=0))

clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('NuSVC test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

ypred = clf.predict(Xtest)
cfm = confusion_matrix(ytest, ypred, normalize='true')
plt.figure(figsize = (10,7))
plt.title('NuSVC Confusion Matrix')
sn.heatmap(cfm)

#%% 

from sklearn.tree import DecisionTreeClassifier
start = time.time()
print('\nDecision Tree Classifier One-vs-One : Encoded labels\n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, yy, test_size=0.2, random_state=69)
# One-vs-Rest
clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    DecisionTreeClassifier(random_state=69))

clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('Decision Treee test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

ypred = clf.predict(Xtest)
cfm = confusion_matrix(ytest.argmax(axis=1), ypred.argmax(axis=1), normalize='pred')
plt.figure(figsize = (10,7))
plt.title('Decision Tree Confusion Matrix')
sn.heatmap(cfm)

#%%  Extra Tree Classifier

from sklearn.tree import ExtraTreeClassifier
start = time.time()
print('\nExtra Tree Classifier One-vs-One : Encoded labels\n')

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, yy, test_size=0.2, random_state=69)
# One-vs-Rest
clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    ExtraTreeClassifier(random_state=69))

clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('Extra Tree test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

ypred = clf.predict(Xtest)
cfm = confusion_matrix(ytest.argmax(axis=1), ypred.argmax(axis=1), normalize='pred')
plt.figure(figsize = (10,7))
plt.title('Extra Tree Confusion Matrix')
sn.heatmap(cfm)


#%%

from sklearn.neighbors import KNeighborsClassifier
start = time.time()


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, yy, test_size=0.2, random_state=69)
# One-vs-Rest
neighbors = 1

print('\n KNeighbors Classifier : Encoded labels\n')

clf = make_pipeline(StandardScaler(),
                    # KMeans(n_clusters=50,random_state=69),
                    KNeighborsClassifier(n_neighbors=neighbors)
                    )

clf.fit(Xtrain,ytrain)

score = clf.score(Xtest,ytest)
print('KNeighbors test score= ', score)
end = time.time()
print('\n\nDone in %.2f seconds.\n\n' % (end-start))

ypred = clf.predict(Xtest)
cfm = confusion_matrix(ytest.argmax(axis=1), ypred.argmax(axis=1), normalize='pred')
plt.figure(figsize = (10,7))
plt.title('KNeighbors Classifier Confusion Matrix, n_neighbors = %.0f' %neighbors)
sn.heatmap(cfm)
#%%

print('\ndone\n')


#%%
print(y)