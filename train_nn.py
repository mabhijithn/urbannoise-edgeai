import tensorflow as tf
import numpy as np
import librosa
import os.path as path
import glob
import pandas as pd
import h5py
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm


def build_nnmodel(n_feats=193):
    inputs = keras.Input(shape=(n_feats,))
    
    x = layers.Dense(128, activation = 'relu')(inputs)
    
    x = layers.Dense(64, activation = 'relu')(x)
    
    outputs = layers.Dense(10, activation = 'sigmoid')(x)
    
    model = keras.Model(inputs,outputs)
    
    model.summary()
    
    return model

def setup_tf():
    """
    Detects GPUs and (currently) sets automatic memory growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu,True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)


def extract_features(filename):
    features = np.empty((0,193))
    X, sample_rate = librosa.load(filename)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = np.vstack([features,ext_features])
    return features

# Save all the features from the dataset (Only do it once!)
save_feats = 'UrbanSounds8k_feats_allfolds.h5'

if path.isfile(save_feats) is False:
    # Checks if the saved features exist, else computes and saves features in HD5 dataset
    warnings.filterwarnings("ignore")
    labels = []
    foldnos = []
    filenames = []
    classes = []
    fullfilenames = []
    for fold in range(11):
        fldr = path.join(urbansndfldr,f'fold{fold}')
        wavfiles = glob.glob(path.join(fldr,'*.wav'))
        for count,item in enumerate(wavfiles):
            fname = item.split('/')[-1]
            idx = (labelinfo['slice_file_name']==fname)
            foldnos.append(labelinfo.loc[idx,'fold'].values[0])
            labels.append(labelinfo.loc[idx,'classID'].values[0])
            classes.append(labelinfo.loc[idx,'class'].values[0])
            filenames.append(fname)
            fullfilenames.append(item)
    
    n_feats = 193
    dt_fl = h5py.vlen_dtype(np.float32)
    dt_int = h5py.vlen_dtype(np.uint8)
    dt_str = h5py.special_dtype(vlen=str)


    print('Saving features to a dataset...')
    with h5py.File(path.join(datafldr,save_feats),'w') as f:
        dset_feats = f.create_dataset('feats',shape=(len(fullfilenames),n_feats),dtype=np.float32)
        dset_labels = f.create_dataset('label',shape=(len(fullfilenames),),dtype=np.uint8)
        dset_fold = f.create_dataset('fold',shape=(len(fullfilenames),),dtype=np.uint8)
        dset_filenames = f.create_dataset('filename',shape=(len(fullfilenames),),dtype=dt_str)
        dset_classes = f.create_dataset('class',shape=(len(fullfilenames),),dtype=dt_str)
        for i in tqdm(range(len(fullfilenames))):
            dset_feats[i] = feats[i]
            dset_labels[i] = labels[i]
            dset_fold[i] = foldnos[i]
            dset_classes[i] = classes[i]
            dset_filenames[i] = filenames[i]       

    
# Load the saved features from the HD5 dataset
with h5py.File(path.join(datafldr,save_feats),'r') as f:
    feats = np.asarray(f['feats'],dtype=np.float16)
    labels = np.asarray(f['label'])
    foldno = np.asarray(f['fold'])
    lbl_desc = np.asarray(f['class'])

# Split the data into train, validation and test set (70-20-10 split)
X_train, X_test, y_train, y_test = train_test_split(feats,labels, test_size=0.1, stratify=labels, random_state=30)
X,Xval,Y,Yval = train_test_split(X_train,y_train,test_size=0.2,random_state=42,stratify=y_train)

setup_tf()
BATCH_SIZE = 256
model_path = path.join('nn_urban8k_model.h5')

# Build and compile the Neural-network model
nn_urban8k = build_nnmodel()
nn_urban8k.compile(optimizer="adam",loss='sparse_categorical_crossentropy', metrics="accuracy")

# Train the model
best_loss = 1e10
n_epochs = 100
val_losses = np.zeros((n_epochs,))
nn_urban8k.fit(X,Y,batch_size=256, epochs=n_epochs)
y_pred = nn_urban8k.predict(Xval)
val_loss, val_acc = nn_urban8k.evaluate(x=Xval,y=Yval,batch_size=BATCH_SIZE) # TODO
print(f'\nValidation Set - Accuracy: {val_acc}, Loss: {val_loss} ')

# Save the model. This model needs to be converted to tensorflow lite to be executed on the raspberry pi
cnn_urban8k.save(model_path)