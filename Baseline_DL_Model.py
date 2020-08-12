# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:19:34 2020

@author: kbhandari
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import pickle
import collections
import boto3
import math
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Concatenate
from keras.layers import GRU
from keras.layers import TimeDistributed
from keras.models import Model
from keras.layers import SpatialDropout1D
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Bidirectional
from keras.layers import Average
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,Callback
from sklearn.model_selection import KFold
from keras.models import load_model, model_from_json
from keras.regularizers import l2
#from tensorflow.keras import backend as K
from keras import backend as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer

wd = "C:/Users/kbhandari/OneDrive - Epsilon/Desktop/Senior ML Test/Baseline Model/"
os.chdir(wd)
raw_data = pd.read_csv(wd + "Data.csv")

DV_WINDOW_START = '2011-10-08'
DV_WINDOW_END = '2011-11-08'
raw_data['StockCode'] = raw_data['StockCode'].str.extract('(\d+)', expand=False).astype(str)
raw_data[['Quantity','Price']] = raw_data[['Quantity','Price']].astype(str)
actuals = raw_data.groupby('Customer ID').agg({'InvoiceDate': max}).reset_index(drop=False)
actuals['DV'] = np.where(actuals['InvoiceDate']>=DV_WINDOW_START, 1, 0)
actuals.drop('InvoiceDate', axis=1, inplace=True)
raw_data = raw_data[raw_data.InvoiceDate < DV_WINDOW_START]

raw_data['Days_Between_TXN'] = pd.to_datetime(raw_data['InvoiceDate']).sub(pd.to_datetime(DV_WINDOW_START)).dt.days.astype(str)
agg_df = raw_data.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID']).agg({
        'StockCode': lambda s: ', '.join(s),
        'Quantity': lambda s: ', '.join(s),
        'Price': lambda s: ', '.join(s),
        'Days_Between_TXN': lambda s: ', '.join(s)
        }).reset_index()

data = pd.merge(actuals, agg_df, how='inner', on='Customer ID')

data.columns = ['Customer ID', 'DV', 'categorical_sequence_items', 'continuous_sequence_quantity', 'continuous_sequence_price', 'continuous_sequence_days_between_txn']

max_sequence_length=120 
batch_size = 32 #int(0.005*len(learn)) #0.005 #2048
epochs = 20

def replace_recency(data):
            columns = data.columns
            cols_to_replace = {}
            for col in columns:
                if 'continuous_sequence' in col.lower() or 'categorical_sequence' in col.lower():
                    cols_to_replace[col] = ''
                else:
                    cols_to_replace[col] = -1 #0
            data = data.fillna(cols_to_replace)
            return data

data = replace_recency(data)
missing = data.isna().sum()

# Continuous Sequence
def get_continuous_sequence(data, max_sequence_length):
    combined_arrays = None    
    first_time = True
    for col in data.columns:    
        if 'continuous_sequence' in col.lower():
            print(col)
            array = data[col].str.split(",", n = max_sequence_length, expand = True).replace('',np.nan).fillna(value=np.nan).fillna(0)
            if array.shape[1]>max_sequence_length:
                array = array.iloc[:,0:max_sequence_length]
            elif array.shape[1]<max_sequence_length:
                cols_to_add = max_sequence_length-array.shape[1]
                rows_to_add = array.shape[0]
                df = pd.DataFrame(np.zeros((rows_to_add,cols_to_add)))
                array = pd.concat([array, df], axis=1)
            array = np.array(array.astype(np.float))
            array = array.reshape(array.shape[0],array.shape[1],1)
            if first_time:
                combined_arrays = array
                first_time = False
            else:
                combined_arrays = np.concatenate((combined_arrays, array), -1)
    return combined_arrays
     
def get_max_length(list_of_list):
    max_value = 0
    for l in list_of_list:
        if len(l) > max_value:
            max_value = len(l)
    return max_value
        
def get_padding(data, categorical_cols, tokenizer = None, max_padding_length = None, max_seq_length=max_sequence_length):
    # Tokenize Sentences
    word_index, max_length, padded_docs, word_tokenizer = {},{},{},{}
    if tokenizer is None:    
        for col in categorical_cols:            
            print("Processing column:", col)        
            t = Tokenizer()
            t.fit_on_texts(data[col].astype(str))
            word_index[col] = t.word_index
            txt_to_seq = t.texts_to_sequences(data[col].astype(str))
            max_length_value = get_max_length(txt_to_seq)
            max_length[col] = max_length_value if max_length_value < max_seq_length else max_seq_length
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_length[col], padding='post')
            word_tokenizer[col] = t
        return word_index, max_length, padded_docs, word_tokenizer

    else:
        for col in categorical_cols:
            print("Processing column:", col)
            t = tokenizer[col]
            txt_to_seq = t.texts_to_sequences(data[col].astype(str))
            padded_docs[col] = pad_sequences(txt_to_seq, maxlen=max_padding_length[col], padding='post')        
        return padded_docs       

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:
        
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 #shape=(input_shape[-1], input_shape[1]),
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     #shape=(input_shape[-1],),
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

# Train-Test Split
learn, validation = train_test_split(data, test_size = 0.2, random_state = 0)
del data
gc.collect()

print("Learn Shape:", learn.shape)
print("Validation Shape:", validation.shape)

cat_cols = [col for col in learn.columns if learn[col].dtypes == 'O' and 'continuous_sequence_' not in col]
dependent_variables = [col for col in learn.columns if col not in ['Customer ID'] and '_sequence_' not in col]
Total_Dependent_Variables = len(dependent_variables)
cols_to_exclude = dependent_variables + cat_cols + ['Customer ID']
continuous_cols = [col for col in learn.columns if col not in cols_to_exclude and '_sequence_' not in col]

#data_config = pickle.load(open(wd+"data_config.pkl", "rb"))
#word_index, max_length, tokenizer, cat_cols, continuous_cols, max_sequence_length = data_config[0], data_config[1], data_config[2], data_config[3], data_config[4], data_config[5]
#padded_docs = get_padding(learn, cat_cols, tokenizer, max_length)

word_index, max_length, padded_docs, tokenizer = get_padding(learn, cat_cols, tokenizer = None, max_seq_length = max_sequence_length)
print("Max Length of Categorical Variables: ", max_length)
       
continuous_sequence_learn = get_continuous_sequence(learn, max_sequence_length=max_sequence_length)
continuous_sequence_learn.shape[-1]

pickle_byte_obj = [word_index, max_length, tokenizer, cat_cols, continuous_cols, max_sequence_length]
pickle.dump(pickle_byte_obj, open(wd+"data_config.pkl", "wb"))

padded_docs_validation = get_padding(validation, cat_cols, tokenizer, max_length, max_sequence_length)
continuous_sequence_validation = get_continuous_sequence(validation, max_sequence_length=max_sequence_length)

print("Padding Done")


num_cols = continuous_cols #learn_continuous.columns.to_list()
#             file_path = wd + "model/"

input_list_learn, input_list_validation = [], []
for col in cat_cols:
    input_list_learn.append(padded_docs[col])
    input_list_validation.append(padded_docs_validation[col])
if continuous_sequence_learn is not None:
    continuous_sequence = True
    input_list_learn.append(continuous_sequence_learn)
    input_list_validation.append(continuous_sequence_validation)
else:
    continuous_sequence = False
#input_list_learn.append(learn[continuous_cols])
#input_list_validation.append(validation[continuous_cols])

y_train = learn[dependent_variables].values
y_val = validation[dependent_variables].values            

# Loss Function  
gamma = 2.0
epsilon = K.epsilon()
def focal_loss(y_true, y_pred):
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    FL = K.pow(1-pt, gamma) * CE
    loss = K.sum(FL, axis=1)
    return loss

# Model
def create_model(categorical_cols, numerical_cols, word_index, max_length, final_layer, continuous_sequence, max_sequence_length):

    inputs = []
    embeddings_1 = []

    # Embedding for categorical columns:
    for col in categorical_cols:

        input_cat_cols = Input(shape=(max_length[col],))
        inputs.append(input_cat_cols)

        vocab_size = len(word_index[col]) + 1
        embed_size = int(np.min([np.ceil((vocab_size)/2), 30])) #25
        embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding_1'.format(col), trainable=True)(input_cat_cols)
        embedding = SpatialDropout1D(0.2)(embedding) #0.25
        if max_length[col] > 30:
            embedding = Bidirectional(GRU(max_length[col], return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
            embedding = Bidirectional(GRU(max_length[col], return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
            #embedding = Bidirectional(GRU(embed_size))(embedding)
            embedding=Attention()(embedding)
            #embedding = TimeDistributed(Dense(max_length[col], activation='relu'))(embedding)
            #embedding = Flatten()(embedding)
        else:
            embedding = Reshape(target_shape=(embed_size*max_length[col],))(embedding)
        embeddings_1.append(embedding)

    if continuous_sequence:
        input_array = Input(shape=(max_sequence_length,3))
        inputs.append(input_array)
        RNN = Bidirectional(LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(input_array)
        RNN = Bidirectional(LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(RNN)
#        RNN = TimeDistributed(Dense(50, activation='relu'))(RNN)
#        RNN = Flatten()(RNN)
        RNN=Attention()(RNN)
        embeddings_1.append(RNN)
    
    # Dense layer for continous variables 1
#    input_num_cols = Input(shape=(len(numerical_cols),))
#    inputs.append(input_num_cols)
#
#    bn0 = BatchNormalization()(input_num_cols)
#    numeric = Dense(600, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(bn0) #50
#    numeric = Dropout(.4)(numeric)
#    numeric = Dense(300, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(numeric) #30  
#    numeric = Dropout(.2)(numeric)
#    embeddings_1.append(numeric)

    x1 = Concatenate()(embeddings_1)

#    bn1 = BatchNormalization()(x1)
#    x1 = Dense(500, activation='relu')(bn1)
#    x1 = Dropout(.4)(x1)
#    bn1 = BatchNormalization()(x1)
#    x1 = Dense(300, activation='relu')(bn1)
#    x1 = Dropout(.3)(x1)
#    bn1 = BatchNormalization()(x1)
#    x1 = Dense(50, activation='relu')(bn1)
    output1 = Dense(final_layer, activation='sigmoid', name='model_1')(x1)   
 

    model = Model(inputs, [output1])
    model.compile(loss = [focal_loss], optimizer = "adam", metrics=['accuracy'])
    return model


def create_model2(categorical_cols, numerical_cols, word_index, max_length, final_layer, continuous_sequence, max_sequence_length):

    inputs = []

    # Embedding for categorical columns:
    for col in categorical_cols:

        input_cat_cols = Input(shape=(max_length[col],))
        inputs.append(input_cat_cols)

        vocab_size = len(word_index[col]) + 1
        embed_size = int(np.min([np.ceil((vocab_size)/2), 30])) #25
        embedding = Embedding(vocab_size, embed_size, input_length=max_length[col], name='{0}_embedding_1'.format(col), trainable=True)(input_cat_cols)
#        embeddings_1.append(embedding)

    if continuous_sequence:
        input_array = Input(shape=(max_sequence_length,3))
        inputs.append(input_array)
        
        input_full = Concatenate(axis=2)([embedding, input_array])
        
        RNN = Bidirectional(LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(input_full)
        RNN = Bidirectional(LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(RNN)
        RNN=Attention()(RNN)
        
    output1 = Dense(final_layer, activation='sigmoid', name='model_1')(RNN)   
 

    model = Model(inputs, [output1])
    model.compile(loss = [focal_loss], optimizer = "adam", metrics=['accuracy'])
    return model


model = create_model(cat_cols, num_cols, word_index, max_length, Total_Dependent_Variables, continuous_sequence, max_sequence_length)
print(model.summary())
reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.6, patience=1, min_lr=1e-5,  verbose=1, mode = 'min')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0001, verbose=1, patience=3, restore_best_weights = True)
filepath="model_checkpoint.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [reduce_learning_rate, early_stopping, checkpoint]

# Class Weights
def create_class_weight(labels_dict, mu=0.15, default=False):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    if default == False:
        for key in keys:
            if labels_dict[key] == 0:
                labels_dict[key] = 1
            score = math.log(mu*total/float(labels_dict[key]))
            class_weight[key] = score if score > 1.0 else 1.0
    else:
        for key in keys:
            class_weight[key] = 1.0
        
    return class_weight

#labels_dict = {0: 2813, 1: 78, 2: 2814, 3: 78, 4: 7914, 5: 248, 6: 7914, 7: 248}
labels_dict = dict(pd.Series(np.sum(y_train, axis=0)))
class_weights = create_class_weight(labels_dict, mu=0.3, default=True) # Change to False to use class weights
print("Class Weights: ", class_weights)

print("Building Model")
entity_embedding_model = model.fit(input_list_learn,[y_train], 
                         validation_data=(input_list_validation,[y_val]), 
                         epochs=epochs, #25
                         callbacks=callbacks_list, 
                         shuffle=True,#False 
                         batch_size=batch_size, #2056
                         verbose=1)

architecture = entity_embedding_model.model.to_json()
weights = entity_embedding_model.model.get_weights()

pickle.dump(architecture, open(wd+"keras-model.json", "wb"))
pickle.dump(weights, open(wd+"keras-weights.pkl", "wb"))

loaded_model = pickle.load(open(wd+"keras-model.json", "rb"))
loaded_weights = pickle.load(open(wd+"keras-weights.pkl", "rb"))

use_rnn = np.max(list(max_length.values()))
if use_rnn > 30 or continuous_sequence_learn is not None:    
    new_model = model_from_json(loaded_model, custom_objects={'Attention': Attention, 'focal_loss': focal_loss})
    new_model.set_weights(loaded_weights)
else:
    new_model = model_from_json(loaded_model)
    new_model.set_weights(loaded_weights)      

print("Scoring")
predictions = new_model.predict(input_list_learn,verbose=1, batch_size=4096)
learn_preds = pd.DataFrame(predictions, columns = dependent_variables)
learn_preds['Customer ID'] = learn['Customer ID'].reset_index(drop=True)

predictions = new_model.predict(input_list_validation,verbose=1, batch_size=4096)
validation_preds = pd.DataFrame(predictions, columns = dependent_variables)
validation_preds['Customer ID'] = validation['Customer ID'].reset_index(drop=True)

print("Learn AUC:",roc_auc_score(y_train, learn_preds.iloc[:,0]))
print("Validation AUC:",roc_auc_score(y_val, validation_preds.iloc[:,0]))

# Test Data
raw_data = pd.read_csv(wd + "Data.csv")

raw_data['StockCode'] = raw_data['StockCode'].str.extract('(\d+)', expand=False).astype(str)
raw_data[['Quantity','Price']] = raw_data[['Quantity','Price']].astype(str)
raw_data['Days_Between_TXN'] = pd.to_datetime(raw_data['InvoiceDate']).sub(pd.to_datetime(DV_WINDOW_END)).dt.days.astype(str)
data = raw_data.sort_values('InvoiceDate', ascending=False).groupby(['Customer ID']).agg({
        'StockCode': lambda s: ', '.join(s),
        'Quantity': lambda s: ', '.join(s),
        'Price': lambda s: ', '.join(s),
        'Days_Between_TXN': lambda s: ', '.join(s)
        }).reset_index()

data.columns = ['Customer ID', 'categorical_sequence_items', 'continuous_sequence_quantity', 'continuous_sequence_price', 'continuous_sequence_days_between_txn']
data = replace_recency(data)

padded_docs_data = get_padding(data, cat_cols, tokenizer, max_length, max_sequence_length)
continuous_sequence_data = get_continuous_sequence(data, max_sequence_length=max_sequence_length)
print("Padding Done")

input_list_data = []
for col in cat_cols:
    input_list_data.append(padded_docs_data[col])
if continuous_sequence_data is not None:
    input_list_data.append(continuous_sequence_data)
    
predictions = new_model.predict(input_list_data,verbose=1, batch_size=4096)
data_preds = pd.DataFrame(predictions, columns = dependent_variables)
data_preds['Customer ID'] = data['Customer ID'].reset_index(drop=True)

actual = pd.read_csv(wd+'Actual.csv')
actual.columns = ['Customer ID', 'Actual']
sample_submission = data_preds[['Customer ID', 'DV']]
sample_submission.columns = ['Customer ID', 'Prediction']

sample_submission = pd.merge(sample_submission, actual, how='left', on='Customer ID')
sample_submission = sample_submission[sample_submission['Actual'].notna()]

from sklearn.metrics import roc_auc_score
print(roc_auc_score(sample_submission['Actual'], sample_submission['Prediction']))



