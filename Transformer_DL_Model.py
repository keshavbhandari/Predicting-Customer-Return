# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 22:21:28 2020

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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        
    def get_config(self):
        cfg = super().get_config()
        return cfg  

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.SpatialDropout1D(rate)
        self.dropout2 = layers.SpatialDropout1D(rate)
        
    def get_config(self):
        cfg = super().get_config()
        return cfg 

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):
        cfg = super().get_config()
        return cfg 

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    

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
pickle.dump(pickle_byte_obj, open('C:/My Desktop/Projects/'+"data_config.pkl", "wb"))

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

        input_cat_cols = layers.Input(shape=(max_length[col],))
        inputs.append(input_cat_cols)

        vocab_size = len(word_index[col]) + 1
        embed_size = int(np.min([np.ceil((vocab_size)/2), 30]))
        embedding = TokenAndPositionEmbedding(max_length[col], vocab_size, embed_size)(input_cat_cols)
        transformer_block = TransformerBlock(embed_size, num_heads=1, ff_dim=32, rate=0.8)(embedding)
        transformer_block = layers.Flatten()(transformer_block)
        embeddings_1.append(transformer_block)

#    if continuous_sequence:
#        input_array = layers.Input(shape=(max_sequence_length,3))
#        inputs.append(input_array)
#        RNN = Bidirectional(LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(input_array)
#        RNN = Bidirectional(LSTM(max_length[col], dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(RNN)
#        RNN = layers.TimeDistributed(Dense(50, activation='relu'))(RNN)
#        RNN = layers.Flatten()(RNN)
##        RNN=Attention()(RNN)
#        embeddings_1.append(RNN)
    
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

#    x1 = layers.Concatenate()(embeddings_1)

#    bn1 = BatchNormalization()(x1)
#    x1 = Dense(500, activation='relu')(bn1)
#    x1 = Dropout(.4)(x1)
#    bn1 = BatchNormalization()(x1)
#    x1 = Dense(300, activation='relu')(bn1)
#    x1 = Dropout(.3)(x1)
#    bn1 = BatchNormalization()(x1)
#    x1 = Dense(50, activation='relu')(bn1)
    x = layers.Dense(20, activation="relu")(transformer_block)
    output1 = layers.Dense(final_layer, activation='sigmoid')(x)   
 

    model = keras.Model(inputs, output1)
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=['accuracy'])
    return model


model = create_model(cat_cols, num_cols, word_index, max_length, Total_Dependent_Variables, continuous_sequence, max_sequence_length)
print(model.summary())
reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.6, patience=1, min_lr=1e-5,  verbose=1, mode = 'min')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0001, verbose=1, patience=3, restore_best_weights = True)
filepath="C:/My Desktop/Projects/model_checkpoint.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [reduce_learning_rate, early_stopping]

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
entity_embedding_model = model.fit(input_list_learn,y_train, 
                         validation_data=(input_list_validation,y_val), 
                         epochs=epochs, #25
                         callbacks=callbacks_list, 
                         shuffle=True,#False 
                         batch_size=batch_size, #2056
                         verbose=1)

#entity_embedding_model.model.save("C:/My Desktop/Projects/keras-model.h5")
entity_embedding_model.model.save_weights('C:/My Desktop/Projects/keras-weights', save_format='tf')
#new_model = load_model("C:/My Desktop/Projects/keras-model.h5", custom_objects={'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'MultiHeadSelfAttention': MultiHeadSelfAttention, 'TransformerBlock': TransformerBlock, 'focal_loss': focal_loss})   
new_model = create_model(cat_cols, num_cols, word_index, max_length, Total_Dependent_Variables, continuous_sequence, max_sequence_length)
new_model.load_weights('C:/My Desktop/Projects/keras-weights')

print("Scoring")
predictions = new_model.predict(input_list_learn,batch_size=4096)
learn_preds = pd.DataFrame(predictions, columns = dependent_variables)
learn_preds['Customer ID'] = learn['Customer ID'].reset_index(drop=True)

predictions = new_model.predict(input_list_validation, batch_size=4096)
validation_preds = pd.DataFrame(predictions, columns = dependent_variables)
validation_preds['Customer ID'] = validation['Customer ID'].reset_index(drop=True)

print("Learn AUC:",roc_auc_score(y_train, learn_preds.iloc[:,0]))
print("Validation AUC:",roc_auc_score(y_val, validation_preds.iloc[:,0]))





vocab_size = len(word_index['categorical_sequence_items']) + 1
embed_size = 32 #int(np.min([np.ceil((vocab_size)/2), 30]))

inputs = layers.Input(shape=(max_length['categorical_sequence_items'],))
embedding_layer = TokenAndPositionEmbedding(max_length['categorical_sequence_items'], vocab_size, embed_size)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_size, 2, 32)
x = transformer_block(x)
#x = layers.GlobalAveragePooling1D()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(Total_Dependent_Variables, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    input_list_learn[0], y_train, batch_size=32, epochs=2, validation_data=(input_list_validation[0], y_val)
)



