from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np

class ASLEEP:
    def __init__(self, *, settings, num_entities, num_relations):
        self.settings = settings

        
        input_head = Input(shape=(1,), dtype='int32', name='input_head')
        input_tail = Input(shape=(1,), dtype='int32', name='input_tail')

        embedding_layer = Embedding(input_dim=num_entities, output_dim=self.settings['embedding_dim'],
                                    input_length=1, activity_regularizer=l2(self.settings['reg']))


        head_embedding_e = embedding_layer(input_head)
        head_embedding_drop = Dropout(self.settings['input_dropout'])(head_embedding_e)
        head_embedding_dense = Dense(self.settings['embedding_dim'] * self.settings['hidden_width_rate'],
                                     activity_regularizer=l2(self.settings['reg']))(head_embedding_drop)
        head_embedding_d = Dropout(self.settings['hidden_dropout'])(head_embedding_dense)

        tail_embedding_e = embedding_layer(input_tail)
        tail_embedding_drop = Dropout(self.settings['input_dropout'])(tail_embedding_e)
        tail_embedding_dense = Dense(self.settings['embedding_dim'] * self.settings['hidden_width_rate'],
                                     activity_regularizer=l2(self.settings['reg']))(tail_embedding_drop)
        tail_embedding_d = Dropout(self.settings['hidden_dropout'])(tail_embedding_dense)

        
        combined = Maximum()([head_embedding_d, tail_embedding_d])
        final_f = Flatten()(combined)
        final_d_relu = Dense(self.settings['embedding_dim'] * self.settings['hidden_width_rate'],
                             activation="relu")(final_f)
        final_d = Dense(num_relations, activation="sigmoid")(final_d_relu)

        self.model = Model(inputs=[input_head, input_tail], outputs=final_d)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

        
    def fit(self,X, y):
        X_Head = np.array(X[:, 0])
        X_Tail = np.array(X[:, 1])
        len_X_Head = len(X_Head)
        len_X_Tail = len(X_Tail)
        X_Head = np.reshape(X_Head, (len_X_Head,1))
        X_Tail = np.reshape(X_Tail, (len_X_Tail,1))
        self.model.fit([X_Head, X_Tail], y, batch_size=self.settings['batch_size'], epochs=self.settings['epochs'],
                       use_multiprocessing=True, verbose=1, shuffle=True)


    def predict(self, X):
        X_Head = np.array(X[:, 0])
        X_Tail = np.array(X[:, 1])
        len_X_Head = len(X_Head)
        len_X_Tail = len(X_Tail)
        X_Head = np.reshape(X_Head, (len_X_Head, 1))
        X_Tail = np.reshape(X_Tail, (len_X_Tail, 1))
        return self.model.predict([X_Head, X_Tail])
  
