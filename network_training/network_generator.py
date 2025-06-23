import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from keras import initializers
from keras.models import load_model

import datetime, os
from sklearn.metrics import  classification_report, confusion_matrix, log_loss, accuracy_score, roc_auc_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

class NetworkGenerator:

    def __init__(self, data_generator , callbacks, input_dims = 11,  prediction_timesteps = 1, output_dims = 9,
                lr = 1e-2, momentum = 0.9, nesterov = False, batch_size = 2048, history_length = 1, data_split = True):
        self.callbacks = callbacks
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.history_length = history_length
        self.prediction_timesteps = prediction_timesteps
        self.data_gen = data_generator
        self.data_gen.data_split(training = 0.7, testing = 0.15, timestep_shift = self.history_length, validating = 0.15) if data_split == True else None
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.batch_size = batch_size
        self.model = None

    def register_network(self, model):
        self.model = model

    def construct_network(self, train_data_generator):
        x_batch, y_batch = next(train_data_generator)
        self.model = keras.Sequential([keras.layers.Flatten(input_shape = (self.history_length, x_batch.shape[-1])),
                keras.layers.Dense(2*self.input_dims+1, activation = 'tanh',  kernel_initializer='he_normal'),
                keras.layers.Dropout(0.15),
                keras.layers.Dense(2*self.input_dims+1, activation = 'tanh',  kernel_initializer='he_normal'),
                keras.layers.Dropout(0.15),
                keras.layers.Dense(self.output_dims, activation = 'softmax')]
        )
        print(self.model.summary())
    
    def train_network(self, callback_list = None, 
                        epochs = 400,
                        validation_steps = 1, validation_data = None, train_data = None, steps_per_epoch = None):
        if callback_list == None:
            callback_list = [self.callbacks.early_stopping, self.callbacks.model_checkpoints]
        return self.model.fit(x = train_data,
                                epochs = epochs,
                                steps_per_epoch = ((len(self.data_gen.train_data_x) - self.history_length) // self.batch_size) if steps_per_epoch == None else steps_per_epoch,
                                validation_data = validation_data,
                                validation_steps = validation_steps,
                                callbacks = callback_list)

    def batch_generator(self, batch_size, sequence_length, num_x_signals, num_y_signals, min_index,
                     x_data, y_data, random = True, all_data = False, classification = True):
        if all_data:
            while True:
                #determine the length of the data:
                length = 0
                for station in x_data:
                    length += (len(station)-sequence_length)
                pos_ctr = 0
                #empty array for feature batch
                x_batch = np.zeros(shape = (length, sequence_length, num_x_signals))
                #empty arras for target batch
                y_batch = np.zeros(shape = (length, num_y_signals))
                ##empty array for dim of sequence length
                x_sequ = np.zeros(shape = (sequence_length, num_x_signals))
                for z in range(len(x_data)):
                    for idx in range(len(x_data[z])-sequence_length):
                        idx = idx + sequence_length
                        for rng in range(min_index, sequence_length):
                            idx_seq = idx - rng
                            x_sequ[rng] = x_data[z].iloc[idx_seq]
                        x_batch[pos_ctr] = x_sequ
                        if classification:
                            y_batch[pos_ctr][int(y_data[z].iloc[idx])] = 1
                        else:
                            y_batch[pos_ctr] = y_data[z].iloc[idx]
                        pos_ctr += 1
                yield(x_batch, y_batch) 
        else:
            while True:
                #empty array for feature batch
                x_batch = np.zeros(shape = (batch_size, sequence_length, num_x_signals))
                #empty arras for target batch
                y_batch = np.zeros(shape = (batch_size, num_y_signals))
                ##empty array for dim of sequence length
                x_sequ = np.zeros(shape = (sequence_length, num_x_signals))
                #select random data
                if random:
                    batch_ctr = 0
                    while (batch_size-1 >= batch_ctr):
                        for z in range(len(x_data)):
                            idx = np.random.randint(min_index+sequence_length, len(x_data[z]))
                            for rng in range(min_index, sequence_length):
                                idx_seq = idx - rng
                                x_sequ[rng] = x_data[z].iloc[idx_seq]
                            x_batch[batch_ctr] = x_sequ
                            if classification:
                                y_batch[batch_ctr][int(y_data[z][idx])] = 1
                            else:
                                y_batch[batch_ctr] = y_data[z][idx]
                            batch_ctr += 1
                            if batch_ctr == (batch_size):
                                break
                else:
                    #create a custom idx for each data source, so that the data 
                    # sources can have different sizes 
                    idx_list = []
                    for z in range(len(x_data)):
                        idx_list.append(min_index+sequence_length)
                    batch_ctr = 0
                    while (batch_size-1 >= batch_ctr):
                        for z in range(len(x_data)):
                            if (idx_list[z] >= len(x_data[z])):
                                idx_list[z] = min_index+sequence_length
                            for rng in range(min_index, sequence_length):
                                idx_seq = idx_list[z] - rng
                                x_sequ[rng] = x_data[z].iloc[idx_seq]
                            x_batch[batch_ctr] = x_sequ
                            if classification:
                                y_batch[batch_ctr][int(y_data[z][idx_list[z]])] = 1
                            else:
                                y_batch[batch_ctr] = y_data[z][idx_list[z]]
                            idx_list[z]+=1
                            batch_ctr += 1
                            if batch_ctr == (batch_size):
                                break
                yield(x_batch, y_batch) 

    def print_training(self, history):
        #Definition der y Werte für die Abbildungen des Trainings/Validierungs Verlustes/Akkuratheit
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history['categorical_accuracy']
        val_accuracy = history.history['val_categorical_accuracy']
        epochs = range(len(loss))

        plt.figure(figsize=(15,5))
        plt.plot(epochs, accuracy, c='black', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, c='green', label='Validation Accuracy')
        plt.plot(epochs, loss, c='red', label='Trainingloss')
        plt.plot(epochs, val_loss, c='blue', label='Validationloss')
        plt.title('Training und Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Categorical Accuracy')
        plt.legend()
    
    def test_predictions(self, model, x_batch, y_batch, print_results = False):
        model.summary()
        loss, acc = model.evaluate(x_batch, y_batch)
        y_probs = model.predict(x_batch)
        y_pred = np.argmax(y_probs, axis=1)
        if print_results:
            confusion=confusion_matrix(np.argmax(y_batch, axis=1), y_pred)
            #Berechnung der Akkuratheit des FFN 
            accuracy=accuracy_score(np.argmax(y_batch, axis=1), y_pred)
            #Berechnung Klassifikationsreport
            report=classification_report(np.argmax(y_batch, axis=1), y_pred)

            print('Confusion Matrix')
            print(confusion)
            print('Klassifikations Report')
            print(accuracy)
            print('Akkuratheit der Klassifikation')
            print(report)
