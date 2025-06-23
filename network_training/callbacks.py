from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

class Callbacks:

    def __init__(self, filepath, patience = 15):

        self.model_checkpoints = ModelCheckpoint (
            filepath = filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only= False,
            mode = 'auto')

        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            mode = 'auto', 
            verbose =1, 
            patience =patience)