import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Conv2D, TimeDistributed, Activation, MaxPool2D, Flatten, Dropout
from tensorflow.keras.layers import Dense
class dressnet():
    def __init__(
        self,
        hyperpara = []):
        self.num_time_steps = hyperpara.fps_standard*hyperpara.time_window
        self.CNN_LSTM()
    def CNN_LSTM(self):
        self.model = keras.Sequential()
        # define CNN model
        self.model.add(TimeDistributed(Conv2D(16, (2,2), padding='same'), input_shape=(self.num_time_steps,90, 160, 3)))
        self.model.add(TimeDistributed(Activation("relu")))
        self.model.add(TimeDistributed(Conv2D(16, (2,2))))
        self.model.add(TimeDistributed(Activation("relu")))
        self.model.add(TimeDistributed(MaxPool2D(pool_size=(2, 2))))
        self.model.add(TimeDistributed(Dropout(0.25)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(TimeDistributed(Dense(10)))
        # Define LSTM model
        self.model.add(LSTM(20, return_sequences=False))
        self.model.add(Dense(1))
        self.model.add(Activation("sigmoid"))
        print(self.model.summary())

# if __name__=="__main__":
#     network = dressnet()