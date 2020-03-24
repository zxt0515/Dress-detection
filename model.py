import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tools import *

class dress_detection():
    def __init__(
        self,
        network = [],
        hyperpara = []):
        self.model = network
        self.hyperpara = hyperpara
    
    def train(
        self, 
        x_train, 
        y_train, 
        model_save_name = []):
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath = model_save_name,
        #     save_weights_only = True)
        self.model.compile(
            optimizer=Adam(learning_rate = self.hyperpara.lr), 
            loss="binary_crossentropy",
            metrics=["accuracy"])
        history_callback = self.model.fit(
            x_train, 
            y_train,
            epochs = self.hyperpara.epoch,
            batch_size = self.hyperpara.batch_size,
            verbose = 2)
        model_json = self.model.to_json()
        with open(self.hyperpara.model_saved,"w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.hyperpara.weight_saved)
        loss_history = np.array(history_callback.history["loss"])
        plt.plot(loss_history)
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.savefig(self.hyperpara.output_dir+"training_loss.png")
        plt.close()
    
    def predict(
        self,
        x_test,
        y_test,
        model_load_name = []):
        # evaluate the trained model by the test dataset
        with open(self.hyperpara.model_saved) as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(self.hyperpara.weight_saved)
        model.compile(
            optimizer=Adam(learning_rate = self.hyperpara.lr), 
            loss="binary_crossentropy",
            metrics=["accuracy"])
        loss, acc = model.evaluate(x_test, y_test, verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100*acc))

    def present(
        self,
        t,
        x_test,
        model_load_name = [],
        json_image_save_name = []):
        #load model 
        with open(self.hyperpara.model_saved) as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(self.hyperpara.weight_saved)
        model.compile(
            optimizer=Adam(learning_rate = self.hyperpara.lr), 
            loss="binary_crossentropy",
            metrics=["accuracy"])
        y_predict = model.predict(x_test)
        data = np.concatenate((t.reshape((-1,1)), y_predict.reshape((-1,1))),axis=1)
        data = {"getting dressed": data}
        with open(self.hyperpara.present_dir+json_image_save_name+".json", "w") as write_file:
            json.dump(data, write_file, cls=NumpyEncoder)
        plt.plot(t,y_predict)
        plt.xlabel("Time")
        plt.ylabel("Probability of getting dressed")
        plt.savefig(self.hyperpara.present_dir+json_image_save_name+".png")
        plt.close()
        return y_predict