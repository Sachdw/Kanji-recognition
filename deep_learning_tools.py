
import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from sklearn import metrics


#Class to implement TensorFlow's Local Response Normalization as a Keras Layer
class LocalResponseNormalization(keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        #Define the main layers as Tensorflow's Local Response Normalization 
        self.main_layers = [keras.layers.Lambda(tf.nn.local_response_normalization)]
        
        #Define the calling function
        def call(self, inputs):
            #Send inputs to the main layers
            Z = inputs
            output = self.main_layers(Z)
            #Return Input data after Normalization
            return output
        
        
        
#Class to implement an Inception Module
class InceptionModule(keras.layers.Layer):
    def __init__(self,filters,activation="relu",**kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        
        #Get the filters
        f1,f2,f2_2,f3,f3_2,f4 = filters
        
        #Define Path one
        self.path_1 = [
            keras.layers.Conv2D(f1,kernel_size=(1,1),activation = self.activation, strides=1,padding="same")
        ]
        
        #Define path two (with a 3x3 kernel)
        self.path_2 = [
            keras.layers.Conv2D(f2,kernel_size=(1,1),activation = self.activation, strides=1, padding="same"),
            keras.layers.Conv2D(f2_2,kernel_size=(3,3),activation = self.activation, strides=1, padding="same")
        ]
        
        #Define path three (with a 5x5 kernel)
        self.path_3 = [
            keras.layers.Conv2D(f3,kernel_size=(1,1),activation = self.activation, strides=1, padding="same"),
            keras.layers.Conv2D(f3_2,kernel_size=(5,5),activation = self.activation, strides=1, padding="same")
        ]
        
        #Define the Maxpooling path
        self.path_4 = [
            keras.layers.MaxPooling2D(pool_size=(3, 3),strides=1,padding="same"),
            keras.layers.Conv2D(f4,kernel_size=(1,1),activation = self.activation, strides=1, padding="same")
        ]
 
    #Define the calling function for the layer
    def call(self, inputs):
        #Copy the inputs for the 4 paths
        p_1 = inputs
        p_2 = inputs
        p_3 = inputs
        p_4 = inputs
        
        #Send the inputs to each path
        for layer in self.path_1:
            p_1 = layer(p_1)
        for layer in self.path_2:
            p_2 = layer(p_2)
        for layer in self.path_3:
            p_3 = layer(p_3)
        for layer in self.path_4:
            p_4 = layer(p_4)
        
        #Concatanate the results of each path and output
        output = tf.concat([p_1,p_2,p_3,p_4],axis = 3)
        return output
    





#Class to implement a Residual Unit for the ResNet34 CNN Architecture       
class ResidualUnit34(keras.layers.Layer):
    def __init__(self,filters,strides = 1, activation="relu",**kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        
        #Define the main layers of the Residual Unit (2 Convolutional layers + normalization and Relu 
        self.main_layers = [
            keras.layers.Conv2D(filters,kernel_size = (3,3), strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters,kernel_size = (3,3),strides=1,padding="same",use_bias=False),
            keras.layers.BatchNormalization()]
        
        #Define the shortcut path
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, kernel_size = (1,1),strides=strides,padding="same",use_bias=False),
                keras.layers.BatchNormalization()]
        
        #Define the calling function for the Residual Unit
        def call(self, inputs):
            #Send inputs to the main layers
            Z = inputs
            for layer in self.main_layers:
                Z = layer(Z)
            #Send inputs to the shortcut path
            skip_Z = inputs
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)
            #Join the results of the main and shortcut path
            return self.activation(Z + skip_Z)
        
        
        
#Class to implement a Residual Unit for the ResNet50 CNN Architecture       
class ResidualUnit50(keras.layers.Layer):
    def __init__(self,filters,strides = 1, activation="relu",**kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        
        #Define the main layers of the Residual Unit (3 convolutional layers + normalization and Relu)
        self.main_layers = [
            keras.layers.Conv2D(filters,kernel_size = (1,1), strides=strides, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters,kernel_size = (3,3),strides=1,padding="same",use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters*4,kernel_size = (1,1),strides=1,padding="same",use_bias=False),
            keras.layers.BatchNormalization()]
        
        #Define the shortcut path of the Residual Unit
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters*4,kernel_size=(1,1), strides=strides, padding="same", use_bias=False),
                keras.layers.BatchNormalization()]
        
        #Define the calling function for the Residual Unit
        def call(self, inputs):
            #Send the input to the main layers
            Z = inputs
            for layer in self.main_layers:
                Z = layer(Z)
            #Copy the input and send to the shortcut layer
            skip_Z = inputs
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)
            #Join the result of the main path and shortcut path
            return self.activation(Z + skip_Z)
        

def evaluate_model(model,x_train, Y_train, x_test, Y_test):
    
    """
    Evaluates a CNN model on a training and test set and prints the results.
    Takes a model and the datasets as inputs and returns the f1-score.
    
    Can process keras.engine.sequential.Sequential objects and 
    can load pretrained models as .h5 files.
    
    """
    
    #Load the model
    if type(model) != keras.engine.sequential.Sequential:
        model = load_model(model)

    #Evaluate the model on the training and test data
    train_score = model.evaluate(x_train, Y_train, verbose=1)
    test_score = model.evaluate(x_test, Y_test, verbose=1)
    print(f'\nResults')
    print('\nTrain loss:', round(train_score[0],2))
    print('Train accuracy:', round(train_score[1],2))
    print('\nTest loss:', round(test_score[0],2))
    print('Test accuracy:', round(test_score[1],20))
        
    #Calculate F1-score
    rounded_labels = np.argmax(Y_test, axis=1)
    y_pred = model.predict_classes(x_test, verbose=0)
    f1 = metrics.f1_score(rounded_labels, y_pred,average='micro')
    print(f'F1 micro score: {f1}')
    
    return f1
    