
import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf

""" Weight initialization for Neural Network Layers"""
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

""" Bias initialization """
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def output(input,w,b):
    return tf.matmul(input,w)+b

""" Neural Network Initialization """
def FitModel(x_train,y_train):
    # Number of features
    x_columns = 8

    # Number of output values
    y_columns = 2

    # Number of neurons in Hidden Layer(Hit and Trial) 
    layer1_num = 7

    # Number of epochs(iterations of forward and backward passes)
    epoch_num = 10

    # Number of batches to be run in an epoch
    train_num = 1000

    # Batch Size for a particular training set
    batch_size =100   #226200

    # AVriable for checking progress
    display_size = 1

    # Placeholders for inputs and outputs
    x = tf.placeholder(tf.float32,[None,x_columns])
    y = tf.placeholder(tf.int32,[None])

    # Hidden layer taking input from x
    layer1 = tf.nn.relu(output(x,weight([x_columns,layer1_num]),bias([layer1_num])))
    
    #Output Layer taking input from layer1
    prediction = output(layer1,weight([layer1_num,y_columns]),bias([y_columns]))

    # Definition of loss function
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    
    # Choice of Optimizer
    train_step = tf.train.AdamOptimizer().minimize(loss)

    # Tensorflow Interactive
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Start Training
    for epoch in range(epoch_num):
        avg_loss = 0.
        for i in range(train_num):
            index = np.random.choice(len(x_train),batch_size)
            x_train_batch = x_train[index]
            y_train_batch = y_train[index]
            _,c = sess.run([train_step,loss],feed_dict={x:x_train_batch,y:y_train_batch})
            avg_loss += c/train_num
        if epoch % display_size == 0:
            print("Epoch:{0},Loss:{1}".format(epoch+1,avg_loss))

    print("Training Finished")

def main():
    # Load feature set
    x_train = np.loadtxt('sampledata.txt', delimiter=' ')
    
    # Remove attribute1[uid] not required as it has no relation to output
    x_train = x_train[:,1:9]

    # Load output labels for respective feature set
    y_train = np.loadtxt('output.txt',delimiter='\n')

    print x_train.shape

    # Call the Neural Network logic
    FitModel(x_train,y_train)
         
if __name__== "__main__":
      main()