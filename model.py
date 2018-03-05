import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf

def FitModel(a,b):
    # number of neurons in each layer
    input_num_units = 1*8
    hidden_num_units = 80
    output_num_units = 1

    # define placeholders
    x = tf.placeholder(tf.float32, [None, input_num_units])
    y = tf.placeholder(tf.float32, [None, output_num_units])

    # set remaining variables
    epochs = 5
    batch_size = 128
    learning_rate = 0.01

    ### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

    weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units])),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units]))
    }

    biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units])),
    'output': tf.Variable(tf.random_normal([output_num_units]))
    }

    hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)

    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
    # create initialized variables
        sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
        for epoch in range(epochs):
            avg_cost = 0
            total_batch = int(train.shape[0]/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
                avg_cost += c / total_batch
            
            print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
    
        print "\nTraining complete!"
    
    
    # find predictions on val set
        pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)})
    
        predict = tf.argmax(output_layer, 1)
        pred = predict.eval({x: test_x.reshape(-1, input_num_units)})


def main():
    print "Hello World!"
    a = np.loadtxt('sampledata.txt', delimiter=' ')
    a = a[:,1:9]
    b = np.loadtxt('output.txt',delimiter='\n')
    print a.shape
    FitModel(a,b)
    print a         
if __name__== "__main__":
      main()