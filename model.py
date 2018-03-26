
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
    layer1_num = 4

    # Number of epochs(iterations of forward and backward passes)
    epoch_num = 1000

    # Number of batches to be run in an epoch
    train_num = 400

    # Batch Size for a particular training set
    batch_size = 512   #226200

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

    #---------------------- Start of bullshit
    #print(tf.trainable_variables())
 
    # Declaration of input to hidden layer variables
    var = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]
    weight_initial_input = sess.run(var)
    delta_weight_input = np.zeros(shape=(x_columns,layer1_num))
    difference_weightinitial_input = np.zeros(shape=(x_columns,layer1_num))
    current_weight_input = np.zeros(shape=(x_columns,layer1_num))
    previous_weight_input = np.zeros(shape=(x_columns,layer1_num))
    sensitivity_input = np.zeros(shape=(x_columns,layer1_num))
    
	#Declaration of hidden to output layer variables
    var = [v for v in tf.trainable_variables() if v.name == "Variable_2:0"][0]
    weight_initial_hidden = sess.run(var)
    delta_weight_hidden = np.zeros(shape=(layer1_num,y_columns))
    difference_weightinitial_hidden = np.zeros(shape=(layer1_num,y_columns))
    current_weight_hidden = np.zeros(shape=(layer1_num,y_columns))
    previous_weight_hidden = np.zeros(shape=(layer1_num,y_columns))
    sensitivity_hidden = np.zeros(shape=(layer1_num,y_columns))


    # Start Training
    for epoch in range(epoch_num):
       	previous_weight_input = current_weight_input
    	previous_weight_hidden = current_weight_hidden


        avg_loss = 0.
        for i in range(train_num):
            x_train_batch = x_train[i*batch_size:(i+1)*(batch_size)]
            y_train_batch = y_train[i*batch_size:(i+1)*(batch_size)]

            _,c = sess.run([train_step,loss],feed_dict={x:x_train_batch,y:y_train_batch})
            avg_loss += c/train_num
        if epoch % display_size == 0:
            print("Epoch:{0},Loss:{1}".format(epoch+1,avg_loss))

    	#--------------------- Random Bullshit

    	# Find sensitivities for input to hidden layer
    	var = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]
    	current_weight_input = sess.run(var)
    	delta_weight_input = np.subtract(current_weight_input,previous_weight_input)
    	difference_weightinitial_input = np.subtract(current_weight_input,weight_initial_input)
    	
    	#--------------------------Loop over for all sensitivity values
    	for row_loop in range(x_columns):
    		for col_loop in range(layer1_num):		
    			if(difference_weightinitial_input[row_loop,col_loop]==0):
    				print "Keeping sensitivity as the same"
    			else:
    				sensitivity_input[row_loop,col_loop] += abs(((delta_weight_input[row_loop,col_loop]*delta_weight_input[row_loop,col_loop])*100.0*current_weight_input[row_loop,col_loop])/(difference_weightinitial_input[row_loop,col_loop]))

    	# Find sensitivities for hidden to output layer
        var = [v for v in tf.trainable_variables() if v.name == "Variable_2:0"][0]
    	current_weight_hidden = sess.run(var)
    	delta_weight_hidden = np.subtract(current_weight_hidden,previous_weight_hidden)
    	difference_weightinitial_hidden = np.subtract(current_weight_hidden,weight_initial_hidden)
    	
    	#--------------------------Loop over for all sensitivity values
    	for row_loop in range(layer1_num):
    		for col_loop in range(y_columns):		
    			if(difference_weightinitial_hidden[row_loop,col_loop]==0):
    				print "Keeping sensitivity as the same"
    			else:
    				sensitivity_hidden[row_loop,col_loop] += abs(((delta_weight_hidden[row_loop,col_loop]*delta_weight_hidden[row_loop,col_loop])*100.0*current_weight_hidden[row_loop,col_loop])/(difference_weightinitial_hidden[row_loop,col_loop]))

    print "Input-Hidden Layer Sensitivities"
    print sensitivity_input
    print "Hidden-Output Layer Sensitivities"
    print sensitivity_hidden
    print("Training Finished")

def main():
    # Load feature set
    x_train = np.loadtxt('sampledata.txt', delimiter=' ')
    
    # Remove attribute1[uid] not required as it has no relation to output
    x_train = x_train[:,1:9]

    # Load output labels for respective feature set
    y_train = np.loadtxt('output.txt',delimiter='\n')

    # Randomize all the tuples
    y_train = y_train[:,None]
    x_train = np.hstack((x_train,y_train))
    np.random.shuffle(x_train)

    row,col = x_train.shape
    y_train = x_train[:,col-1]
    x_train = x_train[:,0:8]

    # Call the Neural Network logic
    FitModel(x_train,y_train)
         
if __name__== "__main__":
      main()