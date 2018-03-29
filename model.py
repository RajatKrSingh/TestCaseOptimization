
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


def CalculateAccuracy(y_labels, prediction_values):
	correct_values = 0
	for i in range(y_labels.shape[0]):
	    if(y_labels[i]==prediction_values[i]):
	        correct_values +=1
	return 1.0*correct_values/y_labels.shape[0]	
    
def GetPrunedWeight(sensitivity_input,sensitivity_hidden,minindex_layer0,minindex_layer1,x_columns,layer1_num):
    if(minindex_layer0 >= x_columns*layer1_num):
        min_htoo = np.partition(sensitivity_hidden.flatten(), minindex_layer1)[minindex_layer1]
        i,j = np.where(sensitivity_hidden==min_htoo)
        return i[0],j[0],1
    elif(minindex_layer1 >= layer1_num*2):
        min_itoh = np.partition(sensitivity_input.flatten(), minindex_layer0)[minindex_layer0]
        i,j = np.where(sensitivity_input == min_itoh)
        return i[0],j[0],0

    min_itoh = np.partition(sensitivity_input.flatten(), minindex_layer0)[minindex_layer0]
    min_htoo = np.partition(sensitivity_hidden.flatten(), minindex_layer1)[minindex_layer1]
    print min_itoh,min_htoo
    if(min_itoh<min_htoo):
    	i,j = np.where(sensitivity_input == min_itoh)
    	print i[0],j[0]
        return i[0],j[0],0

    i,j = np.where(sensitivity_hidden==min_htoo)
    return i[0],j[0],1


""" Neural Network Initialization """
def FitModel(x_train,y_train):
    # Number of features
    x_columns = 8

    # Number of output values
    y_columns = 2

    # Number of neurons in Hidden Layer(Hit and Trial) 
    layer1_num = 4

    # Number of epochs(iterations of forward and backward passes)
    epoch_num = 3000

    # Number of batches to be run in an epoch
    train_num = 400

    # Batch Size for a particular training set
    batch_size = 512   #226200

    # AVriable for checking progress
    display_size = 10

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
    
 
    # Declaration of input to hidden layer variables
    var = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]
    #sess.run(tf.assign(var,tf.cast(tf.constant(np.zeros(shape=(x_columns,layer1_num))),tf.float32)))
    weight_initial_input = sess.run(var)
    #print weight_initial_input
    

    #print(tf.trainable_variables())
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

    np.savetxt("sensitivity_input.txt",sensitivity_input)
    np.savetxt("sensitivity_hidden.txt",sensitivity_hidden)
    

    pruning_prediction = 1.0

    # Start pruning the ANN based on sensitivities   
    count=0

    # Variables that keep count of number of weights pruned in the first and second layer respectively
    minindex_layer0 = 0
    minindex_layer1 = 0

    # Array of indexes of pruned weights for first and second layer
    xinput_pruned = np.array([])
    xhidden_pruned = np.array([])
    yinput_pruned = np.array([])
    yhidden_pruned = np.array([])

    # Variables for final weight pruned when accuracy falls below threshold stored to be restored
    weight_to_restore = 0.0
    layer_to_restore = 0
    xindex_to_restore = 0
    yindex_to_restore = 0

    # Threshold accuracy
    threshold_acc = 0.95

    #Start actual pruning iterations
    while(pruning_prediction>threshold_acc):

    	# Get layer and index of the weight corresponding to the minimum sensitivity
        i_index,j_index,layer = GetPrunedWeight(sensitivity_input,sensitivity_hidden,minindex_layer0,minindex_layer1,x_columns,layer1_num)
        
        # Restore values stored 
        layer_to_restore,xindex_to_restore,yindex_to_restore = layer,i_index,j_index
        
        # Check condition for pruned weight in first layer or second layer
        if(layer==0):

        	# Get current weight
            var = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]
            new_replaced_matrix = sess.run(var)
            
            # Store weight for restore if necessarily
            weight_to_restore = new_replaced_matrix[i_index,j_index]

            # Prune actual weight
            new_replaced_matrix[i_index,j_index] = 0

            # Store the pruned weight index to initialize to 0
            xinput_pruned = (np.append(xinput_pruned,i_index)).astype(int)
            yinput_pruned = (np.append(yinput_pruned,j_index)).astype(int)
            
            # Save the new weight matrix 
            sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))
            
            # Increment variable for number of weights pruned
            minindex_layer0 += 1

        elif(layer==1):

            var = [v for v in tf.trainable_variables() if v.name == "Variable_2:0"][0]
            new_replaced_matrix = sess.run(var)
            
            weight_to_restore = new_replaced_matrix[i_index,j_index]
            
            new_replaced_matrix[i_index,j_index] = 0

            xhidden_pruned = (np.append(xhidden_pruned,i_index)).astype(int)
            yhidden_pruned = (np.append(yhidden_pruned,j_index)).astype(int)
            
            sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))
            
            minindex_layer1 += 1  

       

        print("Layer Pruned:{0},Index Pruned:{1},{2} Prune Count {3} and {4}".format(layer,i_index,j_index,minindex_layer0,minindex_layer1))
        
        # Get accuracy for pruned network against training set
        correct_pred = tf.argmax(prediction,1)
        prediction_values = sess.run([correct_pred],feed_dict={x: x_train})[0]
        print "Pruned Accuracy is:{0}".format(CalculateAccuracy(y_train,prediction_values))
        pruning_prediction = CalculateAccuracy(y_train,prediction_values)

        # Fine tune parameters of ANN
        for epoch in range(20):

            for i in range(train_num):
                x_train_batch = x_train[i*batch_size:(i+1)*(batch_size)]
                y_train_batch = y_train[i*batch_size:(i+1)*(batch_size)]

                _,c = sess.run([train_step,loss],feed_dict={x:x_train_batch,y:y_train_batch})
                
            # Replace the weights for the pruned weights to zero which are non zero because of fine tuning
            var = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]
            new_replaced_matrix = sess.run(var)
            
            for izero_index in range(minindex_layer0):
                new_replaced_matrix[xinput_pruned[izero_index],yinput_pruned[izero_index]] = 0

            sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))
            
            var = [v for v in tf.trainable_variables() if v.name == "Variable_2:0"][0]
            new_replaced_matrix = sess.run(var)
            for izero_index in range(minindex_layer1):
                new_replaced_matrix[xhidden_pruned[izero_index],yhidden_pruned[izero_index]] = 0

            sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))

        # Print accuracy after fine tuning 
        prediction_values = sess.run([correct_pred],feed_dict={x: x_train})[0]
        print "Pruned Accuracy is:{0}".format(CalculateAccuracy(y_train,prediction_values))
        pruning_prediction = CalculateAccuracy(y_train,prediction_values)
        count += 1

    # Restore final weight
    print("Go Back")

    if(layer_to_restore==0):
        var = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]
        new_replaced_matrix = sess.run(var)
        new_replaced_matrix[xindex_to_restore,yindex_to_restore] = weight_to_restore

        sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))
        minindex_layer0 -= 1
    
    else:
        var = [v for v in tf.trainable_variables() if v.name == "Variable_2:0"][0]
        new_replaced_matrix = sess.run(var)
        new_replaced_matrix[xindex_to_restore,yindex_to_restore] = weight_to_restore
        
        sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))
        minindex_layer1 -= 1

    print("Replaced")

    # Replace the weights for the pruned weights to zero which are non zero because of fine tuning
    var = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]
    new_replaced_matrix = sess.run(var)
            
    for izero_index in range(minindex_layer0):
        new_replaced_matrix[xinput_pruned[izero_index],yinput_pruned[izero_index]] = 0

    sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))
            
    var = [v for v in tf.trainable_variables() if v.name == "Variable_2:0"][0]
    new_replaced_matrix = sess.run(var)
            
    for izero_index in range(minindex_layer1):
        new_replaced_matrix[xhidden_pruned[izero_index],yhidden_pruned[izero_index]] = 0

    sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))

    
    # Fine tune network one last time
    for epoch in range(epoch_num):

        for i in range(train_num):
            x_train_batch = x_train[i*batch_size:(i+1)*(batch_size)]
            y_train_batch = y_train[i*batch_size:(i+1)*(batch_size)]

            _,c = sess.run([train_step,loss],feed_dict={x:x_train_batch,y:y_train_batch})
    
    # Replace the weights for the pruned weights to zero which are non zero because of fine tuning
    var = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]
    new_replaced_matrix = sess.run(var)
            
    for izero_index in range(minindex_layer0):
        new_replaced_matrix[xinput_pruned[izero_index],yinput_pruned[izero_index]] = 0

    sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))
            
    var = [v for v in tf.trainable_variables() if v.name == "Variable_2:0"][0]
    new_replaced_matrix = sess.run(var)
            
    for izero_index in range(minindex_layer1):
        new_replaced_matrix[xhidden_pruned[izero_index],yhidden_pruned[izero_index]] = 0

    sess.run(tf.assign(var,tf.cast(tf.constant(new_replaced_matrix),tf.float32)))

    # The final accuracy of pruned network
    prediction_values = sess.run([correct_pred],feed_dict={x: x_train})[0]
    print "Pruned Accuracy is:{0}".format(CalculateAccuracy(y_train,prediction_values))
    pruning_prediction = CalculateAccuracy(y_train,prediction_values)
    
    # Print weights in both layers
    var = [v for v in tf.trainable_variables() if v.name == "Variable:0"][0]
    print sess.run(var)

    var = [v for v in tf.trainable_variables() if v.name == "Variable_2:0"][0]
    print sess.run(var)
    print("Done")

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