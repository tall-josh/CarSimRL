# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import constants as CONST
import os
import copy

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name = "W")
    
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]), name = "B")
    
    
def flatten_layer(layer):
    # Get the shape of the input  layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features
    
def new_fc_layer(prev_layer,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 activation=tf.nn.relu,
                 layer_name="FC_Layer"):
    
    with tf.name_scope(layer_name): 
        with tf.name_scope(layer_name+ "/weights"):
            # Create new weights and biases.
            weights = new_weights(shape=[num_inputs, num_outputs])
            tf.summary.histogram(layer_name+"_weights", weights)
        
        with tf.name_scope(layer_name+ "/biases"):
            biases = new_biases(length=num_outputs)
            tf.summary.histogram(layer_name+"_biases", biases)
        
        with tf.name_scope(layer_name+ "/output"):
            # Calculate the layer as the matrix multiplication of
            # the input and weights, and then add the bias-values.
            layer = tf.matmul(prev_layer, weights) + biases            
            layer = activation(layer)
            tf.summary.histogram(layer_name+"_output", layer)
            
    return layer
   
try: 
    tf.reset_default_graph()      
except:
    pass
        
# Conv Layer 1
filter_sz1 = 5
num_filters1 = 16

# Conv Layer 2
filter_sz2 = 3
num_filters2 = 32

# Conv Layer 3
filter_sz3 = 3
num_filters3 = 64#

# Fully connected
fc1_size = 50

activation = tf.nn.relu

##### Data dimentions #####
image_size = CONST.STATE_MATRIX_SIZE      
num_channels = 1
num_classes = 5

# placeholders for input nodes
state_mat = tf.placeholder(tf.float32, shape=[None, image_size[0], image_size[1], num_channels], name = 'state_mat')

q_target = tf.placeholder(tf.float32, shape=[None, num_classes], name = 'q_target')

with tf.name_scope("CONV_1"):
    layer_conv1 = layers.convolution2d(state_mat, num_outputs=num_filters1,
                                       kernel_size=filter_sz1,
                                       activation_fn=activation,
                                       scope='CONV_1')
    layer_conv1_mp = layers.max_pool2d(layer_conv1, 2)

with tf.name_scope("CONV_2"):
    layer_conv2 = layers.convolution2d(layer_conv1_mp, num_outputs=num_filters2,
                                           kernel_size=filter_sz2,
                                           activation_fn=activation,
                                           scope='CONV_2')
    layer_conv2_mp = layers.max_pool2d(layer_conv2, 2)

with tf.name_scope("CONV_3"):    
    layer_conv3 = layers.convolution2d(layer_conv2_mp, num_outputs=num_filters3,
                                           kernel_size=filter_sz3,
                                           activation_fn=activation,
                                           scope='CONV_3')

with tf.name_scope("Flatten"):    
    layer_flat, num_features = flatten_layer(layer_conv3)

with tf.name_scope("FC_1"):        
    layer_fc1 = new_fc_layer(prev_layer = layer_flat,
                             num_inputs = num_features,
                             num_outputs = fc1_size,
                             activation=tf.nn.relu,
                             layer_name = "FC_1")

with tf.name_scope("Q_matrix"):   
    q_matrix = new_fc_layer(prev_layer = layer_fc1,
                             num_inputs = fc1_size,
                             num_outputs = num_classes,
                             activation=tf.nn.relu,
                             layer_name = "q_matrix")


##### Loss Function #####
reduction = tf.square(q_target - q_matrix)
with tf.name_scope("cost"):
    cost = tf.reduce_mean(reduction)
    tf.summary.scalar("cost", cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    
with tf.name_scope("max_Q"):
    max_Q = tf.reduce_max(q_matrix)
    tf.summary.scalar("Q_MAX", max_Q)

saver = tf.train.Saver()
save_dir = 'SYB_4/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_path = os.path.join(save_dir, 'values')
session = tf.Session()

merged_summ = tf.summary.merge_all()
writer = tf.summary.FileWriter("SYB_L4/", session.graph)
session.run(tf.global_variables_initializer())

def getQMat(state_in):
    #copy and make correct shape for feed_dict
    state_feed = copy.deepcopy(state_in)
    state_feed = np.expand_dims(state_feed, axis=0)
    state_feed = np.expand_dims(state_feed, axis=3)
    
    feed_dict = {state_mat: state_feed}
                 
    return session.run(q_matrix, feed_dict=feed_dict)

    
def fitBatch(batch_state, batch_target, save=False, verbose=False, iteration_count=0):
    global merged_summ
    state_feed = copy.deepcopy(batch_state)
    state_feed = np.expand_dims(state_feed, axis=3)
    
    q_feed = copy.deepcopy(batch_target)
            
    state_action_dict = {state_mat: state_feed, q_target: q_feed}  
    
    for i in range(20):
        session.run(optimizer, feed_dict = state_action_dict)
            
    if verbose:
        result = session.run(merged_summ, feed_dict = state_action_dict)
        writer.add_summary(result, iteration_count)
            
    if save:
        print("Saving. iteration: ", iteration_count)
        saver.save(sess=session, save_path=save_path)
    