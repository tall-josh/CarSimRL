# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import constants as CONST
import os
import copy

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
    
def new_conv_layer(prev_layer, 
                   num_input_channels,
                   filter_size,
                   num_filters,
                   use_pooling=False,
                   layer_name = "Conv_Layer"):
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    
    with tf.name_scope(layer_name):
        with tf.name_scope(layer_name, "_Weights"):
            # Create new weights aka. filters with the given shape.
            weights = new_weights(shape=shape)
            tf.summary.histogram((layer_name + "_w_hist"), weights)
    
        with tf.name_scope(layer_name, "_Biases"):
            # Create new biases, one for each filter.
            biases = new_biases(length=num_filters)
    
        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=prev_layer,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        
        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases
        
        
        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
    
        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights
    
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
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(prev_layer, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer

        
        ##### CNN Layout ##### 
        
try: 
    tf.reset_default_graph()      
except:
    pass
        
# Conv Layer 1
filter_sz1 = 3#5
num_filters1 = 8#16

# Conv Layer 2
filter_sz2 = 5#5
num_filters2 = 10#36

# Conv Layer 3
filter_sz3 = 5#5
num_filters3 = 16#36

# Fully connected
fc1_size = 100#128

# Fully connected
fc2_size = 25#128

##### Data dimentions #####
image_size = CONST.STATE_MATRIX_SIZE      
num_channels = 1
num_classes = 5

# placeholders for input nodes
state_mat = tf.placeholder(tf.float32, shape=[None, image_size[0], image_size[1], num_channels], name = 'state_mat')

q_matrix = tf.placeholder(tf.float32, shape=[None, num_classes], name = 'q_matrix')

q_target = tf.placeholder(tf.float32, shape=[None, num_classes], name = 'q_target')


layer_conv1, weights_conv1 = new_conv_layer(prev_layer=state_mat,
                                                 num_input_channels = num_channels,
                                                 filter_size = filter_sz1,
                                                 num_filters=num_filters1,
                                                 use_pooling=False)

layer_conv2, weights_conv2 = new_conv_layer(prev_layer=layer_conv1,
                                                 num_input_channels = num_filters1,
                                                 filter_size = filter_sz2,
                                                 num_filters=num_filters2,
                                                 use_pooling=True)

layer_conv3, weights_conv3 = new_conv_layer(prev_layer=layer_conv2,
                                                 num_input_channels = num_filters2,
                                                 filter_size = filter_sz3,
                                                 num_filters=num_filters3,
                                                 use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv3)

layer_fc1 = new_fc_layer(prev_layer = layer_flat,
                         num_inputs = num_features,
                         num_outputs = fc1_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(prev_layer = layer_fc1,
                         num_inputs = fc1_size,
                         num_outputs = fc2_size,
                         use_relu=True)

q_matrix = new_fc_layer(prev_layer=layer_fc2,
                         num_inputs=fc2_size,
                         num_outputs=num_classes,
                         use_relu=False)

##### CLASS PREDICTION #####

reduction = tf.square(q_target - q_matrix)
cost = tf.reduce_mean(reduction)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

saver = tf.train.Saver()
save_dir = 'checkpoints_1/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_path = os.path.join(save_dir, 'values')
session = tf.Session()
merged_summ = tf.summary.merge_all()
train_writer = tf.train.SummaryWriter("logs/", session.graph)
session.run(tf.global_variables_initializer())

def getQMat(state_in):
    #copy and make correct shape for feed_dict
    state_feed = copy.deepcopy(state_in)
    state_feed = np.expand_dims(state_feed, axis=0)
    state_feed = np.expand_dims(state_feed, axis=3)
    
    feed_dict = {state_mat: state_feed}
                 
    return session.run(q_matrix, feed_dict=feed_dict)

    
def fitBatch(batch_state, batch_target, save=False, verbose=False):
    state_feed = copy.deepcopy(batch_state)
    state_feed = np.expand_dims(state_feed, axis=3)
    
    q_feed = copy.deepcopy(batch_target)
            
    state_action_dict = {state_mat: state_feed, q_target: q_feed}  
    lossVal = session.run([optimizer, cost], feed_dict = state_action_dict)
    
    if verbose:
        print("LOSS_VAL: {0}: ".format(lossVal))
    if save:
        saver.save(sess=session, save_path=save_path)
    