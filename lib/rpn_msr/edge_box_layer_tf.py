# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import pdb
import math
from networks.network import *
#from networks.network import Network
from keras import backend as K
#import mxnet as mx
DEBUG = False


DEFAULT_PADDING = 'SAME'
def make_var(name, shape, initializer=None, trainable=True):
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

def validate_padding(padding):
    assert padding in ('SAME', 'VALID')

def fc(input, num_out, name, relu=True, trainable=True):
        print "========== fc_layer ========="
	with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]
	
            #print input	
            input_shape = input.get_shape()
            
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = make_var('weights', [dim, num_out], init_weights, trainable)
            biases = make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            
            return fc

def fc1(input, num_out, name, relu=True, trainable=True):
        print "========== fc_layer ========="
	with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]
	    input_shape = input.shape
            #print input	
            
            feed_in, dim = (input, int(input_shape[-1]))
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)

            weights = make_var('weights', [dim, num_out], init_weights, trainable)
            biases = make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            
           
            return fc

def conv( input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

def grouppp(input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=16, trainable=True):
    
    validate_padding(padding)
    c_i = input.get_shape()[1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:

        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
        init_biases = tf.constant_initializer(0.0)
        #kernel = make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
        kernel = make_var('weights', [ k_h, k_w,c_o,c_i/group], init_weights, trainable)
        #biases = make_var('biases', [c_o], init_biases, trainable)
        biases = make_var('biases', [c_o], init_biases, trainable)
        

        input_groups=tf.transpose(input,perm=[0,2,3,1])    
        input_groups = tf.split(input_groups,group,3)
        
        #kernel=tf.transpose(kernel,perm=[3,2,0,1])
        kernel_groups = tf.split( kernel,group,3)
        
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        
        print(output_groups)
        
        conv = tf.concat( output_groups,3)
        #conv=output_groups[0]
        
        if relu:
            bias = tf.nn.bias_add(conv, biases)
            
            return tf.nn.relu(bias, name=scope.name)
        return tf.nn.bias_add(conv, biases, name=scope.name)
     
    

def edge_box_layer(rois,roi_feat,fc_dim=16,feat_dim=1024,
                                    dim=(1024, 1024, 1024),
                                    group=16, index=1):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    n_boxes = len(rois) #128, 256
    # allow boxes to sit over the edge by a small amount
    #_allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]
    
    ofo, ofs = tf.split(roi_feat, [n_boxes, 1], 0)

    roi_feat1 = tf.reshape(ofo, [n_boxes, 1024])
    
    print('==========================================roi_feat')
    print(roi_feat1)
    print('==========================================roi_feat')  
    t=tf.slice(rois,[0,1],[n_boxes,4])
    xmin, ymin, xmax, ymax = tf.split(t,4,1)
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    
    delta_x=tf.multiply(center_x,tf.transpose(center_x))
    
    delta_x = tf.div(delta_x, bbox_width)
    delta_x = tf.div(delta_x, bbox_width)
    delta_x = tf.log(tf.maximum(tf.abs(delta_x), 1e-3))
    delta_y = tf.multiply(center_y,tf.transpose(center_y))
    delta_y = tf. log(tf.maximum(tf.abs(delta_y), 1e-3))
    delta_width = tf.div(bbox_width,tf.transpose(bbox_width))
    delta_width = tf.log(delta_width)
    delta_height = tf.div(bbox_height,tf.transpose(bbox_height))
    delta_height = tf.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    
    for idx, sym in enumerate(concat_list):
        sym = tf.slice(sym, [0,0], [n_boxes,n_boxes]) 
        concat_list[idx] = tf.expand_dims(sym, 2)
       # print(concat_list)
    
    position_matrix = tf.concat([concat_list[0],concat_list[1],concat_list[2],concat_list[3]],2)

    
    
    feat_range = np.arange(0, 8)
    
    dim_mat = np.power(np.full((1,), 1000),(8. / 64) * feat_range)
    
    dim_mat = tf.reshape(dim_mat, [1, 1, 1, -1])
    
    dim_mat=tf.cast(dim_mat,tf.float32)
    
    position_matrix = tf.expand_dims(100.0 * position_matrix, 3)
    
    div_mat = tf.div(position_matrix,dim_mat)
    
    sin_mat = tf.sin(div_mat)
    
    cos_mat = tf.cos(div_mat)
    
    embedding = tf.concat([sin_mat, cos_mat],3)
    embedding = tf.reshape(embedding,[n_boxes, n_boxes, 64])
        # embedding, [num_rois, nongt_dim, feat_dim]
    
 
    dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
    nongt_roi_feat = tf.slice(roi_feat1, [0,0], [-1,feat_dim])
    
        # [num_rois * nongt_dim, emb_dim]
    position_embedding_reshape = tf.reshape(embedding, [n_boxes*n_boxes, 64])
    
        # position_feat_1, [num_rois * nongt_dim, fc_dim]
    position_feat_1 = fc(position_embedding_reshape,fc_dim,name='pair_pos_fc1_' )
    
    position_feat_1_relu = tf.nn.relu(position_feat_1)
    
        # aff_weight, [num_rois, nongt_dim, fc_dim]
    aff_weight = tf.reshape(position_feat_1_relu, [-1, n_boxes, fc_dim])
    
        # aff_weight, [num_rois, fc_dim, nongt_dim]
    aff_weight = tf.transpose(aff_weight, perm=[0, 2, 1])
    

        # multi head
    assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
    
    q_data = fc1(roi_feat1,1024,name='query_' )
    
    q_data_batch = tf.reshape(q_data, [-1, group, dim_group[0]])
    q_data_batch = tf.transpose(q_data_batch, perm=[1, 0, 2])
    k_data = fc(nongt_roi_feat,dim[1],name='key_')
    k_data_batch = tf.reshape(k_data, [-1, group, dim_group[1]])
    k_data_batch = tf.transpose(k_data_batch, perm=[1, 0, 2])
    v_data = nongt_roi_feat
    
    
        # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
    k_data_batch=tf.transpose(k_data_batch,perm=[0, 2, 1])
    aff = K.batch_dot(q_data_batch, k_data_batch)
   
        # aff_scale, [group, num_rois, nongt_dim]
    aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
    aff_scale = tf.transpose(aff_scale, perm=[1, 0, 2])
    
    assert fc_dim == group, 'fc_dim != group'
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
    weighted_aff = tf.log(tf.maximum(aff_weight,1e-6)) + aff_scale
    
    aff_softmax = tf.nn.softmax(weighted_aff, dim=2, name='softmax_' )
    
        # [num_rois * fc_dim, nongt_dim]
    aff_softmax_reshape = tf.reshape(aff_softmax, [n_boxes*16, n_boxes])
    
        # output_t, [num_rois * fc_dim, feat_dim]
    #output_t = np.dot(aff_softmax_reshape, v_data)
    
    output_t = tf.matmul(aff_softmax_reshape, v_data)
    
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
    output_t = tf.reshape(output_t, [n_boxes, fc_dim * feat_dim, 1, 1])
    
    #in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,in12,in13,in14,in15,in16=tf.split(output_t,group,1)
    
    #print(in1)
    #print(in2)
    linear_out = grouppp(output_t,1, 1, 1024,1,1,name='linear_out_' )
   
        # linear_out, [num_rois, dim[2], 1, 1]
    
    
    #output = tf.reshape(linear_out, [0, 0])
    output=tf.reshape(linear_out, [n_boxes, 1024])
   
    return output
