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
from keras import backend as K
import tensorflow as tf
DEBUG = False


def attention_module_multi_head(rois,roi_feat, position_embedding,
                                     fc_dim=16,feat_dim=1024,
                                    dim=(1024, 1024, 1024),
                                    group=16, index=1):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    print()
    print('extra-----------------rois')
    print(rois)
    t=tf.slice(rois,[0,1],[128,4])
    xmin, ymin, xmax, ymax = tf.split(1,4,t)
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
        sym = tf.slice(sym, [0,0], [128,128]) 
        concat_list[idx] = tf.expand_dims(sym, 2)
   # print(concat_list.get_shape().as_list())
    position_matrix = tf.concat(2,concat_list)
    
    
    feat_range = np.arange(0, 64/ 8)
    dim_mat = np.power(np.full((1,), 1000),(8. / 64) * feat_range)
    dim_mat = tf.reshape(dim_mat, [1, 1, 1, -1])
    position_mat = tf.expand_dims(100.0 * position_mat, 3)
    div_mat = tf.div(position_mat,dim_mat)
    sin_mat = tf.sin(div_mat)
    cos_mat = tf.cos(div_mat)
    embedding = tf.concat(3,[sin_mat, cos_mat])
    embedding = tf.concat(3,[sin_mat, cos_mat])
        # embedding, [num_rois, nongt_dim, feat_dim]
    embedding = tf.reshape(embedding,[128, 128, 64])

        
    dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
    nongt_roi_feat = tf.slice(roi_feat, [0,0], [-1,128])
        # [num_rois * nongt_dim, emb_dim]
    position_embedding_reshape = tf.reshape(position_embedding, [-3, -2])
        # position_feat_1, [num_rois * nongt_dim, fc_dim]
    position_feat_1 = Network.fc(position_embedding_reshape,fc_dim,name='pair_pos_fc1_' )
    position_feat_1_relu = Network.relu(position_feat_1)
        # aff_weight, [num_rois, nongt_dim, fc_dim]
    aff_weight = tf.reshape(position_feat_1_relu, [-1, 128, fc_dim])
        # aff_weight, [num_rois, fc_dim, nongt_dim]
    aff_weight = tf.transpose(aff_weight, perm=[0, 2, 1])

        # multi head
    assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
    q_data = Network.fc(roi_feat,num_hidden=dim[0],name='query_' )
    q_data_batch = tf.reshape(q_data, [-1, group, dim_group[0]])
    q_data_batch = tf.transpose(q_data_batch, perm=[1, 0, 2])
    k_data = Network.fc(nongt_roi_feat,dim[1],name='key_')
    k_data_batch = tf.reshape(k_data, [-1, group, dim_group[1]])
    k_data_batch = tf.transpose(k_data_batch, perm=[1, 0, 2])
    v_data = nongt_roi_feat
        # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
    aff = K.batch_dot(q_data_batch, k_data_batch, transpose_a=False, transpose_b=True)
        # aff_scale, [group, num_rois, nongt_dim]
    aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
    aff_scale = tf.transpose(aff_scale, perm=[1, 0, 2])

    assert fc_dim == group, 'fc_dim != group'
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
    weighted_aff = tf.log(tf.maximum(aff_weight,1e-6)) + aff_scale
    aff_softmax = Network.softmax(weighted_aff, dim=2, name='softmax_' )
        # [num_rois * fc_dim, nongt_dim]
    aff_softmax_reshape = tf.reshape(aff_softmax, [-3, -2])
        # output_t, [num_rois * fc_dim, feat_dim]
    output_t = np.dot(aff_softmax_reshape, v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
    output_t = tf.reshape(output_t, [-1, fc_dim * feat_dim, 1, 1])
        # linear_out, [num_rois, dim[2], 1, 1]
    linear_out = tf.nn.conv2d(output_t,kernel=(1, 1), num_filter=dim[2], num_group=fc_dim,name='linear_out_' )
    output = tf.reshape(linear_out, [0, 0])
    print('===================================================================================================output')
    print(output)
    return output


