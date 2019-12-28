import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py

from rpn_msr.union_box_layer_tf import union_box_layer as union_box_layer_py

from rpn_msr.edge_box_layer_tf import edge_box_layer as edge_box_layer_py

from fast_rcnn.config import cfg
import math
from keras import backend as K
from rpn_msr.relation_module_tf import attention_module_multi_head as relation_module_py

from networks.network import *

DEFAULT_PADDING = 'SAME'
#w_file = np.fromfile('./data/w.bin')
#w_file.shape = 21,21
#nongt_dim = cfg.TRAIN.RPN_POST_NMS_TOP_N
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

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

def grouppp(input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=64, trainable=True):
    
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
        
        
        
        conv = tf.concat( output_groups,3)
        #conv=output_groups[0]
        
        if relu:
            bias = tf.nn.bias_add(conv, biases)
            
            return tf.nn.relu(bias, name=scope.name)
        return tf.nn.bias_add(conv, biases, name=scope.name)

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key
                            if not ignore_missing:

                                raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

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
    @layer
    def conv1(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=16, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

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
    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        return tf.reshape(tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales], [tf.float32]),[-1,5],name =name)


    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:

            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3], _feat_stride, anchor_scales],[tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    @layer
    def proposal_target_layer(self, input, classes, name): 
        print "========================= proposal_target_layer ==================="
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:

            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(proposal_target_layer_py,[input[0],input[1],classes],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

            rois = tf.reshape(rois,[-1,5] , name = 'rois') 
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')
            

           
            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    #=====================================relation_module============================================================================#
    
        
    @layer
    def union_box_layer(self, input, name):
        print "========================= union box layer ==================="
        print(input[0])
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
       
        with tf.variable_scope(name) as scope:
            whole = tf.py_func(union_box_layer_py, [input[0], input[1]],[tf.float32])  #input[0]--roi_data
            
            #m=tf.slice()
	    whole = tf.reshape(whole, [-1, 5], name = 'whole')	
            return whole   #(?,5)
    

    @layer
    def edge_box_layer(self, input,n_boxes,fc_dim,feat_dim,dim,group, index,name):
        print "========================= edge box layer ==================="
        #n_boxes = len(input[0][0]) #128, 256
        
    # allow boxes to sit over the edge by a small amount
    #_allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]
        #n_boxes=300
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        ofo, ofs = tf.split(input[1], [n_boxes, 1], 0)
        roi_feat1 = tf.reshape(ofo, [n_boxes, 4096])

        rois=input[0]
        
        t=tf.slice(rois,[0,1],[n_boxes,4])
        
        xmin, ymin, xmax, ymax = tf.split(t,4,1)
        bbox_width = (xmax - xmin) + 1.
        bbox_height = (ymax - ymin) + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        delta_x=tf.multiply(center_x,tf.transpose(center_x))
        #delta_x = tf.div(delta_x, bbox_width)
        delta_x = tf.log(tf.maximum(tf.abs(delta_x/bbox_width), 1e-3))
        delta_y = tf.multiply(center_y,tf.transpose(center_y))
        #delta_y = tf.div(delta_y, bbox_height)
        delta_y = tf. log(tf.maximum(tf.abs(delta_y/bbox_height), 1e-3))
        delta_width = tf.div(bbox_width,tf.transpose(bbox_width))
        delta_width = tf.log(delta_width)
        delta_height = tf.div(bbox_height,tf.transpose(bbox_height))
        delta_height = tf.log(delta_height)

        delta_1=tf.multiply(center_x,tf.transpose(center_x))
	delta_1=pow(delta_1,2)
        delta_1 = tf.div(delta_1, pow(bbox_width,2))
        delta_11 = tf.log(tf.maximum(tf.abs(delta_1), 1e-3))

	delta_2=tf.multiply(center_y,tf.transpose(center_y))
	delta_2=pow(delta_2,2)
        delta_2 = tf.div(delta_2, pow(bbox_height,2))
        delta_12 = tf.log(tf.maximum(tf.abs(delta_2), 1e-3))
	
	#delta_11=pow((center_x - center_y) / (bbox_width + 1), 2)
        #delta_12=pow((center_x - center_y) / (bbox_height + 1), 2)
        concat_list = [delta_x, delta_y, delta_width, delta_height,delta_11,delta_12]
    	print('ppppppppppppppppppppppppppppppppppp',concat_list)
        for idx, sym in enumerate(concat_list):
            sym = tf.slice(sym, [0,0], [n_boxes,n_boxes]) 
            concat_list[idx] = tf.expand_dims(sym, 2)
        #print(concat_list)
    
        position_matrix = tf.concat([concat_list[0],concat_list[1],concat_list[2],concat_list[3],concat_list[4],concat_list[5]],2)

        feat_range = np.arange(0, 12)
    
        dim_mat = np.power(np.full((1,), 1000),(12. / 64) * feat_range)
    
        dim_mat = tf.reshape(dim_mat, [1, 1, 1, -1])
    
        dim_mat=tf.cast(dim_mat,tf.float32)
    
        position_matrix = tf.expand_dims(100.0 * position_matrix, 3)
    
        div_mat = tf.div(position_matrix,dim_mat)
    
        sin_mat = tf.sin(div_mat)
    
        cos_mat = tf.cos(div_mat)
    
        embedding = tf.concat([sin_mat, cos_mat],3)
        embedding = tf.reshape(embedding,[n_boxes, n_boxes, 144])
        # embedding, [num_rois, nongt_dim, feat_dim]
 
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        nongt_roi_feat = tf.slice(roi_feat1, [0,0], [-1,feat_dim])
    
        # [num_rois * nongt_dim, emb_dim]
        position_embedding_reshape = tf.reshape(embedding, [n_boxes*n_boxes, 144])
    
        # position_feat_1, [num_rois * nongt_dim, fc_dim]
        position_feat_1 = fc(position_embedding_reshape,fc_dim,name='pair_pos_fc1_' )
    
        position_feat_1_relu = tf.nn.relu(position_feat_1)
    
        # aff_weight, [num_rois, nongt_dim, fc_dim]
        aff_weight = tf.reshape(position_feat_1_relu, [-1, n_boxes, fc_dim])
    
        # aff_weight, [num_rois, fc_dim, nongt_dim]
        aff_weight = tf.transpose(aff_weight, perm=[0, 2, 1])
    

        # multi head
        assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
    
        q_data = fc1(roi_feat1,4096,name='query_' )
    
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
        aff_softmax_reshape = tf.reshape(aff_softmax, [n_boxes*64, n_boxes])
    
        # output_t, [num_rois * fc_dim, feat_dim]
    #output_t = np.dot(aff_softmax_reshape, v_data)
    
        output_t = tf.matmul(aff_softmax_reshape, v_data)
    
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = tf.reshape(output_t, [n_boxes, fc_dim * feat_dim, 1, 1])
    
   
        linear_out = grouppp(output_t,1, 1, 4096,1,1,name='linear_out_' )
   
        # linear_out, [num_rois, dim[2], 1, 1]
    
    
    #output = tf.reshape(linear_out, [0, 0])
        output=tf.reshape(linear_out, [n_boxes, 4096])
        #edge = tf.reshape(edge, [128, 1024], name = 'edge')
        
        return output

    @layer
    def crop_pool_layer(self, input, name):
	print "========================== crop pool layer ================="
	rois = input[1]
	bottom = input[0]	
	with tf.variable_scope(name) as scope:
      	    batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
      	    bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(16)
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(16)
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = 7 * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')
    #=================================================================================================================#


    @layer
    def reshape_layer(self, input, d,name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

    @layer
    def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
        return feature_extrapolating_op.feature_extrapolating(input,
                              scales_base,
                              num_scale_base,
                              num_per_octave,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, input, axis, name):
	inputs = [input[0], input[1]]
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        print "========== fc_layer ========="
	with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]
	
            print input	
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

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            print fc
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def dropout(self, input, keep_prob, name):
	#print "============ drop out ==========="
        return tf.nn.dropout(input, keep_prob, name=name)

    #======================= my edit ================================
    @layer
    def structure_inference_spmm(self, input, boxes, name):
	print "================ structural inference ==============="
        
	
  	#ofo, ofs = tf.split(input, [n_boxes, 1])               
        
 	n_steps = 2
        n_boxes = boxes #train 128, test 256
        #n_boxes = 300
        n_inputs = 4096 #edit D

        n_hidden_o = 4096 #object
        n_hidden_e = 4096 #edge
	

        ofe = input[1]   
                                                                                              
        ofo, ofs = tf.split(input[0], [n_boxes, 1], 0)

	fo = tf.reshape(ofo, [n_boxes, n_inputs])
          
        #(128,1024)
        #ff=fo+ofe

        fff = tf.reshape(ofe, [n_boxes, n_inputs])
        
        
        fs = tf.reshape(ofs, [1, n_inputs])

        fs = tf.concat(n_boxes * [fs], 0)
        fs = tf.reshape(fs, [n_boxes, 1, n_inputs]) #(128,1,4096)sence
        
	#fe = tf.reshape(ofe, [n_boxes * n_boxes, 12])  #(128*128,12)
        
	#u = tf.get_variable('u', [12, 1], initializer = tf.contrib.layers.xavier_initializer())  #(12,1)
        
	# Form 1
	#W = tf.get_variable('CW', [n_inputs, n_inputs], initializer = tf.orthogonal_initializer())
	
	# Form 2
	#Concat_w = tf.get_variable('Concat_w', [n_inputs * 2, 1], initializer = tf.contrib.layers.xavier_initializer())
        

        E_cell = rnn.GRUCell(n_hidden_e)
       
       
        O_cell = rnn.GRUCell(n_hidden_o)
        
       
	oinput = fs[:, 0, :]#input of sence
        
	hi = fff
        
	
	for t in range(n_steps):
		#with tf.variable_scope('e_gru', reuse=(t!=0)):
                #        _, he = E_cell(inputs= einput, state = he)
		

                with tf.variable_scope('o_gru', reuse=(t!=0)):
                	ho1, hi1 = O_cell(inputs = oinput, state = fo)
                
		with tf.variable_scope('e_gru', reuse=(t!=0)):
                        ho2, hi2 = E_cell(inputs = hi, state = fo)
		
		#maxpooling
		#hi = tf.maximum(hi1, hi2)
		#hi=hi1
		#meanpooling
		hi = tf.concat([hi1, hi2], 0)
		hi = tf.reshape(hi, [2, n_boxes, n_inputs])
		hi = tf.reduce_mean(hi, 0)
		
               # hi=Network.fc(hi,2048)

		
	return hi

