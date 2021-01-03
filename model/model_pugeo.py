import tensorflow as tf
import numpy as np
import math
import os
import sys

from utils import tf_util
from utils.transform_nets import input_transform_net

def get_model(point_cloud,  up_ratio, is_training, bradius=1.0, knn=30,scope='generator',weight_decay=0.0, bn_decay=None, bn=True, fd=64, fD=1024):   
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        input_point_cloud = tf.expand_dims(point_cloud, -1) 

        adj = tf_util.pairwise_distance(point_cloud) 
        nn_idx = tf_util.knn(adj, k=knn) 
        edge_feature = tf_util.get_edge_feature(input_point_cloud, nn_idx=nn_idx, k=knn)

        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_point_cloud = tf.expand_dims(point_cloud_transformed, -1) 
        adj = tf_util.pairwise_distance(point_cloud_transformed)
        nn_idx = tf_util.knn(adj, k=knn)
        edge_feature = tf_util.get_edge_feature(input_point_cloud, nn_idx=nn_idx, k=knn)

        out1 = tf_util.conv2d(edge_feature, fd, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='dgcnn_conv1', bn_decay=bn_decay)
          
        out2 = tf_util.conv2d(out1, fd, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='dgcnn_conv2', bn_decay=bn_decay)

        net_max_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)
        net_mean_1 = tf.reduce_mean(out2, axis=-2, keep_dims=True)

        out3 = tf_util.conv2d(tf.concat([net_max_1, net_mean_1], axis=-1), fd, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='dgcnn_conv3', bn_decay=bn_decay)

        adj = tf_util.pairwise_distance(tf.squeeze(out3, axis=-2))
        nn_idx = tf_util.knn(adj, k=knn)
        edge_feature = tf_util.get_edge_feature(out3, nn_idx=nn_idx, k=knn)

        out4 = tf_util.conv2d(edge_feature, fd, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='dgcnn_conv4', bn_decay=bn_decay)
          
        net_max_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)
        net_mean_2 = tf.reduce_mean(out4, axis=-2, keep_dims=True)

        out5 = tf_util.conv2d(tf.concat([net_max_2, net_mean_2], axis=-1), fd, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='dgcnn_conv5', bn_decay=bn_decay)

        adj = tf_util.pairwise_distance(tf.squeeze(out5, axis=-2))
        nn_idx = tf_util.knn(adj, k=knn)
        edge_feature = tf_util.get_edge_feature(out5, nn_idx=nn_idx, k=knn)

        out6 = tf_util.conv2d(edge_feature, fd, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='dgcnn_conv6', bn_decay=bn_decay)

        net_max_3 = tf.reduce_max(out6, axis=-2, keep_dims=True)
        net_mean_3 = tf.reduce_mean(out6, axis=-2, keep_dims=True)

        out7 = tf_util.conv2d(tf.concat([net_max_3, net_mean_3], axis=-1), fd, [1,1],
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training, weight_decay=weight_decay,
                               scope='dgcnn_conv7', bn_decay=bn_decay)

        out8 = tf_util.conv2d(tf.concat([out3, out5, out7], axis=-1), fD, [1, 1], 
                               padding='VALID', stride=[1,1],
                               bn=bn, is_training=is_training,
                               scope='dgcnn_conv8', bn_decay=bn_decay)

        out_max = tf_util.max_pool2d(out8, [num_point, 1], padding='VALID', scope='maxpool')

        expand = tf.tile(out_max, [1, num_point, 1, 1])

        concat_unweight = tf.concat(axis=3, values=[expand, 
                                             net_max_1,
                                             net_mean_1,
                                             out3,
                                             net_max_2,
                                             net_mean_2,
                                             out5,
                                             net_max_3,
                                             net_mean_3,
                                             out7,
                                             out8])
        
        
        feat_list = ["expand",  "net_max_1", "net_mean_1", "out3", "net_max_2", "net_mean_2", "out5", "net_max_3", "net_mean_3", "out7", "out8"]

        out_attention = tf_util.conv2d(concat_unweight, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                    bn=True, is_training=is_training, scope='attention_conv1', weight_decay=weight_decay)
        out_attention = tf_util.conv2d(out_attention,  64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                    bn=True, is_training=is_training, scope='attention_conv2', weight_decay=weight_decay)
        out_attention = tf_util.conv2d(out_attention, len(feat_list), [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                    bn=True, is_training=is_training, scope='attention_conv3', weight_decay=weight_decay)

        out_attention = tf_util.max_pool2d(out_attention, [num_point, 1], padding='VALID', scope='attention_maxpool')
        out_attention = tf.nn.softmax(out_attention)
        tmp_attention = tf.squeeze(out_attention)

        for i in range(len(feat_list)):
            tmp1 = tf.slice(out_attention, [0,0,0,i], [1,1,1,1])
            exec('dim = %s.get_shape()[-1].value' % feat_list[i])
            tmp2 = tf.tile(tmp1, [1,1,1,dim])
            if i==0:
                attention_weight = tmp2
            else:
                attention_weight = tf.concat([attention_weight, tmp2], axis=-1)
        attention_weight = tf.tile(attention_weight, [1, num_point, 1, 1])
        concat = tf.multiply(concat_unweight, attention_weight)

        concat = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                    bn=True, is_training=is_training, scope='concat_conv', weight_decay=weight_decay)
        concat = tf_util.dropout(concat, keep_prob=0.6, is_training=is_training, scope='dg1')

        with tf.variable_scope('uv_predict'):
            uv_2d = tf_util.conv2d(concat, up_ratio*2, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, activation_fn=None,
                    bn=None, is_training=is_training, scope='uv_conv1', weight_decay=weight_decay)
            uv_2d = tf.reshape(uv_2d, [batch_size, num_point, up_ratio,2])
            uv_2d = tf.concat([uv_2d, tf.zeros([batch_size, num_point, up_ratio, 1])], axis=-1)

        with tf.variable_scope('T_predict'):
            affine_T = tf_util.conv2d(concat, 9, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, activation_fn=None,
                    bn=None, is_training=is_training, scope='patch_conv1', weight_decay=weight_decay)
            affine_T = tf.reshape(affine_T, [batch_size, num_point, 3,3])

            uv_3d = tf.matmul(uv_2d, affine_T)
            uv_3d = uv_3d + tf.tile(tf.expand_dims(point_cloud, axis=-2), [1,1, up_ratio,1])
            uv_3d = tf.transpose(uv_3d, perm=[0,2,1,3])
            uv_3d = tf.reshape(uv_3d, [batch_size, num_point*up_ratio, -1])

        with tf.variable_scope('normal_predict'):
            dense_normal_offset = tf_util.conv2d(concat, up_ratio*3, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, activation_fn=None,
                    bn=None, is_training=is_training, scope='normal_offset_conv1', weight_decay=weight_decay)
            dense_normal_offset = tf.reshape(dense_normal_offset, [batch_size, num_point, up_ratio, 3])

            sparse_normal = tf.convert_to_tensor([0,0,1], dtype=tf.float32)
            sparse_normal = tf.expand_dims(sparse_normal, axis=0)
            sparse_normal = tf.expand_dims(sparse_normal, axis=0)
            sparse_normal = tf.expand_dims(sparse_normal, axis=0)
            sparse_normal = tf.tile(sparse_normal, [batch_size, num_point, 1, 1])
            sparse_normal = tf.matmul(sparse_normal, affine_T)
            sparse_normal = tf.nn.l2_normalize(sparse_normal, axis=-1)

            dense_normal = tf.tile(sparse_normal, [1,1,up_ratio,1]) + dense_normal_offset

            dense_normal = tf.nn.l2_normalize(dense_normal, axis=-1)
            dense_normal = tf.transpose(dense_normal, perm=[0,2,1,3])
            dense_normal = tf.reshape(dense_normal, [batch_size, num_point*up_ratio, -1])

        with tf.variable_scope('up_layer'):
            if not np.isscalar(bradius):
                bradius_expand = tf.expand_dims(tf.expand_dims(bradius, axis=-1), axis=-1)
            else:
                bradius_expand = bradius
            bradius_expand = bradius_expand
            grid = tf.expand_dims(uv_3d*bradius_expand, axis=2)
            
            concat_up = tf.tile(concat, (1, up_ratio, 1,1))
            concat_up = tf.concat([concat_up, grid], axis=-1)
            
            concat_up = tf_util.conv2d(concat_up, 128, [1, 1],
					  padding='VALID', stride=[1, 1],
					  bn=True, is_training=is_training,
					  scope='up_layer1', bn_decay=bn_decay,weight_decay=weight_decay)

            concat_up = tf_util.dropout(concat_up, keep_prob=0.6, is_training=is_training, scope='up_dg1')
            
            concat_up = tf_util.conv2d(concat_up, 128, [1, 1],
					 padding='VALID', stride=[1, 1],
					 bn=True, is_training=is_training,
					 scope='up_layer2',
					 bn_decay=bn_decay,weight_decay=weight_decay)
            concat_up = tf_util.dropout(concat_up, keep_prob=0.6, is_training=is_training, scope='up_dg2')

        # get xyz
        coord_z = tf_util.conv2d(concat_up, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
                    bn=False, is_training=is_training, scope='fc_layer', weight_decay=weight_decay)
        coord_z = tf.reshape(coord_z, [batch_size, up_ratio, num_point, 1])
        coord_z = tf.transpose(coord_z, perm=[0,2,1,3])
        coord_z = tf.concat([tf.zeros_like(coord_z), tf.zeros_like(coord_z), coord_z], axis=-1)
        
        coord_z = tf.matmul(coord_z, affine_T)
        coord_z = tf.transpose(coord_z, perm=[0,2,1,3])
        coord_z = tf.reshape(coord_z, [batch_size, num_point*up_ratio, -1])

        coord = uv_3d + coord_z

    return coord, dense_normal, tf.squeeze(sparse_normal, [2])
    





