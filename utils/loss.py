import os
import sys
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tf_ops.CD import tf_nndistance

def batch_gather(param, idx):
    """
    param: [B, N, C]
    idx: [B, K]
    output: [B, K, C]
    """
    B = param.get_shape()[0].value
    #K = idx.get_shape()[1].value
    K = tf.shape(idx)[1]
    c = tf.convert_to_tensor(np.array(range(B)), dtype=tf.int32)
    c = tf.expand_dims(c,axis=-1)
    c = tf.tile(c,[1,K])
    c = tf.expand_dims(c,axis=-1)
    #c = tf.cast(c, tf.int32)
    idx = tf.expand_dims(idx, axis=-1)
    new = tf.concat([c,idx], axis=-1)

    select = tf.gather_nd(param, new)   
    return select

def cd_loss(gen, gt, radius, ratio=0.5):
    """ pred: BxNxC,
        label: BxN, """
    dists_forward,idx1, dists_backward,idx2 = tf_nndistance.nn_distance(gt, gen)
    cd_dist = 0.5*dists_forward + 0.5*dists_backward
    cd_dist = tf.reduce_mean(cd_dist, axis=1)
    cd_dist_norm = cd_dist/radius
    cd_loss = tf.reduce_mean(cd_dist_norm)
    return cd_loss, idx1, idx2


def abs_dense_normal_loss(gen_normal, gt_normal, idx1, idx2, radius, ratio=0.5):
    batch_size = gen_normal.get_shape()[0].value

    fwd1 = batch_gather(gen_normal, idx1)

    pos_dist1 = (gt_normal - fwd1) ** 2
    pos_dist1 = tf.reduce_mean(pos_dist1, axis=-1)
    neg_dist1 = (gt_normal + fwd1) ** 2
    neg_dist1 = tf.reduce_mean(neg_dist1, axis=-1)

    dist1 = tf.where(pos_dist1 < neg_dist1, pos_dist1, neg_dist1)
    dist1 = tf.reduce_mean(dist1, axis=1, keep_dims=True)

    fwd2 = batch_gather(gt_normal, idx2)

    pos_dist2 = (gen_normal - fwd2) ** 2
    pos_dist2 = tf.reduce_mean(pos_dist2, axis=-1)
    neg_dist2 = (gen_normal + fwd2) ** 2
    neg_dist2 = tf.reduce_mean(neg_dist2, axis=-1)

    dist2 = tf.where(pos_dist2 < neg_dist2, pos_dist2, neg_dist2)
    dist2 = tf.reduce_mean(dist2, axis=1, keep_dims=True)

    dist = 0.5*dist1 + 0.5*dist2
    dist_norm = dist / radius
    normal_loss = tf.reduce_mean(dist_norm)

    return normal_loss

def abs_sparse_normal_loss(gen_normal, gt_normal, radius):
    batch_size = gen_normal.get_shape()[0].value

    pos_dist = (gt_normal - gen_normal) ** 2
    pos_dist = tf.reduce_mean(pos_dist, axis=-1)

    neg_dist = (gt_normal + gen_normal) ** 2
    neg_dist = tf.reduce_mean(neg_dist, axis=-1)

    dist = tf.where(pos_dist < neg_dist, pos_dist, neg_dist)
    dist = tf.reduce_mean(dist, axis=1, keep_dims=True)

    dist_norm = dist / radius

    normal_loss = tf.reduce_mean(dist_norm)

    return normal_loss

def cd_py(array1,array2):
    batch_size, num_point = array1.shape[:2]
    dist = 0
    for i in range(batch_size):
        tree1 = KDTree(array1[i], leaf_size=num_point+1)
        tree2 = KDTree(array2[i], leaf_size=num_point+1)
        distances1, _ = tree1.query(array2[i])
        distances2, idx2 = tree2.query(array1[i])
        av_dist1 = np.mean(distances1)
        av_dist2 = np.mean(distances2)
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist, idx2