# load library
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob
import socket
from datetime import datetime

import utils.data_loader as provider
from utils import loss
from utils.pc_util import extract_knn_patch, normalize_point_cloud
from tf_ops.sampling.tf_sampling import farthest_point_sample

# argparse argument
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train', help='train or test')
parser.add_argument('--gpu',  default='0',  help='which gpu to use')
parser.add_argument('--up_ratio',  type=int,  default=4,   help='Upsampling Ratio')
parser.add_argument('--model', default='model_pugeo', help='Model for upsampling')
parser.add_argument('--num_point', type=int, default=256,help='Point Number')

# for phase train
parser.add_argument('--log_dir', default='PUGeo_x4', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training')
parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--reg_normal1', type=float, default=1.0)
parser.add_argument('--reg_normal2', type=float, default=1.0)
parser.add_argument('--jitter_sigma', type=float, default=0.01)
parser.add_argument('--jitter_max', type=float, default=0.03)

#for phase test
parser.add_argument('--pretrained', default='', help='Model stored')
parser.add_argument('--eval_xyz', default='test_5000', help='Folder to evaluate')
parser.add_argument('--num_shape_point', type=int, default=5000,help='Point Number per shape')
parser.add_argument('--patch_num_ratio', type=int, default=3,help='Number of points covered by patch')
FLAGS = parser.parse_args()
print(FLAGS)

# load model and other utilities
exec('import model.%s as upsample_model' % FLAGS.model)
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()

def build_path(log_path):
    path = os.path.join('/home/qianyue/data/model/pugeo', log_path)
    model_path  = os.path.join(path, 'model')
    eval_path  = os.path.join(path, 'eval')
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    return path, model_path, eval_path

# main:
def main(arg):
    # define placeholder
    training_pl = tf.placeholder(tf.bool, shape=())

    input_sparse_xyz_pl = tf.placeholder(tf.float32, shape=(arg.batch_size, arg.num_point, 3))
    gt_sparse_normal_pl = tf.placeholder(tf.float32, shape=(arg.batch_size, arg.num_point, 3))
    gt_dense_xyz_pl = tf.placeholder(tf.float32, shape=(arg.batch_size, arg.num_point*arg.up_ratio, 3))
    gt_dense_normal_pl = tf.placeholder(tf.float32, shape=(arg.batch_size, arg.num_point*arg.up_ratio, 3))
    input_r_pl = tf.placeholder(tf.float32, shape=(arg.batch_size))

    shape_sparse_xyz_pl = tf.placeholder(tf.float32, shape=[1, arg.num_shape_point, 3])
    shape_ddense_xyz_pl = tf.placeholder(tf.float32, shape=[1, arg.num_point*arg.up_ratio*arg.num_patch, 3])

    # generated point cloud
    gen_dense_xyz, gen_dense_normal, gen_sparse_normal = upsample_model.get_model(input_sparse_xyz_pl, arg.up_ratio, training_pl, knn=30, bradius=input_r_pl, scope='generator')
    
    # fps index
    fps_idx1 = farthest_point_sample(arg.num_patch, shape_sparse_xyz_pl)
    fps_idx2 = farthest_point_sample(arg.num_shape_point*arg.up_ratio, shape_ddense_xyz_pl)

    # loss function
    loss_dense_cd, cd_idx1, cd_idx2 = loss.cd_loss(gen_dense_xyz, gt_dense_xyz_pl, input_r_pl)
    loss_dense_normal = loss.abs_dense_normal_loss(gen_dense_normal, gt_dense_normal_pl, cd_idx1, cd_idx2, input_r_pl)
    loss_sparse_normal = loss.abs_sparse_normal_loss(gen_sparse_normal, gt_sparse_normal_pl, input_r_pl)

    loss_all = 100 * loss_dense_cd  + arg.reg_normal1* loss_dense_normal + arg.reg_normal2* loss_sparse_normal + tf.losses.get_regularization_loss()
    
    # optimizer
    bn_decay = 0.95
    step = tf.Variable(0,trainable=False)

    gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    param_size = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    with tf.control_dependencies(gen_update_ops):
        train_op = tf.train.AdamOptimizer(arg.learning_rate,beta1=0.9).minimize(loss_all,var_list=gen_tvars, global_step=step)
    
    # load data
    dataloader = provider.Fetcher(arg.train_record, batch_size=arg.batch_size, step_ratio=arg.up_ratio, up_ratio=arg.up_ratio, num_in_point=arg.num_point, num_shape_point=arg.num_shape_point, jitter=True, drop_out=1.0, jitter_max=arg.jitter_max, jitter_sigma=arg.jitter_sigma)

    # define ops
    ops = {'training_pl': training_pl,
            'input_sparse_xyz_pl': input_sparse_xyz_pl,
            'gt_sparse_normal_pl': gt_sparse_normal_pl,
            'gt_dense_xyz_pl': gt_dense_xyz_pl,
            'gt_dense_normal_pl': gt_dense_normal_pl,
            'input_r_pl': input_r_pl,
            'shape_sparse_xyz_pl': shape_sparse_xyz_pl,
            'shape_ddense_xyz_pl': shape_ddense_xyz_pl,
            'gen_dense_xyz': gen_dense_xyz,
            'gen_dense_normal': gen_dense_normal,
            'gen_sparse_normal': gen_sparse_normal,
            'fps_idx1': fps_idx1,
            'fps_idx2': fps_idx2,
            'loss_dense_cd': loss_dense_cd,
            'loss_dense_normal': loss_dense_normal,
            'loss_sparse_normal': loss_sparse_normal,
            'loss_all': loss_all,
            'train_op': train_op,
            'step': step
        }

    # create sess
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init, {training_pl: True})
        saver = tf.train.Saver(max_to_keep=10)
        tf.get_default_graph().finalize()

        # if train phase
        if arg.phase == 'train':
            # loop for epoch
            for epoch in tqdm(range(arg.max_epoch+1)):
                train_one_epoch(arg, sess, ops, dataloader)
            # save model
            saver.save(sess, os.path.join(arg.model_path, "model"), global_step=epoch)
            # save xyz files
            eval_shapes(arg, sess, ops, arg.up_ratio, arg.eval_xyz)

        # if eval phase
        if arg.phase == 'test':
            # load model
            saver.restore(sess, arg.pretrained)
            # save xyz files
            eval_shapes(arg, sess, ops, arg.up_ratio, arg.eval_xyz)


# train_one_epoch
def train_one_epoch(arg, sess, ops, dataloader):
    is_training = True
    loss_sum_all = []
    loss_sum_dense_cd = []
    loss_sum_dense_normal = []
    loss_sum_sparse_normal = []

    dataloader.initialize(sess, arg.up_ratio, False)

    # load data, form feed_dict
    for batch_idx in range(dataloader.num_batches):
        input_sparse, gt_dense, input_r,_ = dataloader.fetch(sess)  
 
        # TODO: move normalization in data loading
        input_sparse_xyz = input_sparse[:,:,0:3]
        input_sparse_normal = input_sparse[:,:,3:6]
        sparse_l2 = np.linalg.norm(input_sparse_normal, axis=-1, keepdims=True)
        sparse_l2 = np.tile(sparse_l2, [1,3])
        input_sparse_normal = np.divide(input_sparse_normal, sparse_l2)

        gt_dense_xyz = gt_dense[:,:,0:3]
        gt_dense_normal = gt_dense[:,:,3:6]
        dense_l2 = np.linalg.norm(gt_dense_normal, axis=-1, keepdims=True)
        dense_l2 = np.tile(dense_l2, [1,3])
        gt_dense_normal = np.divide(gt_dense_normal,dense_l2)
        
        feed_dict = {ops['training_pl']: is_training,
                     ops['input_sparse_xyz_pl']: input_sparse_xyz,
                     ops['gt_sparse_normal_pl']: input_sparse_normal,
                     ops['gt_dense_xyz_pl']: gt_dense_xyz,
                     ops['gt_dense_normal_pl']: gt_dense_normal,
                     ops['input_r_pl']: input_r
                    }

        # get loss, print
        step, _, loss_all,loss_dense_cd, loss_dense_normal, loss_sparse_normal = sess.run([ops['step'],ops['train_op'],ops['loss_all'], ops['loss_dense_cd'], ops['loss_dense_normal'], ops['loss_sparse_normal']], feed_dict=feed_dict)
    
        loss_sum_all.append(loss_all)
        loss_sum_dense_cd.append(loss_dense_cd)
        loss_sum_dense_normal.append(loss_dense_normal)
        loss_sum_sparse_normal.append(loss_sparse_normal)

    loss_sum_all = np.asarray(loss_sum_all)
    loss_sum_dense_cd = np.asarray(loss_sum_dense_cd)
    loss_sum_dense_normal = np.asarray(loss_sum_dense_normal)
    loss_sum_sparse_normal = np.asarray(loss_sum_sparse_normal)
    log_string('step: %d total loss: %f, cd: %f, dense normal: %f, sparse normal: %f\n' % (step, round(loss_sum_all.mean(),7), round(loss_sum_dense_cd.mean(),7), round(loss_sum_dense_normal.mean(),7), round(loss_sum_sparse_normal.mean(),7)))    

def eval_per_patch(input_sparse_xyz, sess, arg, ops):
    is_training = False

    # normalize patch
    normalize_sparse_xyz, centroid, furthest_distance = normalize_point_cloud(input_sparse_xyz)
    normalize_sparse_xyz = np.expand_dims(normalize_sparse_xyz,axis=0)
    batch_normalize_sparse_xyz = np.tile(normalize_sparse_xyz, [arg.batch_size, 1, 1])

    # feed_dict and return result
    gen_dense_xyz, gen_dense_normal, gen_sparse_normal = sess.run([ops['gen_dense_xyz'], ops['gen_dense_normal'], ops['gen_sparse_normal']],
        feed_dict={ops['input_sparse_xyz_pl']: batch_normalize_sparse_xyz, 
            ops['training_pl']: is_training,
            ops['input_r_pl']: np.ones([arg.batch_size], dtype='f')
    })

    gen_dense_xyz = np.expand_dims(gen_dense_xyz[0], axis=0)
    gen_dense_xyz = np.squeeze(centroid + gen_dense_xyz * furthest_distance, axis=0)
    gen_dense_normal = gen_dense_normal[0]
    gen_sparse_normal = gen_sparse_normal[0]
    return gen_dense_xyz, gen_dense_normal, gen_sparse_normal


def eval_patches(xyz, sess, arg, ops):
    # get patch
    fps_idx1 = sess.run(ops['fps_idx1'], feed_dict={ops['shape_sparse_xyz_pl']:np.expand_dims(xyz,axis=0)})
    fps_idx1 = fps_idx1[:arg.num_patch]
    fps_idx1 = fps_idx1[0]

    input_sparse_xyz_list = []
    gen_dense_xyz_list=[]
    gen_dense_normal_list = []
    gen_sparse_normal_list = []

    patches = extract_knn_patch(xyz[np.asarray(fps_idx1), :], xyz, arg.num_point)

    # loop patch
    for input_sparse_xyz in tqdm(patches, total=len(patches)):
        # get prediction
        gen_dense_xyz, gen_dense_normal, gen_sparse_normal = eval_per_patch(input_sparse_xyz, sess, arg, ops)

        input_sparse_xyz_list.append(input_sparse_xyz)
        gen_dense_xyz_list.append(gen_dense_xyz)
        gen_dense_normal_list.append(gen_dense_normal)
        gen_sparse_normal_list.append(gen_sparse_normal)
    return input_sparse_xyz_list, gen_dense_xyz_list, gen_dense_normal_list, gen_sparse_normal_list 

def eval_shapes(arg, sess, ops, up_ratio, eval_xyz):
    # loop files
    shapes = glob(eval_xyz + "/*.xyz")
    for i,item in enumerate(shapes):
        obj_name = item.split('/')[-1] # with .xyz
        data = np.loadtxt(item)
        input_sparse_xyz = data[:, 0:3]
        input_sparse_normal = data[:, 3:6]

        # normalize point cloud
        normalize_sparse_xyz, centroid, furthest_distance = normalize_point_cloud(input_sparse_xyz)
        
        # get patchwise prediction
        input_sparse_xyz_list, gen_dense_xyz_list, gen_dense_normal_list, gen_sparse_normal_list = eval_patches(normalize_sparse_xyz, sess, arg, ops)

        # un-normalize
        gen_ddense_xyz = np.concatenate(gen_dense_xyz_list, axis=0)
        gen_ddense_xyz = (gen_ddense_xyz*furthest_distance) + centroid
        gen_ddense_normal = np.concatenate(gen_dense_normal_list, axis=0)

        # formulate to fps point
        fps_idx2 = sess.run(ops['fps_idx2'], feed_dict={ops['shape_ddense_xyz_pl']:gen_ddense_xyz[np.newaxis,...]})
        fps_idx2 = fps_idx2[0]
        gen_ddense = np.concatenate([gen_ddense_xyz, gen_ddense_normal], axis=-1)
        gen_dense = gen_ddense[fps_idx2,0:6]

        # save pc
        path = os.path.join(arg.eval_path, obj_name)
        np.savetxt(path, np.squeeze(gen_dense))


if __name__ == "__main__":
    if FLAGS.phase == 'test':
        assert FLAGS.pretrained is not None
        FLAGS.log_dir = os.path.dirname(os.path.dirname(FLAGS.pretrained))
    FLAGS.log_dir, FLAGS.model_path, FLAGS.eval_path = build_path(FLAGS.log_dir)

    global LOG_FOUT
    LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log.txt'),'w')
    LOG_FOUT.write(str(datetime.now())+ '\n')
    LOG_FOUT.write(os.path.abspath(__file__) + '\n')
    LOG_FOUT.write(str(FLAGS) + '\n') 

    FLAGS.train_record = 'tfrecord_x%d_normal/*.tfrecord' % FLAGS.up_ratio
    FLAGS.num_patch = int(FLAGS.num_shape_point / FLAGS.num_point * FLAGS.patch_num_ratio)

    main(FLAGS)
    
