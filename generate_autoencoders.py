import math
import time
import random

random.seed(11)
import numpy as np

np.random.seed(11)
import pandas as pd
import tensorflow as tf

tf.set_random_seed(11)
from sklearn.utils import shuffle
from tqdm import tqdm
from model import Model
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold

datadir= './data/'
datadir_enc ="./autoencoders/"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('numepochs', 60, "")
tf.app.flags.DEFINE_integer('batchsize', 256, "")
tf.app.flags.DEFINE_boolean('denoise', True, "")  # compute 2 results

if FLAGS.denoise:
    print('Denoising!')
else:
    print('NOT denoising!')

tour = pd.read_csv(datadir  + 'numerai_tournament_data.csv')
df_train = pd.read_csv(datadir  + 'numerai_training_data.csv')
df_valid = tour[tour['data_type'].isin(['validation'])]

df_live = tour[tour['data_type'].isin(['live'])]
df_test = tour[tour['data_type'].isin(['test'])]

feature_cols = [f for f in df_train.columns if "feature" in f]
X_train = df_train[feature_cols].values
X_valid = df_valid[feature_cols].values

X_test = df_test[feature_cols].values
X_live = df_live[feature_cols].values

num_features = len(feature_cols)
features = tf.placeholder(tf.float32, shape=[None, num_features], name='features')

with tf.variable_scope('model'):
    train_model = Model(features, denoise=FLAGS.denoise, is_training=True)

with tf.variable_scope('model', reuse=True):
    test_model = Model(features, denoise=FLAGS.denoise, is_training=False)

best = None
wait = 0
summary_op = tf.summary.merge_all()  # merge_all_summaries()
logdir = 'logs/{}'.format(int(time.time()))
supervisor = tf.train.Supervisor(logdir=logdir, summary_op=None)
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
with supervisor.managed_session() as sess:
    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    print('Training model with {} parameters...'.format(train_model.num_parameters))
    with tqdm(total=FLAGS.numepochs) as pbar:
        for epoch in range(FLAGS.numepochs):
            X_train_epoch = shuffle(X_train)
            num_batches = len(X_train_epoch) // FLAGS.batchsize
            losses = []
            for batch_index in range(num_batches):
                batch_start = batch_index * FLAGS.batchsize
                batch_end = batch_start + FLAGS.batchsize

                X_train_batch = X_train_epoch[batch_start:batch_end]

                _, loss = sess.run([
                    train_model.train_step,
                    train_model.loss,
                ], feed_dict={
                    features: X_train_batch,
                })
                losses.append(loss)
            loss_train = np.mean(losses)

            loss_valid, summary_str = sess.run([
                test_model.loss,
                summary_op,
            ], feed_dict={
                features: X_valid,
            })
            if best is None or loss_valid < best:
                best = loss_valid
                wait = 0
            else:
                wait += 1
            summary_writer.add_summary(summary_str, epoch)
            summary_writer.flush()
            pbar.set_description('[{}] loss (train): {:.8f}, loss (valid): {:.8f} [best: {:.8f}, wait: {}]' \
                                 .format(epoch, loss_train, loss_valid, best, wait))
            pbar.update()

    summary_writer.close()

    loss_valid = sess.run(test_model.loss, feed_dict={
        features: X_valid,
    })
    print('Validation loss: {}'.format(loss_valid))

    t_Group_eras = df_train["era"].values
    group_kfold = GroupKFold(n_splits=10)
    z_train = np.zeros([len(X_train), 32])
    for f, (train_index, test_index) in enumerate(group_kfold.split(X_train, None, t_Group_eras)):
        x_train = X_train[test_index]

        z_train_tmp = sess.run(test_model.z, feed_dict={features: x_train})

        z_train[test_index] = z_train_tmp

    v_Group_eras = df_valid["era"].values
    group_kfold2 = GroupKFold(n_splits=10)
    z_valid = np.zeros([len(X_valid), 32])
    for f, (train_index, test_index) in enumerate(group_kfold2.split(X_valid, None, v_Group_eras)):
        x_valid = X_valid[test_index]

        z_valid_tmp = sess.run(test_model.z, feed_dict={features: x_valid})

        z_valid[test_index] = z_valid_tmp

    kf = KFold(n_splits=5)
    z_test = np.zeros([len(X_test), 32])
    for f, (train_index, test_index) in enumerate(kf.split(X_test, None, None)):
        x_test = X_test[test_index]
        # here in test_model.z we have acccess to the 32 layers of the model.
        z_test_tmp = sess.run(test_model.z, feed_dict={features: x_test})

        z_test[test_index] = z_test_tmp

    z_live = sess.run(test_model.z, feed_dict={features: X_live})

    if FLAGS.denoise:
        np.savez(datadir_enc  + 'denoising.npz', z_train=z_train, z_valid=z_valid, z_test=z_test, z_live=z_live)
        print("SavedDEnc")
    else:
        np.savez(datadir_enc  +  'autoencoder.npz', z_train=z_train, z_valid=z_valid, z_test=z_test, z_live=z_live)
        print("SavedEnc")
