import timeit

import numpy as np
import tensorflow as tf

from libs.fid import calculate_activation_statistics, calculate_frechet_distance
from libs.input_helper import Cifar10
from libs.utils import save_images, mkdir
from net import DCGANGenerator, SNDCGAN_Discrminator
import _pickle as pickle
from libs.inception_score.model import get_inception_score
from resnet import ResNetGenerator, ResNetDiscrminator

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('max_iter', 100000, '')
flags.DEFINE_integer('snapshot_interval', 1000, 'interval of snapshot')
flags.DEFINE_integer('evaluation_interval', 10000, 'interval of evalution')
flags.DEFINE_integer('display_interval', 100, 'interval of displaying log to console')
flags.DEFINE_float('adam_alpha', 0.0001, 'learning rate')
flags.DEFINE_float('adam_beta1', 0.0, 'beta1 in Adam')
flags.DEFINE_float('adam_beta2', 0.9, 'beta2 in Adam')
flags.DEFINE_integer('n_dis', 5, 'n discrminator train')
flags.DEFINE_boolean('isResNet', False, '')

INCEPTION_FILENAME = 'inception_score.pkl'
FID_FILENAME = 'FID_score.pkl'
config = FLAGS.flag_values_dict()
data_set = Cifar10(batch_size=FLAGS.batch_size)

if FLAGS.isResNet is False:
    tmp_dir = 'tmp'
    snapshot_dir = 'snapshots'
    generator = DCGANGenerator(**config)
    discriminator = SNDCGAN_Discrminator(**config)
else:
    tmp_dir = 'resnet_tmp'
    snapshot_dir = 'resnet_snapshots'
    generator = ResNetGenerator(**config)
    discriminator = ResNetDiscrminator(**config)

INCEPTION_FILENAME = tmp_dir + '/' + INCEPTION_FILENAME
FID_FILENAME = tmp_dir + '/' + FID_FILENAME

mkdir(tmp_dir)

global_step = tf.Variable(0, name="global_step", trainable=False)
increase_global_step = global_step.assign(global_step + 1)
is_training = tf.placeholder(tf.bool, shape=())
z = tf.placeholder(tf.float32, shape=[None, generator.generate_noise().shape[1]])
x_hat = generator(z, is_training=is_training)
x = tf.placeholder(tf.float32, shape=x_hat.shape)

d_fake = discriminator(x_hat, update_collection=None)
# Don't need to collect on the second call, put NO_OPS
d_real = discriminator(x, update_collection="NO_OPS")
# Softplus at the end as in the official code of author at chainer-gan-lib github repository
# d_loss = tf.reduce_mean(tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real)) + 1e-3 * tf.reduce_mean(tf.get_collection('partialL2'))
d_loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real)) + tf.reduce_mean(tf.nn.relu(1.0 + d_fake))
g_loss = -tf.reduce_mean(d_fake)
d_loss_summary_op = tf.summary.scalar('d_loss', d_loss)
g_loss_summary_op = tf.summary.scalar('g_loss', g_loss)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(snapshot_dir)

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
d_optimizer = tf.train.AdamOptimizer(learning_rate=4.*FLAGS.adam_alpha, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2)
g_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_alpha, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2)
d_gvs = d_optimizer.compute_gradients(d_loss, var_list=d_vars)
g_gvs = g_optimizer.compute_gradients(g_loss, var_list=g_vars)
d_solver = d_optimizer.apply_gradients(d_gvs)
g_solver = g_optimizer.apply_gradients(g_gvs)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if tf.train.latest_checkpoint(snapshot_dir) is not None:
    saver.restore(sess, tf.train.latest_checkpoint(snapshot_dir))

np.random.seed(1337)
sample_noise = generator.generate_noise()
np.random.seed()
iteration = sess.run(global_step)
start = timeit.default_timer()

is_start_iteration = True
inception_scores = []
fid_scores = []
while iteration < FLAGS.max_iter:
    _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={z: generator.generate_noise(), is_training: True})
    for _ in range(FLAGS.n_dis):
        _, d_loss_curr, summaries = sess.run([d_solver, d_loss, merged_summary_op],
                                             feed_dict={x: data_set.get_next_batch(), z: generator.generate_noise(), is_training: True})
    # increase global step after updating G and D
    # before saving the model so that it will be written into the ckpt file
    sess.run(increase_global_step)
    if (iteration + 1) % FLAGS.display_interval == 0 and not is_start_iteration:
        summary_writer.add_summary(summaries, global_step=iteration)
        stop = timeit.default_timer()
        print('Iter {}: d_loss = {:4f}, g_loss = {:4f}, time = {:2f}s'.format(iteration, d_loss_curr, g_loss_curr, stop - start))
        start = stop
    if (iteration + 1) % FLAGS.snapshot_interval == 0 and not is_start_iteration:
        saver.save(sess, snapshot_dir+'/model.ckpt', global_step=iteration)
        sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
        save_images(sample_images, tmp_dir+'/{:06d}.png'.format(iteration))
    if (iteration + 1) % FLAGS.evaluation_interval == 0:
        sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
        save_images(sample_images, tmp_dir+'{:06d}.png'.format(iteration))
        # Sample 50000 images for evaluation
        print("Evaluating...")
        num_images_to_eval = 50000
        eval_images = []
        num_batches = num_images_to_eval // FLAGS.batch_size + 1
        print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
        np.random.seed(0)
        for _ in range(num_batches):
            images = sess.run(x_hat, feed_dict={z: generator.generate_noise(), is_training: False})
            eval_images.append(images)
        np.random.seed()
        eval_images = np.vstack(eval_images)
        eval_images = eval_images[:num_images_to_eval]
        eval_images = np.clip((eval_images + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        # Calc Inception score
        eval_images = list(eval_images)
        inception_score_mean, inception_score_std, generated_images_activation = get_inception_score(eval_images)
        print("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std))
        inception_scores.append(dict(mean=inception_score_mean, std=inception_score_std))
        with open(INCEPTION_FILENAME, 'wb') as f:
            pickle.dump(inception_scores, f)

        true_images = []
        for _ in range(num_batches):
            true_images.append(data_set.get_next_batch())
        true_images = np.vstack(true_images)
        true_images = true_images[:num_images_to_eval]
        true_images = np.clip((true_images + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        true_images = list(true_images)
        true_inception_score_mean, true_inception_score_std, true_images_activation = get_inception_score(true_images)
        print("True Image Inception Score: Mean = {} \tStd = {}.".format(true_inception_score_mean, true_inception_score_std))

        true_mu, true_sigma = calculate_activation_statistics(true_images_activation)
        generated_mu, generated_sigma = calculate_activation_statistics(generated_images_activation)
        fid_score = calculate_frechet_distance(true_mu, true_sigma, generated_mu, generated_sigma)
        print("FID Score: {}".format(fid_score))
        fid_scores.append(dict(fid=fid_score))
        with open(FID_FILENAME, 'wb') as f:
            pickle.dump(fid_scores, f)

    iteration += 1
    is_start_iteration = False
