from libs.ops import *


def upsample(value, h, w):
    return tf.image.resize_nearest_neighbor(value, (h, w), align_corners=True)


def downsample(value):
    return tf.layers.average_pooling2d(value, (2, 2), (2, 2), padding='SAME')


class ResNetGenerator(object):
    def __init__(self, hidden_dim=128, batch_size=64, hidden_activation=tf.nn.relu, output_activation=tf.nn.tanh, use_batch_norm=True, z_distribution='normal', scope='generator', **kwargs):
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.z_distribution = z_distribution
        self.scope = scope

    def __call__(self, z, is_training=True, **kwargs):
        def res_block(x, h, w, name):
            with tf.variable_scope(name):
                x_ = batch_norm(x, name='bn1', is_training=is_training)
                x_ = self.hidden_activation(x_)
                x_ = upsample(x_, h, w)
                x_ = conv2d(x_, 256, 3, 3, 1, 1, spectral_normed=False, update_collection=None, stddev=0.02, name='c1')
                x_ = batch_norm(x_, name='bn2', is_training=is_training)
                x_ = self.hidden_activation(x_)
                x_ = conv2d(x_, 256, 3, 3, 1, 1, spectral_normed=False, update_collection=None, stddev=0.02, name='c2')

                x = upsample(x, h, w)
                x = conv2d(x, 256, 1, 1, 1, 1, spectral_normed=False, update_collection=None, stddev=0.02, name='c3')
                return x + x_

        with tf.variable_scope(self.scope):
            if self.use_batch_norm:
                l0 = linear(z, 4 * 4 * 256, name='l0', stddev=0.02)
                l0 = tf.reshape(l0, [self.batch_size, 4, 4, 256])

                dc1 = res_block(l0, 8, 8, name='dc1')
                dc2 = res_block(dc1, 16, 16, name='dc2')
                dc3 = res_block(dc2, 32, 32, name='dc3')

                dc4 = batch_norm(dc3, name='bn', is_training=is_training)
                dc4 = self.hidden_activation(dc4)
                dc4 = conv2d(dc4, 3, 3, 3, 1, 1, spectral_normed=False, update_collection=None, stddev=0.02, name='dc4')
                dc4 = self.output_activation(dc4)
            else:
                raise NotImplementedError
            x = dc4
        return x

    def generate_noise(self):
        if self.z_distribution == 'normal':
            return np.random.randn(self.batch_size, self.hidden_dim).astype(np.float32)
        elif self.z_distribution == 'uniform' :
            return np.random.uniform(-1, 1, (self.batch_size, self.hidden_dim)).astype(np.float32)
        else:
            raise NotImplementedError


class ResNetDiscrminator(object):
    def __init__(self, batch_size=64, hidden_activation=lrelu, output_dim=1, scope='critic', **kwargs):
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim
        self.scope = scope

    def __call__(self, x, update_collection=tf.GraphKeys.UPDATE_OPS, **kwargs):
        def res_block(x, name, is_downsample=False):
            with tf.variable_scope(name):
                x_ = self.hidden_activation(x)
                x_ = conv2d(x_, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1')
                x_ = self.hidden_activation(x_)
                x_ = conv2d(x_, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2')

                if is_downsample is True:
                    x_ = downsample(x_)
                    x = conv2d(x, 128, 1, 1, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c3')
                    x = downsample(x)
                return x + x_

        with tf.variable_scope(self.scope):
            b1_ = conv2d(x, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c1')
            b1_ = self.hidden_activation(b1_)
            b1_ = conv2d(b1_, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c2')
            b1_ = downsample(b1_)

            b1 = downsample(x)
            b1 = conv2d(b1, 128, 1, 1, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='c3')

            b1 = b1 + b1_

            b2 = res_block(b1, name='b2', is_downsample=True)
            b3 = res_block(b2, name='b3', is_downsample=False)
            b4 = res_block(b3, name='b4', is_downsample=False)
            b4 = tf.nn.relu(b4)
            b4 = tf.reduce_sum(b4, axis=(1, 2))

            l4 = linear(b4, self.output_dim, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='l4')
        return tf.reshape(l4, [-1])
