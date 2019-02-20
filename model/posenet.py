import numpy as np
import tensorflow as tf


class PoseNet(object):
    def __init__(self, x, weights_path):
        self.X = x
        self.WEIGHTS_PATH = weights_path

        self.create()

    def create(self):
        # convolution 1
        with tf.variable_scope("conv1"):
            conv1 = tf.layers.conv2d(self.X, filters=64, kernel_size=7, strides=2, padding="same",
                                     activation=tf.nn.relu, use_bias=True, name="7x7_s2")
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=2, padding="same", name="pool1_3x3_s2")
            pool1_norm = tf.nn.lrn(pool1, 2, 1.0, 2e-05, 0.75, name="pool1_norm")

        # convolution 2
        with tf.variable_scope("conv2"):
            conv2_reduce = tf.layers.conv2d(pool1_norm, filters=64, kernel_size=1, strides=1, padding="valid",
                                            activation=tf.nn.relu,
                                            use_bias=True, name="3x3_reduce")
            conv2 = tf.layers.conv2d(conv2_reduce, filters=192, kernel_size=3, strides=1, padding="same",
                                     activation=tf.nn.relu, use_bias=True, name="3x3")
            conv2_norm = tf.nn.lrn(conv2, 2, 1.0, 2e-05, 0.75, name="pool2_norm")
            pool2 = tf.layers.max_pooling2d(conv2_norm, pool_size=3, strides=2, padding="same",
                                            name="pool2_3x3_s2")

        # inception 3a
        with tf.variable_scope("inception_3a"):
            inception_3a_1x1 = tf.layers.conv2d(pool2, filters=64, kernel_size=1, strides=1, padding="same",
                                                activation=tf.nn.relu,
                                                use_bias=True, name="1x1")
            inception_3a_3x3_reduce = tf.layers.conv2d(pool2, filters=96, kernel_size=1, strides=1, padding="same",
                                                       activation=tf.nn.relu,
                                                       use_bias=True, name="3x3_reduce")
            inception_3a_3x3 = tf.layers.conv2d(inception_3a_3x3_reduce, filters=128, kernel_size=3, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="3x3")
            inception_3a_5x5_reduce = tf.layers.conv2d(pool2, filters=16, kernel_size=1, strides=1, padding="same",
                                                       activation=tf.nn.relu,
                                                       use_bias=True, name="5x5_reduce")
            inception_3a_5x5 = tf.layers.conv2d(inception_3a_5x5_reduce, filters=32, kernel_size=5, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="5x5")
            inception_3a_max_pool = tf.layers.max_pooling2d(pool2, pool_size=3, strides=1, padding="same",
                                                            name="pool3_3x3_s1")
            inception_3a_pool_proj = tf.layers.conv2d(inception_3a_max_pool, filters=32, kernel_size=1, strides=1,
                                                      padding="same", activation=tf.nn.relu,
                                                      use_bias=True, name="pool_proj")

            inception_3a_output = tf.concat([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5,
                                             inception_3a_pool_proj], 3)

        # inception 3b
        with tf.variable_scope("inception_3b"):
            inception_3b_1x1 = tf.layers.conv2d(inception_3a_output, filters=128, kernel_size=1, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="1x1")
            inception_3b_3x3_reduce = tf.layers.conv2d(inception_3a_output, filters=128, kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="3x3_reduce")
            inception_3b_3x3 = tf.layers.conv2d(inception_3b_3x3_reduce, filters=192, kernel_size=3, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="3x3")
            inception_3b_5x5_reduce = tf.layers.conv2d(inception_3a_output, filters=32, kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="5x5_reduce")
            inception_3b_5x5 = tf.layers.conv2d(inception_3b_5x5_reduce, filters=96, kernel_size=5, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="5x5")
            inception_3b_max_pool = tf.layers.max_pooling2d(inception_3a_output, pool_size=3, strides=1,
                                                            padding="same", name="pool3_3x3_s1")
            inception_3b_pool_proj = tf.layers.conv2d(inception_3b_max_pool, filters=64, kernel_size=1, strides=1,
                                                      padding="same", activation=tf.nn.relu,
                                                      use_bias=True, name="pool_proj")
            inception_3b_output = tf.concat([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5,
                                             inception_3b_pool_proj], 3)

        max_pool_between_inceptions_3b_4a = tf.layers.max_pooling2d(inception_3b_output, pool_size=3, strides=2,
                                                                    padding="same", name="pool_intermediate_3b_4a")

        # inception 4a
        with tf.variable_scope("inception_4a"):
            inception_4a_1x1 = tf.layers.conv2d(max_pool_between_inceptions_3b_4a, filters=192, kernel_size=1,
                                                strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="1x1")
            inception_4a_3x3_reduce = tf.layers.conv2d(max_pool_between_inceptions_3b_4a, filters=96,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="3x3_reduce")
            inception_4a_3x3 = tf.layers.conv2d(inception_4a_3x3_reduce, filters=208, kernel_size=3, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="3x3")
            inception_4a_5x5_reduce = tf.layers.conv2d(max_pool_between_inceptions_3b_4a, filters=16,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="5x5_reduce")
            inception_4a_5x5 = tf.layers.conv2d(inception_4a_5x5_reduce, filters=48, kernel_size=5, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="5x5")
            inception_4a_max_pool = tf.layers.max_pooling2d(max_pool_between_inceptions_3b_4a, pool_size=3,
                                                            strides=1,
                                                            padding="same", name="pool4_3x3_s1")
            inception_4a_pool_proj = tf.layers.conv2d(inception_4a_max_pool, filters=64, kernel_size=1, strides=1,
                                                      padding="same", activation=tf.nn.relu,
                                                      use_bias=True, name="pool_proj")
            inception_4a_output = tf.concat([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5,
                                             inception_4a_pool_proj], 3)

        # inception 4b
        with tf.variable_scope("inception_4b"):
            inception_4b_1x1 = tf.layers.conv2d(inception_4a_output, filters=160, kernel_size=1,
                                                strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="1x1")
            inception_4b_3x3_reduce = tf.layers.conv2d(inception_4a_output, filters=112,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="3x3_reduce")
            inception_4b_3x3 = tf.layers.conv2d(inception_4b_3x3_reduce, filters=224, kernel_size=3, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="3x3")
            inception_4b_5x5_reduce = tf.layers.conv2d(inception_4a_output, filters=24,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="5x5_reduce")
            inception_4b_5x5 = tf.layers.conv2d(inception_4b_5x5_reduce, filters=64, kernel_size=5, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="5x5")
            inception_4b_max_pool = tf.layers.max_pooling2d(inception_4a_output, pool_size=3,
                                                            strides=1,
                                                            padding="same", name="pool4_3x3_s1")
            inception_4b_pool_proj = tf.layers.conv2d(inception_4b_max_pool, filters=64, kernel_size=1, strides=1,
                                                      padding="same", activation=tf.nn.relu,
                                                      use_bias=True, name="pool_proj")
            inception_4b_output = tf.concat([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5,
                                             inception_4b_pool_proj], 3)

        # inception 4c
        with tf.variable_scope("inception_4c"):
            inception_4c_1x1 = tf.layers.conv2d(inception_4b_output, filters=128, kernel_size=1,
                                                strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="1x1")
            inception_4c_3x3_reduce = tf.layers.conv2d(inception_4b_output, filters=128,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="3x3_reduce")
            inception_4c_3x3 = tf.layers.conv2d(inception_4c_3x3_reduce, filters=256, kernel_size=3, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="3x3")
            inception_4c_5x5_reduce = tf.layers.conv2d(inception_4b_output, filters=24,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="5x5_reduce")
            inception_4c_5x5 = tf.layers.conv2d(inception_4c_5x5_reduce, filters=64, kernel_size=5, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="5x5")
            inception_4c_max_pool = tf.layers.max_pooling2d(inception_4b_output, pool_size=3,
                                                            strides=1,
                                                            padding="same", name="pool4_3x3_s1")
            inception_4c_pool_proj = tf.layers.conv2d(inception_4c_max_pool, filters=64, kernel_size=1, strides=1,
                                                      padding="same", activation=tf.nn.relu,
                                                      use_bias=True, name="pool_proj")
            inception_4c_output = tf.concat([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5,
                                             inception_4c_pool_proj], 3)

        # inception 4d
        with tf.variable_scope("inception_4d"):
            inception_4d_1x1 = tf.layers.conv2d(inception_4c_output, filters=112, kernel_size=1,
                                                strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="1x1")
            inception_4d_3x3_reduce = tf.layers.conv2d(inception_4c_output, filters=144,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="3x3_reduce")
            inception_4d_3x3 = tf.layers.conv2d(inception_4d_3x3_reduce, filters=288, kernel_size=3, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="3x3")
            inception_4d_5x5_reduce = tf.layers.conv2d(inception_4c_output, filters=32,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="5x5_reduce")
            inception_4d_5x5 = tf.layers.conv2d(inception_4d_5x5_reduce, filters=64, kernel_size=5, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="5x5")
            inception_4d_max_pool = tf.layers.max_pooling2d(inception_4c_output, pool_size=3,
                                                            strides=1,
                                                            padding="same", name="pool4_3x3_s1")
            inception_4d_pool_proj = tf.layers.conv2d(inception_4d_max_pool, filters=64, kernel_size=1, strides=1,
                                                      padding="same", activation=tf.nn.relu,
                                                      use_bias=True, name="pool_proj")
            inception_4d_output = tf.concat([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5,
                                             inception_4d_pool_proj], 3)

        # inception 4e
        with tf.variable_scope("inception_4e"):
            inception_4e_1x1 = tf.layers.conv2d(inception_4d_output, filters=256, kernel_size=1,
                                                strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="1x1")
            inception_4e_3x3_reduce = tf.layers.conv2d(inception_4d_output, filters=160,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="3x3_reduce")
            inception_4e_3x3 = tf.layers.conv2d(inception_4e_3x3_reduce, filters=320, kernel_size=3, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="3x3")
            inception_4e_5x5_reduce = tf.layers.conv2d(inception_4d_output, filters=32,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="5x5_reduce")
            inception_4e_5x5 = tf.layers.conv2d(inception_4e_5x5_reduce, filters=128, kernel_size=5, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="5x5")
            inception_4e_max_pool = tf.layers.max_pooling2d(inception_4d_output, pool_size=3,
                                                            strides=1,
                                                            padding="same", name="pool4_3x3_s1")
            inception_4e_pool_proj = tf.layers.conv2d(inception_4e_max_pool, filters=128, kernel_size=1, strides=1,
                                                      padding="same", activation=tf.nn.relu,
                                                      use_bias=True, name="pool_proj")
            inception_4e_output = tf.concat([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5,
                                             inception_4e_pool_proj], 3)

        max_pool_between_inceptions_4e_5a = tf.layers.max_pooling2d(inception_4e_output, pool_size=3, strides=2,
                                                                    padding="same", name="pool_intermediate_4e_5a")

        # inception 5a
        with tf.variable_scope("inception_5a"):
            inception_5a_1x1 = tf.layers.conv2d(max_pool_between_inceptions_4e_5a, filters=256, kernel_size=1,
                                                strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="1x1")
            inception_5a_3x3_reduce = tf.layers.conv2d(max_pool_between_inceptions_4e_5a, filters=160,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="3x3_reduce")
            inception_5a_3x3 = tf.layers.conv2d(inception_5a_3x3_reduce, filters=320, kernel_size=3, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="3x3")
            inception_5a_5x5_reduce = tf.layers.conv2d(max_pool_between_inceptions_4e_5a, filters=32,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="5x5_reduce")
            inception_5a_5x5 = tf.layers.conv2d(inception_5a_5x5_reduce, filters=128, kernel_size=5, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="5x5")
            inception_5a_max_pool = tf.layers.max_pooling2d(max_pool_between_inceptions_4e_5a, pool_size=3,
                                                            strides=1,
                                                            padding="same", name="pool4_3x3_s1")
            inception_5a_pool_proj = tf.layers.conv2d(inception_5a_max_pool, filters=128, kernel_size=1, strides=1,
                                                      padding="same", activation=tf.nn.relu,
                                                      use_bias=True, name="pool_proj")
            inception_5a_output = tf.concat([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5,
                                             inception_5a_pool_proj], 3)

        # inception 5b
        with tf.variable_scope("inception_5b"):
            inception_5b_1x1 = tf.layers.conv2d(inception_5a_output, filters=384, kernel_size=1,
                                                strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="1x1")
            inception_5b_3x3_reduce = tf.layers.conv2d(inception_5a_output, filters=192,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="3x3_reduce")
            inception_5b_3x3 = tf.layers.conv2d(inception_5b_3x3_reduce, filters=384, kernel_size=3, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="3x3")
            inception_5b_5x5_reduce = tf.layers.conv2d(inception_5a_output, filters=48,
                                                       kernel_size=1, strides=1,
                                                       padding="same", activation=tf.nn.relu,
                                                       use_bias=True, name="5x5_reduce")
            inception_5b_5x5 = tf.layers.conv2d(inception_5b_5x5_reduce, filters=128, kernel_size=5, strides=1,
                                                padding="same", activation=tf.nn.relu,
                                                use_bias=True, name="5x5")
            inception_5b_max_pool = tf.layers.max_pooling2d(inception_5a_output, pool_size=3,
                                                            strides=1,
                                                            padding="same", name="pool4_3x3_s1")
            inception_5b_pool_proj = tf.layers.conv2d(inception_5b_max_pool, filters=128, kernel_size=1, strides=1,
                                                      padding="same", activation=tf.nn.relu,
                                                      use_bias=True, name="pool_proj")
            inception_5b_output = tf.concat([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5,
                                             inception_5b_pool_proj], 3)

        final_avg_pool = tf.layers.average_pooling2d(inception_5b_output, pool_size=7, strides=1, padding="valid",
                                                     name="final_avg_pool")

        final_dropout = tf.layers.dropout(final_avg_pool, rate=0.4, seed=42, name="final_dropout")

        with tf.variable_scope("loss3"):
            googlenet_fc = tf.layers.dense(final_dropout, 1000, activation=tf.nn.relu, use_bias=True,
                                           name="classifier")

        posenet_fc = tf.layers.dense(googlenet_fc, 2048, activation=tf.nn.relu, use_bias=True,
                                     kernel_initializer=tf.initializers.he_normal(), trainable=True,
                                     bias_initializer=tf.constant_initializer(1),
                                     name="posenet_fc")

        self.posenet_output_pose = tf.layers.dense(inputs=posenet_fc, units=3, activation=None, use_bias=True,
                                                   kernel_initializer=tf.initializers.he_normal(), trainable=True,
                                                   bias_initializer=tf.constant_initializer(1),
                                                   name="posenet_output_pose")
        self.posenet_output_rotation = tf.layers.dense(inputs=posenet_fc, units=4, activation=None, use_bias=True,
                                                       kernel_initializer=tf.initializers.he_normal(), trainable=True,
                                                       bias_initializer=tf.constant_initializer(1),
                                                       name="posenet_output_rotation")

    def load_initial_weights(self, session):
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='latin1').item()

        for op_name in weights_dict:
            parsed_op_name = op_name.split("_")
            if "inception" in parsed_op_name:
                scope_name = "_".join(parsed_op_name[:2])
                var_name = "_".join(parsed_op_name[2:])
            else:
                scope_name = "_".join(parsed_op_name[:1])
                var_name = "_".join(parsed_op_name[1:])
            with tf.variable_scope(scope_name, reuse=True):
                var = tf.get_variable(var_name + '/bias', trainable=False)
                session.run(var.assign(weights_dict[op_name]["biases"]))

                var = tf.get_variable(var_name + '/kernel', trainable=False)
                session.run(var.assign(weights_dict[op_name]["weights"]))
