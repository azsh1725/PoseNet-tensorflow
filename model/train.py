import tensorflow as tf
import numpy as np
from data.data_generator import get_train_and_test_iterator
from model.posenet import PoseNet

EPOCH_NUM = 100
BATCH_SIZE = 32
CHECKPOINT_PATH = "../checkpoints/"

tf.reset_default_graph()

train_iterator, test_iterator = get_train_and_test_iterator(BATCH_SIZE)

x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="x")
y_pose = tf.placeholder(tf.float32, [None, 1, 1, 3], name="y_pose")
y_rotation = tf.placeholder(tf.float32, [None, 1, 1, 4], name="y_rotation")

posenet_model = PoseNet(x, "./pretrained_models/googlenet.npy")

pose_preditction = posenet_model.posenet_output_pose
rotation_prediction = posenet_model.posenet_output_rotation

pose_loss = tf.norm(tf.subtract(posenet_model.posenet_output_pose, y_pose), 2)
rotation_loss = tf.norm(tf.subtract(posenet_model.posenet_output_rotation, tf.divide(y_rotation, tf.norm(y_rotation))),
                        2)

# For the indoor scenes beta was between 120 to 750
final_loss = pose_loss + 150 * rotation_loss

loss_summary = tf.summary.scalar("loss value", final_loss)

# rmse_score_pose = tf.metrics.root_mean_squared_error(y_pose, pose_preditction)
# tf.summary.scalar("rmse for pose", rmse_score_pose)
# rmse_score_rotation = tf.metrics.root_mean_squared_error(y_rotation, rotation_prediction)
# tf.summary.scalar("rmse for rotation", rmse_score_rotation)

# merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter("./logs")

opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001,
                             use_locking=False, name='Adam').minimize(final_loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

train_batches_per_epoch = int(np.floor(1000 / BATCH_SIZE))
next_element = train_iterator.get_next()

with tf.Session() as sess:
    sess.run(init)
    writer.add_graph(tf.get_default_graph())
    posenet_model.load_initial_weights(sess)
    for epoch in range(EPOCH_NUM):
        sess.run(train_iterator.initializer)
        batch_num = 0
        while True:
            try:
                x_train, y_pose_train, y_rotation_train = sess.run(next_element)

                _, loss, summary_val = sess.run([opt, final_loss, loss_summary], feed_dict={"x:0": x_train,
                                                                                            "y_pose:0": y_pose_train,
                                                                                            "y_rotation:0": y_rotation_train})

                writer.add_summary(summary_val, epoch * train_batches_per_epoch + batch_num)

                print("epoch: {}, batch_num: {}, train_batches_per_epoch: {}, loss: {}".format(epoch, batch_num,
                                                                                               train_batches_per_epoch,
                                                                                               loss))

                batch_num += 1
            except tf.errors.OutOfRangeError:
                break

        if epoch % 10 == 0:
            saver.save(sess, CHECKPOINT_PATH + "Posenet_" + str(epoch) + ".ckpt")
