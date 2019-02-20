import sys, os
import tensorflow as tf
from model.posenet import PoseNet

CHECKPOINT_PATH = "../checkpoints/"

if os.path.exists(CHECKPOINT_PATH):
    img_path = sys.argv[1]
    print("You run inference for file : {}".format(img_path))

    image_string = tf.read_file(img_path)
    image_decoded = tf.image.decode_png(image_string, 3) / 255
    image_resized = tf.image.resize_images(image_decoded, [224, 224])

    x = tf.placeholder(tf.float32, [1, 224, 224, 3], name="x")

    posenet_model = PoseNet(x, "./pretrained_models/googlenet.npy")

    pose_prediction = posenet_model.posenet_output_pose
    rotation_prediction = posenet_model.posenet_output_rotation

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    files = os.listdir("./")
    files.sort(key=os.path.getmtime, reverse=True)

    last_checkpoint_name = files[0]

    with tf.Session() as sess:
        sess.run(init())

        saver.restore(sess, CHECKPOINT_PATH + last_checkpoint_name)

        predicted_pose, predicted_rotation = sess.run([pose_prediction, rotation_prediction],
                                                      feed_dict={x: [image_resized]})

        print("predicted pose: {}".format(predicted_pose))
        print("predicted rotation: {}".format(predicted_rotation))
else:
    print("Please, train model before running the model in inference mode")
