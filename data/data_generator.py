import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


def read_image(img_path, y_pose, y_rotation):
    image_string = tf.read_file(img_path)
    image_decoded = tf.image.decode_png(image_string, 3) / 255
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    return image_resized, y_pose, y_rotation


def get_train_and_test_iterator(batch_size):
    path_to_info = "../data/camera_relocalization_sample_dataset/{}"
    path_to_img = "../data/camera_relocalization_sample_dataset/images/{}"

    raw_df = pd.read_csv(path_to_info.format("info.csv"))

    image_file_col = ["ImageFile"]
    pose_col = ["POS_X", "POS_Y", "POS_Z"]
    rotation_col = ["Q_W", "Q_X", "Q_Y", "Q_Z"]

    X_train, X_test, y_train, y_test = train_test_split(raw_df[image_file_col], raw_df[pose_col + rotation_col],
                                                        test_size=0.15, train_size=0.85, random_state=42,
                                                        shuffle=True)

    pose_train = y_train[pose_col]
    rotation_train = y_train[rotation_col]

    pose_test = y_test[pose_col]
    rotation_test = y_test[rotation_col]

    X_train["ImageFile"] = X_train.ImageFile.apply(lambda x: path_to_img.format(x))
    X_test["ImageFile"] = X_test.ImageFile.apply(lambda x: path_to_img.format(x))

    with tf.variable_scope("data_generator"):
        x_train = tf.constant(X_train.values.flatten().tolist(), name="x")
        y_pose_train = tf.constant(pose_train.values.reshape((-1, 1, 1, 3)), name="y_pose")
        y_rotation_train = tf.constant(rotation_train.values.reshape((-1, 1, 1, 4)), name="y_rotation")

        x_test = tf.constant(X_test.values.flatten().tolist(), name="x")
        y_pose_test = tf.constant(pose_test.values.reshape((-1, 1, 1, 3)), name="y_pose")
        y_rotation_test = tf.constant(rotation_test.values.reshape((-1, 1, 1, 4)), name="y_rotation")

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_pose_train, y_rotation_train))
    train_dataset = train_dataset.map(read_image)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_pose_test, y_rotation_test))
    test_dataset = test_dataset.map(read_image)

    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    return train_iterator, test_iterator
