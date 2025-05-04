import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_num_samples(tfrecord_paths):
    num_samples = 0
    for tfrecord_path in tfrecord_paths:
        for record in tf.compat.v1.python_io.tf_record_iterator(tfrecord_path):
            num_samples += 1
    return num_samples


def _parse_function(example_proto):
    features = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string, default_value=""),
        # 'image/format': tf.FixedLenFeature([], default_value='jpeg', dtype=tf.string),
        "image/name": tf.io.FixedLenFeature([], default_value="", dtype=tf.string),
        "image/height": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "image/width": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    image_decoded = tf.image.decode_image(parsed_features["image/encoded"])
    image_decoded = tf.reshape(image_decoded, (128, 128, 3))
    image_decoded = tf.cast(image_decoded, tf.float32)
    return image_decoded


def decode_img(image_path):
    # Convert the compressed string to a 3D uint8 tensor
    image = tf.io.read_file(image_path)
    image_decoded = tf.io.decode_jpeg(image, channels=3)
    image_decoded = tf.image.resize_with_crop_or_pad(image_decoded, 128, 128)
    # image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    # image_decoded = tf.reshape(image_decoded, (128, 128, 3))
    image_decoded = tf.cast(image_decoded, tf.float32)
    return image_decoded


def decode_eval_img(image_path):
    # Convert the compressed string to a 3D uint8 tensor
    image = tf.io.read_file(image_path)
    image_decoded = tf.io.decode_jpeg(image, channels=3)
    # image_decoded = tf.image.resize_with_crop_or_pad(image_decoded, 512, 768)
    # image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    # image_decoded = tf.reshape(image_decoded, (512, 768, 3))
    image_decoded = tf.image.resize_with_crop_or_pad(image_decoded, 512, 768)
    image_decoded = tf.cast(image_decoded, tf.float32)
    return image_decoded


def _parser_function_from_dir(example_proto):
    image_decoded = tf.image.convert_image_dtype(example_proto, tf.float32)
    return image_decoded


def get_dataset_snr(snr_db):
    imgnet_filenames = [
        "dataset/imagenet/ILSVRC_train-crop_128.tfrecord", 
        # 运行 tf_record_maker_single_thread.py（单线程版本）得到的一个.tfrecord文件
        # 测试可运行：
        # "/home/jay/workspace/datasets/imagenet/tfrecord/imagenet-128-0.tfrecord"
    ]
    train_nums = get_num_samples(imgnet_filenames)
    imgnet_path_ds = tf.data.TFRecordDataset(imgnet_filenames)
    imgnet_ds = imgnet_path_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    snrdb_train_ds = tf.data.Dataset.from_tensor_slices(
        snr_db * np.ones(shape=(train_nums,), dtype=np.float32)
    )
    train_ds = tf.data.Dataset.zip(((imgnet_ds, snrdb_train_ds), imgnet_ds))
    return train_ds, train_nums


def get_dataset_snr_from_dir(snr_db):
    imgnet_dir = "/home/jay/workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/train"
    imgnet_path_ds = tf.data.Dataset.list_files(imgnet_dir + "/*/*.JPEG", shuffle=False)
    train_nums = imgnet_path_ds.cardinality().numpy() - 1
    error_img = b"/home/jay/workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/train/n02391049/n02391049_5165.JPEG"
    imgnet_path_ds = imgnet_path_ds.filter(lambda x: not tf.math.equal(x, error_img))

    imgnet_ds = imgnet_path_ds.map(decode_img, num_parallel_calls=AUTOTUNE)

    snrdb_train_ds = tf.data.Dataset.from_tensor_slices(
        snr_db * np.ones(shape=(train_nums,), dtype=np.float32)
    )
    train_ds = tf.data.Dataset.zip(
        ((imgnet_ds, snrdb_train_ds), imgnet_ds),
    )
    return train_ds, train_nums


def get_dataset_snr_range(snr_db_low, snr_db_high):
    imgnet_filenames = [
        "dataset/imagenet/ILSVRC_train-crop_128.tfrecord",
    ]
    train_nums = get_num_samples(imgnet_filenames)
    imgnet_path_ds = tf.data.TFRecordDataset(imgnet_filenames)
    imgnet_ds = imgnet_path_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    snrdb_train = (
        np.random.rand(
            train_nums,
        )
        * (snr_db_high - snr_db_low)
        + snr_db_low
    )
    snrdb_train_ds = tf.data.Dataset.from_tensor_slices(snrdb_train)
    train_ds = tf.data.Dataset.zip(((imgnet_ds, snrdb_train_ds), imgnet_ds))
    return train_ds, train_nums


def get_dataset_snr_range_from_dir(snr_db_low, snr_db_high):
    imgnet_dir = "/home/jay/workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/train"
    imgnet_path_ds = tf.data.Dataset.list_files(imgnet_dir + "/*/*.JPEG", shuffle=False)
    train_nums = imgnet_path_ds.cardinality().numpy() - 1

    error_img = b"/home/jay/workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/train/n02391049/n02391049_5165.JPEG"
    imgnet_path_ds = imgnet_path_ds.filter(lambda x: not tf.math.equal(x, error_img))
    imgnet_ds = imgnet_path_ds.map(decode_img, num_parallel_calls=AUTOTUNE)

    snrdb_train = (
        np.random.rand(
            train_nums,
        )
        * (snr_db_high - snr_db_low)
        + snr_db_low
    )
    snrdb_train_ds = tf.data.Dataset.from_tensor_slices(snrdb_train)
    train_ds = tf.data.Dataset.zip(((imgnet_ds, snrdb_train_ds), imgnet_ds))
    return train_ds, train_nums


def get_eval_dataset_snr_range_from_dir(snr):
    imgnet_dir = "/home/jay/workspace/datasets/kodak"
    imgnet_path_ds = tf.data.Dataset.list_files(imgnet_dir + "/*.png", shuffle=False)
    train_nums = imgnet_path_ds.cardinality().numpy()

    # error_img = b"/home/jay/Documents/datasets/imagenet/ILSVRC/Data/CLS-LOC/train/n02391049/n02391049_5165.JPEG"
    # error_img = b"/home/jay/workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/train/n02391049/n02391049_5165.JPEG"
    # imgnet_path_ds = imgnet_path_ds.filter(lambda x: not tf.math.equal(x, error_img))
    imgnet_ds = imgnet_path_ds.map(decode_eval_img, num_parallel_calls=AUTOTUNE)

    snrdb_train = (
        np.ones(
            train_nums,
        )
        * snr
    )
    snrdb_train_ds = tf.data.Dataset.from_tensor_slices(snrdb_train)
    train_ds = tf.data.Dataset.zip(((imgnet_ds, snrdb_train_ds), imgnet_ds))
    return train_ds, train_nums


def get_dataset_snr_range_and_h(snr_db_low, snr_db_high):
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_nums, test_nums = len(x_train), len(x_test)
    h_real_train = np.sqrt(1 / 2) * np.random.randn(
        train_nums,
    )
    h_imag_train = np.sqrt(1 / 2) * np.random.randn(
        train_nums,
    )
    snrdb_train = (
        np.random.rand(
            train_nums,
        )
        * (snr_db_high - snr_db_low)
        + snr_db_low
    )
    snrdb_test = (
        np.random.rand(
            test_nums,
        )
        * (snr_db_high - snr_db_low)
        + snr_db_low
    )
    h_real_test = np.sqrt(1 / 2) * np.random.randn(
        test_nums,
    )
    h_imag_test = np.sqrt(1 / 2) * np.random.randn(
        test_nums,
    )
    train_ds = tf.data.Dataset.from_tensor_slices(
        ((x_train, snrdb_train, h_real_train, h_imag_train), x_train)
    )
    test_ds = tf.data.Dataset.from_tensor_slices(
        ((x_test, snrdb_test, h_real_test, h_imag_test), x_test)
    )
    return (train_ds, train_nums), (test_ds, test_nums)


def get_test_dataset_burst(snr_db, b_prob, b_stddev):
    cifar10 = tf.keras.datasets.cifar10
    (_, _), (x_test, _) = cifar10.load_data()
    test_nums = len(x_test)
    snrdb_test = snr_db * np.ones(shape=(test_nums,))
    b_prob_test = b_prob * np.ones(shape=(test_nums,))
    b_stddev_test = b_stddev * np.ones(shape=(test_nums,))
    test_ds = tf.data.Dataset.from_tensor_slices(
        ((x_test, snrdb_test, b_prob_test, b_stddev_test), x_test)
    )
    return test_ds, test_nums