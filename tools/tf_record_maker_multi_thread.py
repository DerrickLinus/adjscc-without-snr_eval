#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/jay/git/ADJSCC/tf_record_maker.py
# Project: ADJSCC
# Created Date: Friday, April 14th 2023, 2:17:00 pm
# Author: Shisui
# Copyright (c) 2023 Uchiha
# 多进程版本，生成多个.tfrecord文件，利用多核 CPU，可以显著加快大规模数据集（如 ImageNet）的处理速度
# ----------	---	----------------------------------------------------------
###

# %%

import tensorflow as tf
import os
import multiprocessing as mp
from tqdm import tqdm

coord = tf.train.Coordinator()
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# %%


def get_image_path_list(dir):
    extens = [".jpg", ".png", ".jpeg", ".JPEG"]
    path_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] in extens:
                path_list.append(os.path.join(root, file))
    return path_list


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(encoded, name, height, width):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        "image/encoded": _bytes_feature(encoded),
        "image/name": _bytes_feature(name),
        "image/height": _int64_feature(height),
        "image/width": _int64_feature(width),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def random_crop(image_path):
    crop_list = []
    shape = [0, 0]
    image = tf.io.read_file(image_path)
    try:
        image = tf.image.decode_jpeg(image, channels=3)
    except tf.errors.InvalidArgumentError:
        print("Error: ", image)
        return crop_list, shape
    shape = image.shape
    if shape[0] < 128 and shape[1] < 128:
        return crop_list, shape
    crop_time = round((shape[0] // 128) * (shape[1] // 128) * 1.5)
    for i in range(crop_time):
        crop = tf.image.random_crop(image, size=(128, 128, 3), seed=1)
        crop = tf.image.encode_jpeg(crop)
        crop_list.append(crop)
    return crop_list, shape


def make_example_list(image_path):
    serialized_example_list = []
    name = bytes(os.path.basename(image_path), "utf-8")
    crop_list, shape = random_crop(image_path)
    for crop in crop_list:
        example = serialize_example(
            encoded=crop,
            name=name,
            height=shape[0],
            width=shape[1],
        )
        serialized_example_list.append(example)
    return serialized_example_list


def _parse_function(example_proto):
    features = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image/name": tf.io.FixedLenFeature([], default_value="", dtype=tf.string),
        "image/height": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "image/width": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    image_decoded = tf.image.decode_image(parsed_features["image/encoded"])
    return image_decoded


def test():
    exa = make_example_list("pics/ADJSCC.png")
    result = _parse_function(exa[0])
    assert result.shape == (128, 128, 3)


def tf_writer(process_index, image_path_list, lock, dst):
    with lock:
        pbar = tqdm(
            desc=f"Position {process_index}",
            total=len(image_path_list),
            position=process_index,
            leave=False,
        )
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    record_path = os.path.join(dst, f"imagenet-128-{process_index}.tfrecord")
    print("writing to", record_path, "...")
    with tf.io.TFRecordWriter(record_path) as writer:
        for image_path in image_path_list:
            example_list = make_example_list(image_path)
            for example in example_list:
                writer.write(example)
            with lock:
                pbar.update(1)
    with lock:
        pbar.close()


def make_tf_record(src, dst, process_num=16):
    image_path_list = get_image_path_list(src)
    processes = []
    lock = mp.Manager().Lock()
    for process_index in range(process_num):
        start_index = process_index * len(image_path_list) // process_num
        end_index = (process_index + 1) * len(image_path_list) // process_num
        args = (process_index, image_path_list[start_index:end_index], lock, dst)
        if process_index == process_num - 1:
            args = (process_index, image_path_list[start_index:], lock, dst)
        p = mp.Process(target=tf_writer, args=args)
        p.start()
        processes.append(p)
    print("Done!")
    coord.join(processes)


# %%

if __name__ == "__main__":
    src = "/home/jay/workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/train"
    dst = "/home/jay/workspace/datasets/imagenet/tfrecord"
    make_tf_record(src=src, dst=dst, process_num=16) # /home/jay/workspace/datasets/imagenet/tfrecord下已经有30个.tfrecord文件？
