#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/jay/git/ADJSCC/tf_record_maker.py
# Project: ADJSCC
# Created Date: Friday, April 14th 2023, 2:17:00 pm
# Author: Shisui
# Copyright (c) 2023 Uchiha
# 多进程版本，增加“计算并保存总样本数”的功能，避免在运行dataset_imagenet.py时计算总样本数
# ---------- --- ----------------------------------------------------------
###

# %%

import tensorflow as tf
import os
import multiprocessing as mp
from tqdm import tqdm
import json # Import json for saving metadata

# coord = tf.train.Coordinator() # Coordinator is deprecated, use join directly
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# %%


def get_image_path_list(dir):
    """获取目录下所有图片的路径列表"""
    extens = [".jpg", ".png", ".jpeg", ".JPEG"]
    path_list = []
    print(f"正在扫描目录 '{dir}' 中的图片...")
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] in extens:
                path_list.append(os.path.join(root, file))
    print(f"找到 {len(path_list)} 张图片。")
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
    """从单个图像随机裁剪出多个 128x128 的区域"""
    crop_list = []
    shape = [0, 0, 0] # Initialize with 3 dimensions
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False) # Decode any format, ensure 3 channels
        image.set_shape([None, None, 3]) # Help TF infer the shape
    except tf.errors.InvalidArgumentError as e:
        # Skip corrupted images or formats tf.image.decode_image can't handle
        print(f"错误：解码图片时出错 {image_path}: {e}。跳过此图片。")
        return crop_list, shape
    except Exception as e: # Catch other potential errors
        print(f"错误：读取或处理图片时发生未知错误 {image_path}: {e}。跳过此图片。")
        return crop_list, shape


    shape = tf.shape(image).numpy() # Get actual shape after decoding

    # Check if image is large enough for at least one crop
    if shape[0] < 128 or shape[1] < 128:
        # print(f"信息：图片 {os.path.basename(image_path)} ({shape[0]}x{shape[1]}) 太小，无法裁剪。跳过。")
        return crop_list, shape

    # Calculate number of crops based on area, with some overlap factor (1.5)
    # Ensure integer division and minimum 1 crop if possible
    crop_time = max(1, round((shape[0] / 128.0) * (shape[1] / 128.0) * 1.5))

    for _ in range(crop_time):
        # Perform random crop
        crop = tf.image.random_crop(image, size=(128, 128, 3))
        # Encode the crop back to JPEG format for storage
        try:
            encoded_crop = tf.image.encode_jpeg(crop, quality=95) # Specify quality
            crop_list.append(encoded_crop)
        except Exception as e:
            print(f"错误：编码裁剪后的图片时出错 {os.path.basename(image_path)}: {e}")
            continue # Skip this crop if encoding fails

    return crop_list, shape


def make_example_list(image_path):
    """为单个图片生成多个序列化的 tf.train.Example"""
    serialized_example_list = []
    name = bytes(os.path.basename(image_path), "utf-8")
    crop_list, shape = random_crop(image_path)
    for crop in crop_list:
        example = serialize_example(
            encoded=crop,
            name=name,
            height=shape[0], # Store original height
            width=shape[1],  # Store original width
        )
        serialized_example_list.append(example)
    return serialized_example_list


def _parse_function(example_proto):
    """解析函数（用于测试）"""
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
    """测试函数"""
    # Create a dummy image file for testing if it doesn't exist
    test_img_path = "dummy_test_image.png"
    if not os.path.exists(test_img_path):
        try:
            dummy_img = tf.random.uniform([200, 300, 3], maxval=256, dtype=tf.int32)
            dummy_img_encoded = tf.image.encode_png(tf.cast(dummy_img, tf.uint8))
            tf.io.write_file(test_img_path, dummy_img_encoded)
            print(f"创建了测试图片: {test_img_path}")
        except Exception as e:
            print(f"无法创建测试图片: {e}")
            return

    print(f"正在测试图片: {test_img_path}")
    exa_list = make_example_list(test_img_path)
    if not exa_list:
        print("测试失败：make_example_list 未返回任何样本。")
        return
    print(f"从测试图片生成了 {len(exa_list)} 个样本。")
    result = _parse_function(exa_list[0])
    print(f"解析后的样本形状: {result.shape}")
    assert result.shape == (128, 128, 3), f"测试失败：预期形状 (128, 128, 3)，但得到 {result.shape}"
    print("测试通过！")
    # Clean up the dummy image
    # try:
    #     os.remove(test_img_path)
    # except OSError as e:
    #     print(f"无法删除测试图片: {e}")


def tf_writer(process_index, image_path_list, lock, dst, results_queue):
    """
    子进程函数，处理一部分图片并写入一个 TFRecord 文件。
    同时计数写入的样本数并通过队列返回。
    """
    local_sample_count = 0 # 初始化此进程的样本计数器
    record_path = os.path.join(dst, f"imagenet-128-{process_index}.tfrecord")
    print(f"[进程 {process_index}] 开始写入到 {record_path}...")

    # 使用上下文管理器确保 pbar 被正确关闭
    with lock:
        # leave=True 可以在所有进程结束后保留进度条
        pbar = tqdm(
            desc=f"进程 {process_index}",
            total=len(image_path_list),
            position=process_index, # 让每个进程的进度条在不同行显示
            leave=True,
        )

    try:
        with tf.io.TFRecordWriter(record_path) as writer:
            for image_path in image_path_list:
                example_list = make_example_list(image_path)
                for example in example_list:
                    writer.write(example)
                    local_sample_count += 1 # 每写入一个样本，计数器加1
                # 更新进度条
                with lock:
                    pbar.update(1)
    except Exception as e:
        print(f"[进程 {process_index}] 写入文件时出错: {e}")
    finally:
        # 确保进度条关闭
        with lock:
            pbar.close()
        # 将此进程处理的样本数放入结果队列
        results_queue.put(local_sample_count)
        print(f"[进程 {process_index}] 完成。写入了 {local_sample_count} 个样本。")


def make_tf_record(src, dst, process_num=16):
    """
    使用多进程将源目录中的图片转换为 TFRecord 文件。
    计算总样本数并保存到 metadata.json。
    """
    # 确保目标目录存在
    os.makedirs(dst, exist_ok=True)

    image_path_list = get_image_path_list(src)
    if not image_path_list:
        print("错误：在源目录中没有找到图片，无法继续。")
        return

    processes = []
    lock = mp.Manager().Lock() # 用于同步 tqdm 进度条的访问
    results_queue = mp.Queue() # 用于从子进程收集样本数

    print(f"\n将使用 {process_num} 个进程生成 TFRecord 文件...")

    # 分配任务给每个进程
    chunk_size = len(image_path_list) // process_num
    for process_index in range(process_num):
        start_index = process_index * chunk_size
        # 对于最后一个进程，包含所有剩余的图片
        end_index = None if process_index == process_num - 1 else (process_index + 1) * chunk_size
        process_image_list = image_path_list[start_index:end_index]

        if not process_image_list: # 如果某个进程没有分配到任务，则跳过
             print(f"警告：进程 {process_index} 没有分配到图片，跳过。")
             continue

        args = (process_index, process_image_list, lock, dst, results_queue)
        p = mp.Process(target=tf_writer, args=args)
        p.start()
        processes.append(p)

    # 等待所有子进程完成
    for p in processes:
        p.join()

    print("\n所有进程已完成 TFRecord 文件写入。")

    # 从队列中收集所有子进程的样本计数
    total_sample_count = 0
    while not results_queue.empty():
        try:
            count = results_queue.get_nowait() # 使用非阻塞获取
            total_sample_count += count
        except mp.queues.Empty: # 理论上 join 后不应发生，但以防万一
            break
        except Exception as e:
            print(f"从结果队列获取计数时出错: {e}")


    print(f"计算得到的总样本数: {total_sample_count}")

    # 将总样本数保存到 metadata.json 文件
    metadata_path = os.path.join(dst, "metadata.json")
    metadata = {"num_samples": total_sample_count}
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"总样本数已保存到: {metadata_path}")
    except IOError as e:
        print(f"错误：无法写入元数据文件 {metadata_path}: {e}")
    except Exception as e:
        print(f"写入元数据时发生未知错误: {e}")

    print("TFRecord 文件生成完毕！")


# %%

if __name__ == "__main__":
    # 运行测试函数
    # test()

    # 设置源图片目录和目标 TFRecord 文件目录
    # 注意：请确保这些路径在您的系统上是正确的
    src_dir = "/home/jay/workspace/datasets/imagenet/ILSVRC/Data/CLS-LOC/train" # 源图片目录
    dst_dir = "/home/jay/workspace/datasets/imagenet/tfrecord/" # TFRecord 输出目录

    # 检查源目录是否存在
    if not os.path.isdir(src_dir):
         print(f"错误：源目录 '{src_dir}' 不存在或不是一个目录。请检查路径。")
    else:
        # 设置要使用的进程数 (可以根据 CPU 核心数调整)
        num_processes = 16 # 例如，使用 16 个进程
        make_tf_record(src=src_dir, dst=dst_dir, process_num=num_processes)

