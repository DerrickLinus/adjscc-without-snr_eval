import tensorflow as tf
import numpy as np
import os
import time
import json # Import json for reading metadata

# Define AUTOTUNE for tf.data optimizations
AUTOTUNE = tf.data.experimental.AUTOTUNE


# get_num_samples 函数现在不再被 get_dataset_snr* 函数调用，
# 但可以保留用于其他目的或调试。
def get_num_samples(tfrecord_paths):
    """
    Calculates the total number of samples across a list of TFRecord files.
    (保留此函数，但不再是加载数据的主要方式)
    """
    num_samples = 0
    if not tfrecord_paths:
        print("Warning: No TFRecord file paths provided to get_num_samples.")
        return 0
    print(f"Calculating total samples from {len(tfrecord_paths)} TFRecord files (get_num_samples)...")
    start_time = time.time()
    for i, tfrecord_path in enumerate(tfrecord_paths):
        try:
            dataset = tf.data.TFRecordDataset(tfrecord_path)
            local_samples = dataset.reduce(np.int64(0), lambda x, _: x + 1)
            num_samples += local_samples.numpy()
        except tf.errors.NotFoundError:
            print(f"Warning: TFRecord file not found at {tfrecord_path}. Skipping.")
        except Exception as e:
            print(f"Error processing file {tfrecord_path}: {e}. Skipping.")
    end_time = time.time()
    print(f"Finished calculating samples (get_num_samples). Total: {num_samples}. Time taken: {end_time - start_time:.2f} seconds.")
    return num_samples


def _parse_function(example_proto):
    """
    Parses a single tf.train.Example proto into image tensor.
    """
    features = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "image/name": tf.io.FixedLenFeature([], default_value="", dtype=tf.string),
        "image/height": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "image/width": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    image_decoded = tf.image.decode_image(parsed_features["image/encoded"], channels=3)
    image_decoded = tf.reshape(image_decoded, (128, 128, 3))
    image_decoded = tf.cast(image_decoded, tf.float32)
    return image_decoded


# Functions for loading directly from image directories (remain unchanged)
# ... (decode_img, decode_eval_img, _parser_function_from_dir) ...
def decode_img(image_path):
    # Convert the compressed string to a 3D uint8 tensor
    image = tf.io.read_file(image_path)
    image_decoded = tf.io.decode_jpeg(image, channels=3)
    image_decoded = tf.image.resize_with_crop_or_pad(image_decoded, 128, 128)
    image_decoded = tf.cast(image_decoded, tf.float32)
    return image_decoded

def decode_eval_img(image_path):
    # Convert the compressed string to a 3D uint8 tensor
    image = tf.io.read_file(image_path)
    image_decoded = tf.io.decode_jpeg(image, channels=3)
    image_decoded = tf.image.resize_with_crop_or_pad(image_decoded, 512, 768)
    image_decoded = tf.cast(image_decoded, tf.float32)
    return image_decoded

def _parser_function_from_dir(example_proto):
    image_decoded = tf.image.convert_image_dtype(example_proto, tf.float32)
    return image_decoded


# --- Functions modified to load from multiple TFRecords and read metadata ---

def _load_num_samples_from_metadata(tfrecord_dir):
    """尝试从 metadata.json 文件加载样本总数"""
    metadata_path = os.path.join(tfrecord_dir, "metadata.json")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        num_samples = metadata.get("num_samples") # 使用 .get() 更安全
        if num_samples is None:
             raise KeyError("'num_samples' key not found in metadata file.")
        if not isinstance(num_samples, int) or num_samples < 0:
             raise ValueError(f"Invalid 'num_samples' value found in metadata: {num_samples}")
        print(f"从 {metadata_path} 加载样本总数: {num_samples}")
        return num_samples
    except FileNotFoundError:
        print(f"错误：元数据文件 {metadata_path} 未找到。")
        print("请先运行 tf_record_maker.py 来生成 TFRecord 文件和 metadata.json。")
        raise # 重新抛出异常，让调用者处理
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"错误：读取或解析元数据文件 {metadata_path} 时出错: {e}")
        raise # 重新抛出异常
    except Exception as e:
        print(f"加载元数据时发生未知错误: {e}")
        raise # 重新抛出异常


def get_dataset_snr(snr_db):
    """
    Loads the dataset from multiple TFRecord files and pairs each image
    with a fixed Signal-to-Noise Ratio (SNR) value.
    Reads total sample count from metadata.json.
    """
    # --- MODIFIED SECTION START ---
    tfrecord_dir = "/home/jay/workspace/datasets/imagenet/tfrecord/"
    tfrecord_pattern = os.path.join(tfrecord_dir, "imagenet-128-*.tfrecord")
    imgnet_filenames = tf.io.gfile.glob(tfrecord_pattern)

    if not imgnet_filenames:
        raise FileNotFoundError(
            f"错误：在目录下找不到匹配 '{tfrecord_pattern}' 的 TFRecord 文件。"
            f"请确认路径 '{tfrecord_dir}' 是否正确以及文件是否存在且命名符合 'imagenet-128-*.tfrecord' 格式。"
        )
    print(f"找到 {len(imgnet_filenames)} 个 TFRecord 文件用于 get_dataset_snr。")

    # 从 metadata.json 加载样本总数，不再调用 get_num_samples
    try:
        train_nums = _load_num_samples_from_metadata(tfrecord_dir)
    except Exception as e:
        # 如果加载失败，可以选择停止或回退到慢速计算（不推荐）
        print(f"无法从元数据加载样本数: {e}")
        # raise RuntimeError("无法继续，因为样本总数未知。") from e
        # 或者，如果绝对需要，可以取消下面一行的注释来回退：
        # print("警告：回退到慢速计算样本总数...")
        # train_nums = get_num_samples(imgnet_filenames)
        # if train_nums == 0: raise ValueError("错误：计算得到的样本数为 0。")
        raise RuntimeError("无法继续，因为样本总数未知。") from e # 推荐停止

    # --- MODIFIED SECTION END ---

    # 创建 TFRecordDataset
    imgnet_path_ds = tf.data.TFRecordDataset(imgnet_filenames, num_parallel_reads=AUTOTUNE)
    # 解析数据
    imgnet_ds = imgnet_path_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    # 创建 SNR 数据集 (使用从元数据加载的 train_nums)
    snrdb_train_ds = tf.data.Dataset.from_tensor_slices(
        snr_db * np.ones(shape=(train_nums,), dtype=np.float32)
    )
    # 打包数据集
    train_ds = tf.data.Dataset.zip(((imgnet_ds, snrdb_train_ds), imgnet_ds))

    return train_ds, train_nums


def get_dataset_snr_range(snr_db_low, snr_db_high):
    """
    Loads the dataset from multiple TFRecord files and pairs each image
    with a random Signal-to-Noise Ratio (SNR) value within a specified range.
    Reads total sample count from metadata.json.
    """
    # --- MODIFIED SECTION START ---
    tfrecord_dir = "/home/jay/workspace/datasets/imagenet/tfrecord/"
    tfrecord_pattern = os.path.join(tfrecord_dir, "imagenet-128-*.tfrecord")
    imgnet_filenames = tf.io.gfile.glob(tfrecord_pattern)

    if not imgnet_filenames:
        raise FileNotFoundError(
            f"错误：在目录下找不到匹配 '{tfrecord_pattern}' 的 TFRecord 文件。"
            f"请确认路径 '{tfrecord_dir}' 是否正确以及文件是否存在且命名符合 'imagenet-128-*.tfrecord' 格式。"
        )
    print(f"找到 {len(imgnet_filenames)} 个 TFRecord 文件用于 get_dataset_snr_range。")

    # 从 metadata.json 加载样本总数
    try:
        train_nums = _load_num_samples_from_metadata(tfrecord_dir)
    except Exception as e:
        print(f"无法从元数据加载样本数: {e}")
        raise RuntimeError("无法继续，因为样本总数未知。") from e
        # 或者回退:
        # print("警告：回退到慢速计算样本总数...")
        # train_nums = get_num_samples(imgnet_filenames)
        # if train_nums == 0: raise ValueError("错误：计算得到的样本数为 0。")

    # --- MODIFIED SECTION END ---

    # 创建 TFRecordDataset
    imgnet_path_ds = tf.data.TFRecordDataset(imgnet_filenames, num_parallel_reads=AUTOTUNE)
    # 解析数据
    imgnet_ds = imgnet_path_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)
    # 创建随机 SNR 数据集 (使用从元数据加载的 train_nums)
    snrdb_train = (
        np.random.rand(train_nums,) * (snr_db_high - snr_db_low) + snr_db_low
    ).astype(np.float32)
    snrdb_train_ds = tf.data.Dataset.from_tensor_slices(snrdb_train)
    # 打包数据集
    train_ds = tf.data.Dataset.zip(((imgnet_ds, snrdb_train_ds), imgnet_ds))

    return train_ds, train_nums


# --- 其他函数 (get_dataset_snr_from_dir, get_eval_dataset_snr_range_from_dir 等) 保持不变 ---
# ...

# --- CIFAR-10 related functions (using tf.keras.datasets) remain unchanged ---
# ... (get_dataset_snr_range_and_h, get_test_dataset_burst) ...
def get_dataset_snr_range_and_h(snr_db_low, snr_db_high):
    # Loads CIFAR-10, no changes needed.
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_nums, test_nums = len(x_train), len(x_test)
    # Generate channel coefficients (h) and SNRs
    h_real_train = np.sqrt(1 / 2) * np.random.randn(train_nums,).astype(np.float32)
    h_imag_train = np.sqrt(1 / 2) * np.random.randn(train_nums,).astype(np.float32)
    snrdb_train = (np.random.rand(train_nums,) * (snr_db_high - snr_db_low) + snr_db_low).astype(np.float32)

    h_real_test = np.sqrt(1 / 2) * np.random.randn(test_nums,).astype(np.float32)
    h_imag_test = np.sqrt(1 / 2) * np.random.randn(test_nums,).astype(np.float32)
    snrdb_test = (np.random.rand(test_nums,) * (snr_db_high - snr_db_low) + snr_db_low).astype(np.float32)

    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices(
        ((x_train.astype(np.float32), snrdb_train, h_real_train, h_imag_train), x_train.astype(np.float32)) # Cast images too
    )
    test_ds = tf.data.Dataset.from_tensor_slices(
        ((x_test.astype(np.float32), snrdb_test, h_real_test, h_imag_test), x_test.astype(np.float32))
    )
    return (train_ds, train_nums), (test_ds, test_nums)


def get_test_dataset_burst(snr_db, b_prob, b_stddev):
    # Loads CIFAR-10 test set, no changes needed.
    cifar10 = tf.keras.datasets.cifar10
    (_, _), (x_test, _) = cifar10.load_data()
    test_nums = len(x_test)
    snrdb_test = snr_db * np.ones(shape=(test_nums,), dtype=np.float32)
    b_prob_test = b_prob * np.ones(shape=(test_nums,), dtype=np.float32)
    b_stddev_test = b_stddev * np.ones(shape=(test_nums,), dtype=np.float32)
    test_ds = tf.data.Dataset.from_tensor_slices(
        ((x_test.astype(np.float32), snrdb_test, b_prob_test, b_stddev_test), x_test.astype(np.float32))
    )
    return test_ds, test_nums