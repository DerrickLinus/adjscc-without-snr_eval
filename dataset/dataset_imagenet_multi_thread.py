import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_num_samples(tfrecord_paths):
    """计算给定TFRecord文件列表中的总样本数"""
    num_samples = 0
    # 检查路径列表是否为空
    if not tfrecord_paths:
        print("Warning: No TFRecord files provided to get_num_samples.")
        return 0
    for tfrecord_path in tfrecord_paths:
        try:
            # 使用 tf.data.TFRecordDataset 更高效地计数
            # 注意：如果文件非常大且数量多，这仍然可能较慢
            # 考虑在生成 TFRecord 时记录总数，或接受一个预估值
            dataset = tf.data.TFRecordDataset(tfrecord_path)
            # 使用 reduce 来计数，避免加载整个记录
            local_samples = dataset.reduce(np.int64(0), lambda x, _: x + 1)
            num_samples += local_samples.numpy()
        except tf.errors.NotFoundError:
            print(f"Warning: TFRecord file not found: {tfrecord_path}")
        except Exception as e:
            print(f"Error processing file {tfrecord_path}: {e}")
            # 可以选择在此处停止或继续处理其他文件
    return num_samples


def _parse_function(example_proto):
    """解析单个 TFRecord 样本"""
    features = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string, default_value=""),
        # 'image/format': tf.FixedLenFeature([], default_value='jpeg', dtype=tf.string), # 通常不需要格式
        "image/name": tf.io.FixedLenFeature([], default_value="", dtype=tf.string),
        "image/height": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "image/width": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    # 解码图像
    image_decoded = tf.image.decode_image(parsed_features["image/encoded"], channels=3) # 明确指定通道数
    # 确保形状和类型符合预期 (tf_record_maker 中是 128x128)
    image_decoded = tf.reshape(image_decoded, (128, 128, 3))
    image_decoded = tf.cast(image_decoded, tf.float32)
    return image_decoded


# decode_img, decode_eval_img, _parser_function_from_dir 函数保持不变，
# 因为它们用于从目录直接读取，而非 TFRecord 文件


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
    """从多个TFRecord文件加载数据，并为每个样本分配固定的SNR值"""
    # --- 修改开始 ---
    # 定义 TFRecord 文件所在的目录和匹配模式
    tfrecord_dir = "/home/jay/workspace/datasets/imagenet/tfrecord/"
    tfrecord_pattern = os.path.join(tfrecord_dir, "imagenet-128-*.tfrecord")

    # 使用 glob 查找所有匹配的文件
    imgnet_filenames = tf.io.gfile.glob(tfrecord_pattern)

    # 检查是否找到了文件
    if not imgnet_filenames:
        raise FileNotFoundError(f"错误：在目录下找不到匹配 '{tfrecord_pattern}' 的 TFRecord 文件。请检查路径和文件名模式。")
    print(f"找到 {len(imgnet_filenames)} 个 TFRecord 文件用于 get_dataset_snr。")
    # --- 修改结束 ---

    # 计算总样本数 (如果需要精确值)
    # 注意：这可能会比较慢，特别是对于大量大文件
    train_nums = get_num_samples(imgnet_filenames)
    if train_nums == 0:
        raise ValueError("错误：从 TFRecord 文件中读取到的样本数为 0。")
    print(f"计算得到总样本数: {train_nums}")

    # 创建 TFRecordDataset，传入所有文件名，并启用并行读取
    imgnet_path_ds = tf.data.TFRecordDataset(imgnet_filenames, num_parallel_reads=AUTOTUNE)

    # 解析数据
    imgnet_ds = imgnet_path_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)

    # 创建 SNR 数据集
    snrdb_train_ds = tf.data.Dataset.from_tensor_slices(
        snr_db * np.ones(shape=(train_nums,), dtype=np.float32)
    )

    # 将图像数据集和 SNR 数据集打包
    # ((图像, SNR), 图像) 格式，通常用于 (输入特征, 标签)
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
    """从多个TFRecord文件加载数据，并为每个样本分配指定范围内的随机SNR值"""
    # --- 修改开始 ---
    # 定义 TFRecord 文件所在的目录和匹配模式
    tfrecord_dir = "/home/jay/workspace/datasets/imagenet/tfrecord/"
    tfrecord_pattern = os.path.join(tfrecord_dir, "imagenet-128-*.tfrecord")

    # 使用 glob 查找所有匹配的文件
    imgnet_filenames = tf.io.gfile.glob(tfrecord_pattern)

    # 检查是否找到了文件
    if not imgnet_filenames:
         raise FileNotFoundError(f"错误：在目录下找不到匹配 '{tfrecord_pattern}' 的 TFRecord 文件。请检查路径和文件名模式。")
    print(f"找到 {len(imgnet_filenames)} 个 TFRecord 文件用于 get_dataset_snr_range。")
   # --- 修改结束 ---

    # 计算总样本数 (如果需要精确值)
    train_nums = get_num_samples(imgnet_filenames)
    if train_nums == 0:
        raise ValueError("错误：从 TFRecord 文件中读取到的样本数为 0。")
    print(f"计算得到总样本数: {train_nums}")


    # 创建 TFRecordDataset，传入所有文件名，并启用并行读取
    imgnet_path_ds = tf.data.TFRecordDataset(imgnet_filenames, num_parallel_reads=AUTOTUNE)

    # 解析数据
    imgnet_ds = imgnet_path_ds.map(_parse_function, num_parallel_calls=AUTOTUNE)

    # 创建随机 SNR 数据集
    snrdb_train = (
        np.random.rand(train_nums,) * (snr_db_high - snr_db_low) + snr_db_low
    ).astype(np.float32) # 确保类型为 float32
    snrdb_train_ds = tf.data.Dataset.from_tensor_slices(snrdb_train)

    # 将图像数据集和 SNR 数据集打包
    train_ds = tf.data.Dataset.zip(((imgnet_ds, snrdb_train_ds), imgnet_ds))
    return train_ds, train_nums

# --- 其他函数 (get_dataset_snr_from_dir, get_eval_dataset_snr_range_from_dir 等) 保持不变 ---
# --- 因为它们是从原始图像目录加载数据，而不是 TFRecord ---

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