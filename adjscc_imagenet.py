from util_channel import Channel
from util_module import Attention_Encoder, Attention_Decoder
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import argparse
from dataset import dataset_imagenet_multi_thread # 多线程版本，对应 tf_record_maker_multi_thread.py
# from dataset import dataset_imagenet_multi_thread_read_metadata # 多线程直接读取元数据版本，对应 tf_record_maker_multi_thread_sample_count.py
# from dataset import dataset_imagenet_single_thread # 单线程版本，对应 tf_record_maker_single_thread.py
import os
import json
import datetime
from yaml import dump
import pandas as pd


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_kodak():
    images = np.empty(shape=[0,512,768,3])
    for i in range(1, 25):
        if i<10:
            image_path = '/home/jay/workspace/datasets/kodak/kodim0' + str(i) + '.png'
        else:
            image_path = '/home/jay/workspace/datasets/kodak/kodim' + str(i) + '.png'
        img_file = tf.io.read_file(image_path)
        image = tf.image.decode_png(img_file, channels=3)
        if image.shape[0] == 768:
            image = tf.transpose(image, [1, 0, 2])
        image = image[np.newaxis,:]
        images = np.append(images, image, axis=0)
    return images


def train(args, model):
    print("########## Making work dir ##########")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(args.loss_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, 'args.yaml'), 'w') as f:
        dump(args, f)
    if args.load_model_path is not None:
        model.load_weights(args.load_model_path)
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_low_train) + 'to' + str(
        args.snr_up_train) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    cbk = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, save_weights_only=True, save_freq=100)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.board_dir, histogram_freq=1, update_freq=20, profile_batch='1000,1020')
    train_ds, train_nums = dataset_imagenet_multi_thread.get_dataset_snr_range(args.snr_low_train, args.snr_up_train)
    # train_ds, train_nums = dataset_imagenet.get_dataset_snr_range(args.snr_low_train, args.snr_up_train)
    # train_ds = train_ds.shuffle(buffer_size=train_nums, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    # train_step = (
    #     train_nums // args.batch_size if train_nums % args.batch_size == 0 else train_nums // args.batch_size + 1)
    print("########## Start Training ##########")
    print("image nums: ", train_nums)
    h = model.fit(train_ds, epochs=args.epochs, batch_size=args.batch_size, callbacks=[cbk, tensorboard_callback], workers=8, use_multiprocessing=True)

def eval(args, model):
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_low_eval) + 'to' + str(
        args.snr_up_eval) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    model.load_weights(model_path)
    snr_list = []
    mse_list = []
    psnr_list = []
    # kodak = get_kodak()
    for snrdb in range(args.snr_low_eval, args.snr_up_eval + 1):
        imse = []
        # test 10 times each snr
        eval_dataset, eval_nums = dataset_imagenet.get_eval_dataset_snr_range_from_dir(snrdb)
        eval_dataset = eval_dataset.prefetch(buffer_size=AUTOTUNE)
        eval_dataset = eval_dataset.batch(1)
        eval_dataset = eval_dataset.cache()
        for i in range(0, 5):
            mse = model.evaluate(eval_dataset)
            imse.append(mse)
        mse = np.mean(imse)
        psnr = 10 * np.log10(255**2 / mse)
        snr_list.append(snrdb)
        mse_list.append(mse)
        psnr_list.append(psnr)
        with open(os.path.join(args.eval_dir, filename + '_fs' + str('none' if args.feedback_snr is None else args.feedback_snr) + '.json'), mode='w') as f:
            json.dump({'snr': snr_list, 'mse': mse_list, 'psnr': psnr_list}, f)
        result = pd.DataFrame({'snr': snr_list, 'mse': mse_list, 'psnr': psnr_list})
        result.to_csv(os.path.join(args.eval_dir, filename + 'fs' + str('none' if args.feedback_snr is None else args.feedback_snr) + '.csv'), index=False)

def predict(args, model):
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_low_train) + 'to' + str(
        args.snr_up_train) + '_bs' + str(args.batch_size) + '_lr' + str(args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    model.load_weights(model_path)
    img_name = 'kodim24'
    image_path = 'dataset/kodak/'+img_name+'.png'
    img_file = tf.io.read_file(image_path)
    image = tf.image.decode_png(img_file, channels=3)
    image = image[np.newaxis, :]
    r_image = model.predict(x=[image,args.snr_predict*np.ones([1,])])
    r_image = r_image[0]
    psnr = tf.image.psnr(image[0] / 255, r_image / 255, max_val=1)
    ssim = tf.image.ssim(image[0] / 255, r_image / 255, max_val=1)
    img = tf.cast(r_image, tf.uint8)
    cont = tf.image.encode_png(img)
    tf.io.write_file('predict_pic/' + img_name + '_adjscc_imagenet_snrdb' + str(args.snr_predict) + '_fs' + str(args.feedback_snr) + f'dB_{psnr:.2f}dB' + f'_{ssim:.2f}' + '.png', cont)
    print(f"psnr: {psnr} \n ssim: {ssim} ")


def main(args):
    # construct encoder-decoder model
    if args.command == 'train':
        input_imgs = Input(shape=(128, 128, 3))
    elif args.command == 'eval':
        input_imgs = Input(shape=(512, 768, 3))
    elif args.command == 'predict':
        input_imgs = Input(shape=(512, 768, 3))
    input_snrdb = Input(shape=(1,))
    normal_imgs = Lambda(lambda x: x / 255, name='normal')(input_imgs)
    if args.feedback_snr is None:
        input_snrdb_encoder = input_snrdb
        input_snrdb_decoder = input_snrdb
    else:
        input_snrdb_encoder = tf.ones_like(input_snrdb) * args.feedback_snr
        input_snrdb_decoder = tf.ones_like(input_snrdb) * args.feedback_snr
    encoder = Attention_Encoder(normal_imgs, input_snrdb_encoder, args.transmit_channel_num)
    rv = Channel(channel_type='awgn')(encoder, input_snrdb)
    decoder = Attention_Decoder(rv, input_snrdb_decoder)
    rv_imgs = Lambda(lambda x: x * 255, name='denormal')(decoder)
    model = Model(inputs=[input_imgs, input_snrdb], outputs=rv_imgs)
    model.compile(Adam(args.learning_rate), 'mse', run_eagerly=False)
    model.summary()
    if args.command == 'train':
        train(args, model)
    elif args.command == 'eval':
        eval(args, model)
    elif args.command == 'predict':
        predict(args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help='train/eval/eval_pic/')
    parser.add_argument("-ct", '--channel_type', help="awgn", default='awgn')
    parser.add_argument("-md", '--model_dir', help="dir for model", default='model/')
    parser.add_argument("-lmp", '--load_model_path', help="model path for loading")
    parser.add_argument("-bs", "--batch_size", help="Batch size for training", default=96, type=int)
    parser.add_argument("-e", "--epochs", help="epochs for training", default=2, type=int)
    parser.add_argument("-lr", "--learning_rate", help="learning_rate for training", default=0.0001, type=float)
    parser.add_argument("-tcn", "--transmit_channel_num", help="transmit_channel_num for djscc model", default=16,
                        type=int)
    parser.add_argument("-snr_low_train", "--snr_low_train", help="snr_low for training", default=0, type=int)
    parser.add_argument("-snr_up_train", "--snr_up_train", help="snr_up for training", default=20, type=int)
    parser.add_argument("-snr_low_eval", "--snr_low_eval", help="snr_low for evaluation", default=0, type=int)
    parser.add_argument("-snr_up_eval", "--snr_up_eval", help="snr_up for evaluation", default=20, type=int)
    parser.add_argument("-snr_eval", "--snr_eval", help="snr for evaluation", default=10, type=int)
    parser.add_argument("-snr_predict", "--snr_predict", help="snr for predict", default=10, type=int)
    parser.add_argument("-ldd", "--loss_dir", help="loss_dir for training", default='loss/')
    parser.add_argument("-ed", "--eval_dir", help="eval_dir", default='eval/')
    parser.add_argument("-bd", "--board_dir", help="board_dir", default='board/')
    parser.add_argument("-wd", '--work_dir', help="dir for work", default=None)
    parser.add_argument("-fs", "--feedback_snr", help="eval_snr", default=None, type=int)
    global args
    args = parser.parse_args()
    
    # set working directory
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.work_dir is None:
        args.work_dir = f"checkpoint/adjscc_imagenet-{args.batch_size}-{args.transmit_channel_num}ch-{current_time}"
    args.model_dir = os.path.join(args.work_dir, args.model_dir)
    args.eval_dir = os.path.join(args.work_dir, args.eval_dir)
    args.loss_dir = os.path.join(args.work_dir, args.loss_dir)
    args.board_dir = os.path.join(args.work_dir, args.board_dir)
    
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)