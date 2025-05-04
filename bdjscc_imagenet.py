
#%%
from util_channel import Channel
from util_module import Basic_Encoder, Basic_Decoder
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import numpy as np
import argparse
from dataset import dataset_imagenet
import os
import json
import datetime
from yaml import dump

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
AUTOTUNE = tf.data.experimental.AUTOTUNE

#%%

def get_kodak():
    images = np.empty(shape=[0,512,768,3])
    for i in range(1, 25):
        if i<10:
            image_path = 'dataset/kodak/kodim0' + str(i) + '.png'
        else:
            image_path = 'dataset/kodak/kodim' + str(i) + '.png'
        img_file = tf.io.read_file(image_path)
        image = tf.image.decode_png(img_file, channels=3)
        if image.shape[0] == 768:
            image = tf.transpose(image, [1, 0, 2])
        image = image[np.newaxis,:]
        images = np.append(images, image, axis=0)
    return images


def train(args, model):
    if args.load_model_path is not None:
        model.load_weights(args.load_model_path)
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_train) + '_bs' + str(args.batch_size)+'_lr'+str(args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    cbk = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, save_weights_only=True, save_freq=100)
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.board_dir, histogram_freq=1, update_freq=10, profile_batch='500,520')
    
    # for epoch in range(0, args.epochs):
    train_ds, train_nums = dataset_imagenet.get_dataset_snr_from_dir(args.snr_train)
    train_ds = train_ds.shuffle(buffer_size=256, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    # train_step = (train_nums//args.batch_size if train_nums%args.batch_size==0 else train_nums//args.batch_size+1)
    h = model.fit(train_ds, epochs=args.epochs, batch_size=args.batch_size, callbacks=[cbk, tensorboard_callback], workers=8, use_multiprocessing=True)


def train_mix(args, model):
    if args.load_model_path is not None:
        model.load_weights(args.load_model_path)
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdbmix_bs' + str(args.batch_size) + '_lr' + str(
        args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    cbk = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, save_weights_only=True, save_freq=100)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.board_dir, histogram_freq=1, update_freq=10, profile_batch='500,520')
    # for epoch in range(0, args.epochs):
    train_ds, train_nums = dataset_imagenet.get_dataset_snr_range_from_dir(0,20)
    train_ds = train_ds.shuffle(reshuffle_each_iteration=True)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    # train_step = (
    #     train_nums // args.batch_size if train_nums % args.batch_size == 0 else train_nums // args.batch_size + 1)
    h = model.fit(train_ds, epochs=args.epochs, batch_size=args.batch_size, callbacks=[cbk, tensorboard_callback], workers=8, use_multiprocessing=True)


def eval_mismatch(args, model):
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_eval) + '_bs' + str(args.batch_size) + '_lr' + str(
        args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    model.load_weights(model_path)
    snr_list = []
    psnr_list = []
    kodak = get_kodak()
    for snrdb in range(0, 21):
        imse = []
        for i in range(100):
            mse = model.evaluate(x=[kodak, snrdb * np.ones((24,))], y=kodak)
            imse.append(mse)
        mse = np.mean(imse)
        psnr = 10 * np.log10(255 ** 2 / mse)
        snr_list.append(snrdb)
        psnr_list.append(psnr)
    with open(args.eval_dir + filename + '.json', mode='w') as f:
        json.dump({'snr': snr_list, 'psnr': psnr_list}, f)


def eval_pic(args, model):
    filename = os.path.basename(__file__).split('.')[0] + '_' + str(args.channel_type) + '_tcn' + str(
        args.transmit_channel_num) + '_snrdb' + str(args.snr_eval) + '_bs' + str(args.batch_size) + '_lr' + str(
        args.learning_rate)
    model_path = args.model_dir + filename + '.h5'
    model.load_weights(model_path)
    image_path = 'dataset/kodak/kodim03.png'
    img_file = tf.io.read_file(image_path)
    image = tf.image.decode_png(img_file, channels=3)
    image = image[np.newaxis,:]
    mse = model.evaluate(x=[image,args.snr_eval*np.ones((2,))], y=image)
    print(mse)


def main(args):
    # construct encoder-decoder model
    if args.command == 'train' or args.command == 'train_mix':
        input_imgs = Input(shape=(128, 128, 3))
    elif args.command == 'eval_mismatch' or args.command == 'eval_pic':
        input_imgs = Input(shape=(512, 768, 3))
    input_snrdb = Input(shape=(1,))
    normal_imgs = Lambda(lambda x: x / 255, name='normal')(input_imgs)
    encoder = Basic_Encoder(normal_imgs, args.transmit_channel_num)
    rv = Channel(channel_type='awgn')(encoder, input_snrdb)
    decoder = Basic_Decoder(rv)
    rv_imgs = Lambda(lambda x: x * 255, name='denormal')(decoder)
    model = Model(inputs=[input_imgs, input_snrdb], outputs=rv_imgs)
    model.compile(optimizer=Adam(args.learning_rate), loss=tf.keras.losses.MeanSquaredError())
    model.summary()
    if args.command == 'train':
        train(args, model)
    elif args.command == 'train_mix':
        train_mix(args, model)
    elif args.command == 'eval_mismatch':
        eval_mismatch(args, model)
    elif args.command == 'eval_pic':
        eval_pic(args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help='train/train_mix/eval_mismatch/eval_pic')
    parser.add_argument("-ct", '--channel_type', default='awgn', help="awgn/slow_fading/slow_fading_eq")
    parser.add_argument("-wd", '--work_dir', help="dir for work", default='work/')
    parser.add_argument("-md", '--model_dir', help="dir for model", default='model/')
    parser.add_argument("-lmp", '--load_model_path', help="model path for loading")
    parser.add_argument("-bs", "--batch_size", help="Batch size for training", default=96, type=int)
    parser.add_argument("-e", "--epochs", help="epochs for training", default=100, type=int)
    parser.add_argument("-lr", "--learning_rate", help="learning_rate for training", default=0.0001, type=float)
    parser.add_argument("-tcn", "--transmit_channel_num", help="transmit_channel_num for djscc model", default=16,
                        type=int)
    parser.add_argument("-snr_train", "--snr_train", help="snr for training", default=10, type=int)
    parser.add_argument("-snr_eval", "--snr_eval", help="snr for evaluation", default=10, type=int)
    parser.add_argument("-ldd", "--loss_dir", help="loss_dir for training", default='loss/')
    parser.add_argument("-ed", "--eval_dir", help="eval_dir", default='eval/')
    parser.add_argument("-bd", "--board_dir", help="board_dir", default='board/')
    global args
    args = parser.parse_args()
    
    # correct the directory
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.work_dir = f"checkpoint/bdjscc_imagenet-{args.snr_train}dB-{args.batch_size}-{args.transmit_channel_num}ch-{current_time}"
    args.model_dir = os.path.join(args.work_dir, args.model_dir)
    args.evl_dir = os.path.join(args.work_dir, args.eval_dir)
    args.loss_dir = os.path.join(args.work_dir, args.loss_dir)
    args.board_dir = os.path.join(args.work_dir, args.board_dir)
    
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.evl_dir, exist_ok=True)
    os.makedirs(args.loss_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, 'args.yaml'), 'w') as f:
        dump(args, f)
    main(args)
