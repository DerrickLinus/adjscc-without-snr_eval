#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/jay/git/ADJSCC/dataset_gen.py
# Project: ADJSCC
# Created Date: Friday, April 14th 2023, 1:38:09 pm
# Author: Shisui
# Copyright (c) 2023 Uchiha
# ----------	---	----------------------------------------------------------
###

# %%
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import multiprocessing as mp
from tqdm import tqdm
import tf

# %%


def get_image_path_list(dir):
    extens = [".jpg", ".png", ".jpeg", ".JPEG"]
    path_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] in extens:
                path_list.append(os.path.join(root, file))
    return path_list


def image_crop(
    image_path,
    save_dir="/home/jay/Documents/datasets/imagenet/ILSVRC-crop_128",
    size=128,
):
    img = Image.open(image_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img_w, img_h = img.size
    image_name = os.path.basename(image_path)
    if img_w >= size and img_h >= size:
        for i in range(img_w // size):
            for j in range(img_h // size):
                box = (i * size, j * size, (i + 1) * size, (j + 1) * size)
                img.crop(box).save(
                    os.path.join(save_dir, str(i) + "_" + str(j) + "-" + image_name)
                )


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.




def main(input_dir, output_dir, size=128, workers=32):
    image_path_list = get_image_path_list(input_dir)
    print("image num: ", len(image_path_list))
    input_data = [image_path_list]
    # crop_func = image_crop_wrapper(save_dir=output_dir, size=size)
    with mp.Pool(processes=workers) as pool:
        with tqdm(total=len(image_path_list)) as pbar:
            for _ in pool.imap_unordered(image_crop, image_path_list):
                pbar.update(1)
    print("done")


# %%
if __name__ == "__main__":
    input_dir = "/home/jay/Documents/datasets/imagenet/ILSVRC/Data/CLS-LOC/train"
    output_dir = "/home/jay/Documents/datasets/imagenet/ILSVRC-crop_128"
    main(input_dir, output_dir, size=128)
