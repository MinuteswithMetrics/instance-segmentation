# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import argparse
import datetime
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.backend import tensorflow_backend as KTF

import mask_rcnn as modellib
from mask_rcnn import log
import utils
import visualize

from dsb_config import DSBConfig
from dsb_dataset import DSBDataset


def main(args):
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # data path
    DATA_PATH = args.data_path

    # Get train and test IDs
    data_ids = next(os.walk(DATA_PATH))[1]

    # create validation ids
    train_ids, val_ids = train_test_split(data_ids, test_size=0.1)

    # dataset name
    dataset_name = os.path.basename(args.data_path)

    # dataset config
    config = DSBConfig()
    config.NAME = dataset_name
    config.STEPS_PER_EPOCH  = int(len(data_ids)/(config.GPU_COUNT*config.IMAGES_PER_GPU))
    config.VALIDATION_STEPS = int(len(val_ids)/(config.GPU_COUNT*config.IMAGES_PER_GPU))
    config.display()

    # dataset
    # Training dataset
    input_size = args.input_size
    dataset_train = DSBDataset()
    dataset_train.load_bowl(args.data_path)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSBDataset()
    dataset_val.load_bowl(args.data_path)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)


    if args.init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif args.init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(
                COCO_MODEL_PATH,
                by_name=True,
                exclude=[
                    "mrcnn_class_logits",
                    "mrcnn_bbox_fc",
                    "mrcnn_bbox",
                    "mrcnn_mask"])
    elif args.init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    else:
        print("wrong mode")

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    if args.freeze:
        model.train(
                dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args.freeze_epochs,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=args.epochs,
            layers="all")
    model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=args.epochs * 2,
            layers="all")
    model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE/100,
            epochs=args.epochs * 3,
            layers="all")


    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    #now = datetime.datetime.today()
    #model_path = os.path.join(MODEL_DIR, str(now.day)+"-"+str(now.hour)+"-"+"DSB.h5")
    #model.keras_model.save_weights(model_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
            description='SENet')
    argparser.add_argument(
            '-d',
            '--data_path',
            type=str,
            default="../../dataset/DSB/stage1_train/",
            help='path to data list')
    argparser.add_argument(
            '-s',
            '--input_size',
            type=int,
            default=320,
            help='input image size')
    argparser.add_argument(
            '-e',
            '--epochs',
            type=int,
            default=10,
            help='number of epochs')
    argparser.add_argument(
            '-g',
            '--gpu',
            type=str,
            default="0",
            help='number of gpu')
    argparser.add_argument(
            '--init_with',
            type=str,
            default="imagenet",
            help='init with pretrained weights')
    argparser.add_argument(
            '--freeze',
            type=bool,
            default=True,
            help='freeze weights except heads layers')
    argparser.add_argument(
            '--freeze_epochs',
            type=int,
            default=5,
            help='epochs of freeze weights except heads layers')
    args = argparser.parse_args()

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    session = tf.Session(config=config)
    KTF.set_session(session)
    """

    main(args)
