import mask_rcnn as modellib
import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm
from inference_config import inference_config
from dsb_dataset import DSBDataset
from utils import rle_encode, rle_decode, rle_to_string
import functions as f

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = model.find_last()[1]
model_path = "./logs/dsb20180322T1409/mask_rcnn_dsb_0030.h5"
name, ext = os.path.splitext(os.path.basename(model_path))

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_test = DSBDataset()
dataset_test.load_bowl('../dataset/DSB/test')
dataset_test.prepare()

output = []
sample_submission = pd.read_csv('../dataset/DSB/stage1_sample_submission.csv')
ImageId = []
EncodedPixels = []
for image_id in tqdm(sample_submission.ImageId):
    image_path = os.path.join('../dataset/DSB/test', image_id, 'images', image_id + '.png')
    original_image = cv2.imread(image_path)
    results = model.detect([original_image], verbose=0)
    r = results[0]
    masks = r['masks']
    ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, image_id, r['scores'])
    ImageId += ImageId_batch
    EncodedPixels += EncodedPixels_batch



f.write2csv('./submit/{}.csv'.format(name), ImageId, EncodedPixels)
