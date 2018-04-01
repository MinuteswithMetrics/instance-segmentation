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

def main(model_paths, cluster, output_dir):
    # test dataset
    dataset_test = DSBDataset()
    dataset_test.load_bowl('../dataset/DSB/stage1_test')
    dataset_test.prepare()

    # Recreate the model in inference mode
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    model = modellib.MaskRCNN(
            mode="inference",
            config=inference_config,
            model_dir=MODEL_DIR)

    for model_path in tqdm(model_paths):
        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        name, ext = os.path.splitext(os.path.basename(model_path))
        output = []
        sample_submission = pd.read_csv('../dataset/DSB/stage1_sample_submission.csv')
        ImageId = []
        EncodedPixels = []
        for image_id in sample_submission.ImageId:
            image_path = os.path.join('../dataset/DSB/stage1_test', image_id, 'images', image_id + '.png')
            original_image = cv2.imread(image_path)
            results = model.detect([original_image], verbose=0)
            r = results[0]
            masks = r['masks']
            ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, image_id, r['scores'])
            ImageId += ImageId_batch
            EncodedPixels += EncodedPixels_batch
        # save csv
        df = f.write2csv(ImageId, EncodedPixels)
        if cluster:
            df = f.tocluster(df, cluster)
        if not os.path.exists('./submit/{}'.format(output_dir)):
            os.mkdir('./submit/{}'.format(output_dir))
        df.to_csv('./submit/{}/{}.csv'.format(output_dir, name), index=False, columns=['ImageId', 'EncodedPixels'])


if __name__ == "__main__":
    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    #model_path = model.find_last()[1]
    cluster = ""
    output_dir = "1e-3all"
    model_paths = [
            "./logs/train_all20180401T1513/mask_rcnn_train_all_0015.h5",
            "./logs/train_all20180401T1513/mask_rcnn_train_all_0020.h5",
            "./logs/train_all20180401T1513/mask_rcnn_train_all_0025.h5",
            "./logs/train_all20180401T1513/mask_rcnn_train_all_0030.h5",
            ]

    main(model_paths, cluster, output_dir)
