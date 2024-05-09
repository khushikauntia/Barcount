from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
from detectron2.utils.visualizer import ColorMode

import streamlit as st
import random
import cv2
import matplotlib.pyplot as plt

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    
    for s in random.sample(dataset_custom,n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata = dataset_custom_metadata, scale = 0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    
    cfg.DATALOADER.NUM_WORKERS = 2
    
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 200
    cfg.SOLVER.STEPS = []
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    
    return cfg

def on_image(im, predictor):
    file_bytes = np.asarray(bytearray(im.read()), dtype=np.uint8)
    im = cv2.imdecode(file_bytes, 1)
    outputs = predictor(im)
    v = Visualizer(im[:, :,::-1],metadata={},scale=0.5,instance_mode = ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(20,5))
    plt.imshow(v.get_image())
    plt.axis("off")

# Save the plot as a PNG image
    plt.savefig("image.png")

# Display the image in Streamlit
    st.image("image.png")
    class_id=0
    num_detections=len(outputs["instances"][outputs["instances"].pred_classes == class_id])
    ans='Number of bars: '+str(num_detections)
    st.write(ans)