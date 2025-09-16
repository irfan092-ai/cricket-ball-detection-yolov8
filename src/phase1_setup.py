"""
Expert-Level Cricket Ball Detection Pipeline – Phase 1 Setup
Author: Muhammad Irfan
"""
!pip install --upgrade ultralytics
import os
import cv2
import shutil
import logging
from tqdm import tqdm
import glob
import albumentations as A
from ultralytics import YOLO
from IPython.display import display, Image
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler()]
)

class Config:
    DATASET_PATH = '/content/data'
    OUTPUT_PATH = '/content/cricket_ball_aug'
    PREDICTION_PATH = '/content/predictions'
    IMG_SIZE = (640, 640)
    AUGMENTATIONS_PER_IMAGE = 2
    SUBSETS = ['train', 'valid', 'test']
    EPOCHS = 50
    BATCH_SIZE = 16
    IMG_TRAIN_SIZE = 640
    MODEL_NAME = 'cricket_ball_train'
    PRETRAINED_MODEL = 'yolov8n.pt'
    CONF_THRESHOLD = 0.25
