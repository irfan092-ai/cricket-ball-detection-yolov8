"""
Dataset preprocessing & augmentation
"""

import os, cv2, logging
from phase1_setup import Config
from tqdm import tqdm
import albumentations as A

def safe_makedir(path):
    os.makedirs(path, exist_ok=True)

def read_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        logging.warning(f"Could not read image: {img_path}")
    return img

def write_image(img, path):
    cv2.imwrite(path, img)

def write_labels(bboxes, classes, path):
    with open(path, 'w') as f:
        for cls, bbox in zip(classes, bboxes):
            bbox = [max(0.0, min(1.0, x)) for x in bbox]
            f.write(f"{cls} {' '.join([str(round(x,6)) for x in bbox])}\n")

augmentation_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.MotionBlur(p=0.3),
    A.HueSaturationValue(p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def load_image_and_labels(img_path, lbl_path):
    img = read_image(img_path)
    if img is None: return None, [], []
    img = cv2.resize(img, Config.IMG_SIZE)
    bboxes, classes = [], []
    try:
        with open(lbl_path,'r') as f:
            for line in f.read().splitlines():
                parts = line.strip().split()
                cls = int(parts[0])
                bbox = list(map(float, parts[1:]))
                if any(x<0 or x>1 for x in bbox): continue
                bboxes.append(bbox)
                classes.append(cls)
    except Exception as e:
        logging.error(f"Error reading {lbl_path}: {e}")
    return img, bboxes, classes

def save_image_and_labels(img, bboxes, classes, img_name, out_img_folder, out_lbl_folder):
    safe_makedir(out_img_folder)
    safe_makedir(out_lbl_folder)
    img_path = os.path.join(out_img_folder, img_name)
    lbl_path = os.path.join(out_lbl_folder, os.path.splitext(img_name)[0]+'.txt')
    write_image(img, img_path)
    write_labels(bboxes, classes, lbl_path)

def augment_and_save(img, bboxes, classes, img_name, out_img_folder, out_lbl_folder):
    for i in range(Config.AUGMENTATIONS_PER_IMAGE):
        augmented = augmentation_transform(image=img, bboxes=bboxes, class_labels=classes)
        save_image_and_labels(augmented['image'], augmented['bboxes'], augmented['class_labels'],
                              os.path.splitext(img_name)[0]+f'_aug{i}.jpg', out_img_folder, out_lbl_folder)

def preprocess_dataset():
    logging.info("Starting dataset preprocessing...")
    for subset in Config.SUBSETS:
        logging.info(f"Processing subset: {subset}")
        img_folder = os.path.join(Config.DATASET_PATH, subset, 'images')
        lbl_folder = os.path.join(Config.DATASET_PATH, subset, 'labels')
        out_img_folder = os.path.join(Config.OUTPUT_PATH, subset, 'images')
        out_lbl_folder = os.path.join(Config.OUTPUT_PATH, subset, 'labels')
        img_files = [f for f in os.listdir(img_folder) if f.lower().endswith('.jpg')]
        lbl_files = [f for f in os.listdir(lbl_folder) if f.lower().endswith('.txt')]
        valid_images = set(img_files) & set([f.replace('.txt','.jpg') for f in lbl_files])
        for img_file in valid_images:
            img_path = os.path.join(img_folder,img_file)
            lbl_path = os.path.join(lbl_folder, os.path.splitext(img_file)[0]+'.txt')
            img, bboxes, classes = load_image_and_labels(img_path,lbl_path)
            if img is None or len(bboxes)==0: continue
            save_image_and_labels(img, bboxes, classes, img_file, out_img_folder, out_lbl_folder)
            augment_and_save(img, bboxes, classes, img_file, out_img_folder, out_lbl_folder)
    yaml_path = os.path.join(Config.OUTPUT_PATH,'cricket_ball.yaml')
    with open(yaml_path,'w') as f:
        f.write(f"path: {Config.OUTPUT_PATH}\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['ball']\n")
    logging.info(f"Preprocessing complete. YAML saved at {yaml_path}")
    return yaml_path
