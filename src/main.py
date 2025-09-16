"""Main Execution Script"""
from preprocess import preprocess_dataset
from train_model import train_model

if __name__=="__main__":
    skip_preprocessing = False
    skip_training = False

    if not skip_preprocessing:
        yaml_path = preprocess_dataset()
    else:
        yaml_path = '/content/cricket_ball_aug/cricket_ball.yaml'

    if not skip_training:
        best_model_path = train_model(yaml_path)
    else:
        best_model_path = '/content/runs/detect/cricket_ball_train/weights/best.pt'
