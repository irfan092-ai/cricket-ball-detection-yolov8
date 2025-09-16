"""YOLOv8 Training Script"""
from phase1_setup import Config, YOLO, logging

def train_model(yaml_path):
    logging.info("Starting YOLOv8 training...")
    model = YOLO(Config.PRETRAINED_MODEL)
    model.train(
        data=yaml_path,
        epochs=Config.EPOCHS,
        batch=Config.BATCH_SIZE,
        imgsz=Config.IMG_TRAIN_SIZE,
        name=Config.MODEL_NAME
    )
    best_model_path = f"/content/runs/detect/{Config.MODEL_NAME}/weights/best.pt"
    logging.info(f"Training complete. Best model saved at {best_model_path}")
    return best_model_path
