"""YOLOv8 Prediction Script"""
import os, shutil, glob, cv2
from phase1_setup import Config, YOLO
import matplotlib.pyplot as plt
from IPython.display import display, Image

class YOLOPredictor:
    def __init__(self, model_path, output_folder=Config.PREDICTION_PATH):
        self.model = YOLO(model_path)
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def predict_image(self, img_path, conf_threshold=Config.CONF_THRESHOLD):
        results = self.model.predict(source=img_path, conf=conf_threshold, save=True, save_txt=True)
        for r in results:
            plotted = r.plot()
            plt.figure(figsize=(8,6))
            plt.imshow(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
        pred_folder = results[0].save_dir
        saved_images = glob.glob(os.path.join(pred_folder,"*.jpg"))
        for img_file in saved_images:
            dest_name = "pred_" + os.path.basename(img_path)
            dest_path = os.path.join(self.output_folder,dest_name)
            shutil.copy(img_file,dest_path)
            display(Image(filename=dest_path))

    def predict_folder(self, folder_path, conf_threshold=Config.CONF_THRESHOLD):
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.jpg','.jpeg','.png')):
                self.predict_image(os.path.join(folder_path,img_file), conf_threshold)
