import os
from ultralytics import YOLO

# Dataset structure:
# ├── augmented_images/
# │   ├── images/
# │   │   ├── train/
# │   │   ├── val/
# │   ├── labels/
# │   │   ├── train/
# │   │   ├── val/
# ├── yolo-display.yaml


dataset_path = "sample_rico/augmented_images/images"

yaml_file_path = "sample_rico/augmented_images/yolo-display.yaml"

# Define the model and load the pre-trained weights
model = YOLO("yolov8n.pt")  # You can choose different versions (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)

# Start training and save the trained model
model.train(data=yaml_file_path, epochs=400, imgsz=640, batch=16, device=0)
model.save("sample_rico/yolov8_custom_model_4.pt")

# Evaluate the model on the validation set
metrics = model.val(data=yaml_file_path)
metrics.results_dict
print(metrics.box.r)
print(metrics.box.ap)


