'''
Create examples for paper of YOLOv8 method.
'''
from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 nano model
model = YOLO("sample_rico/yolov8_custom_model_4.pt")

val_dataset_path = "sample_rico/augmented_images/images/val"
image_paths = ["sample_rico/augmented_images/images/val/35474.jpg","sample_rico/augmented_images/images/val/6559.jpg"]

output_dir = "yolo_validation_outputs_new"
os.makedirs(output_dir, exist_ok=True)

# Run inference on validation images
results = model.predict(image_paths)

for idx, number in zip(range(2),[35474,6559]):
    result = results[idx]
    img_path = image_paths[idx]
    img = cv2.imread(img_path)

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert box to integer values
        conf = box.conf[0].item()
        class_id = int(box.cls[0].item())
        label = f"{model.names[class_id]} {conf:.2f}" 

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 8)

    # Save image
    output_path = os.path.join(output_dir, f"output_{idx}.jpg")
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
