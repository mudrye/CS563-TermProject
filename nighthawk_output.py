'''
Create examples to show for nighthawk results.
Chatgpt used to help create script.
'''
import torch
import torchvision.transforms as T
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract number of classes from checkpoint
    num_classes = checkpoint['roi_heads.box_predictor.cls_score.weight'].shape[0]
    
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace the classifier head with the correct number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor()
    ])
    return transform(image), image

def draw_boxes(image, boxes, scores, threshold=0.5):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    for box, score in zip(boxes, scores):
        if score >= threshold:
            draw.rectangle(box.tolist(), outline="lime", width=5)  # Neon green border
            draw.text((box[0], box[1]), f"{score:.2f}", fill="lime", font=font)
    return image

def run_inference(model, val_images_dir, output_dir, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in os.listdir(val_images_dir):
        img_path = os.path.join(val_images_dir, img_name)
        image_tensor, image = preprocess_image(img_path)
        
        with torch.no_grad():
            prediction = model([image_tensor])
        
        boxes, scores = prediction[0]['boxes'], prediction[0]['scores']
        
        # Only save images where at least one box meets the threshold
        if any(score >= threshold for score in scores):
            output_image = draw_boxes(image, boxes, scores, threshold)
            output_image.save(os.path.join(output_dir, img_name))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Faster R-CNN on validation images.")
    parser.add_argument("model_path", type=str, help="Path to the Faster R-CNN model.")
    parser.add_argument("val_images_dir", type=str, help="Directory containing validation images.")
    parser.add_argument("output_dir", type=str, help="Directory to save output images.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for displaying boxes.")
    
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    run_inference(model, args.val_images_dir, args.output_dir, args.threshold)