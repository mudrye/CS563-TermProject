import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import glob
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_recall_curve, auc

#Taken from NightHawk paper custum data used help of chatgpt


# Custom Dataset (Pascal VOC Format)
class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, class_names, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.class_names = class_names
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".xml"))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = T.ToTensor()(img)
        
        h, w, _ = img.shape
        boxes, labels = [], []
        
        if os.path.exists(label_path):
            tree = ET.parse(label_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name in self.class_names:
                    class_id = self.class_names.index(class_name)
                    bbox = obj.find("bndbox")
                    x_min = float(bbox.find("xmin").text)
                    y_min = float(bbox.find("ymin").text)
                    x_max = float(bbox.find("xmax").text)
                    y_max = float(bbox.find("ymax").text)

                    # Ensure the box has a valid height and width
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)}
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, target


# Custom collate function
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# Load Model
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Evaluation Function
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_targets, all_scores = [], [], []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                # Extract predicted labels
                pred_labels = output['labels'].cpu().numpy() if 'labels' in output else []
                # Extract predicted scores
                pred_scores = output['scores'].cpu().numpy() if 'scores' in output else []
                # Extract target labels
                target_labels = target['labels'].cpu().numpy() if 'labels' in target else []
                
                if len(pred_labels) == 0:
                    pred_labels = [0]
                    pred_scores = [0.0]  # Add default score for empty predictions
                if len(target_labels) == 0:
                    target_labels = [0]
                
                # Save predicted labels and scores
                all_preds.extend(pred_labels)
                all_scores.extend(pred_scores)  # Extend the list with predicted scores
                all_targets.extend(target_labels)
    
    min_len = min(len(all_preds), len(all_targets))
    all_preds, all_targets = all_preds[:min_len], all_targets[:min_len]
    print(all_targets)
    precision, recall, f1, ap_values = [], [], [], []
    for c in range(len(class_names)):
        y_true = np.array([1 if t == c else 0 for t in all_targets])
        y_scores = np.array([s if p == c else 0 for p, s in zip(all_preds, all_scores)])

        # Compute precision-recall curve and AUC for AP calculation
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        ap = auc(rec, prec)  # Compute AP as area under PR curve
        ap_values.append(ap)

        tp = sum((p == c and t == c) for p, t in zip(all_preds, all_targets))
        fp = sum((p == c and t != c) for p, t in zip(all_preds, all_targets))
        fn = sum((p != c and t == c) for p, t in zip(all_preds, all_targets))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1.append(2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0)
        precision.append(prec)
        recall.append(rec)
    mean_ap = np.mean(ap_values)
    mean_ar = np.mean(recall)
    
    # Print and save results
    metrics = "Evaluation Metrics per Class:\n"
    for i, class_name in enumerate(class_names[1:]):  # Skip background
        metrics += f"{class_name}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-score: {f1[i]:.4f}, AP: {ap_values[i]:.4f}\n"
    metrics += f"\nMean Average Precision (mAP): {mean_ap:.4f}\n"
    metrics += f"Mean Average Recall (mAR): {mean_ar:.4f}\n"

    print(metrics)
    with open("evaluation_metrics.txt", "w") as f:
        f.write(metrics)

# Training Function
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10, class_names=[]):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    print("Training complete!")
    torch.save(model.state_dict(), "faster_rcnn_model.pth")
    

    evaluate_model(model, val_loader, device, class_names)

# Main Execution
def main():
    class_names = ['background', 'text_overlap', 'component_occlusion']
    train_dataset = CustomDataset("sample_rico/augmented_images/images/train", "voc_train_annotations", class_names)
    val_dataset = CustomDataset("sample_rico/augmented_images/images/val", "voc_val_annotations", class_names)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    num_classes = len(class_names)
    model = get_model(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, optimizer, device, num_epochs=100, class_names=class_names)

if __name__ == "__main__":
    main()
