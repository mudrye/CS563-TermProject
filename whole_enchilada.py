import json
import random
import cv2
import numpy as np
import os

def load_rico_data(image_path, json_path):
    image = cv2.imread(image_path)
    with open(json_path, 'r') as f:
        ui_data = json.load(f)
    return image, ui_data

def get_ui_elements(ui_data):
    elements = []
    
    def traverse(node):
        if isinstance(node, dict):
            if 'class' in node and node['class'] in ['android.widget.TextView', 'android.widget.ImageView', 'android.widget.Button', 'android.widget.EditText']:
                if 'bounds' in node and isinstance(node['bounds'], list) and len(node['bounds']) == 4:
                    elements.append(node)
            if 'children' in node and isinstance(node['children'], list):
                for child in node['children']:
                    traverse(child)
    
    if 'activity' in ui_data and 'root' in ui_data['activity']:
        traverse(ui_data['activity']['root'])
    
    return elements

# def normalize_bbox(bbox, image_shape):
#     x_center = (bbox[0] + bbox[2]) / 2 / image_shape[1]
#     y_center = (bbox[1] + bbox[3]) / 2 / image_shape[0]
#     width = (bbox[2] - bbox[0]) / image_shape[1]
#     height = (bbox[3] - bbox[1]) / image_shape[0]
#     return max(0, min(x_center, 1)), max(0, min(y_center, 1)), max(0, min(width, 1)), max(0, min(height, 1))

# def split_train_val(files, train_ratio=0.8):
#     random.shuffle(files)
#     split_idx = int(len(files) * train_ratio)
#     return files[:split_idx], files[split_idx:]

def generate_text_overlap(image, ui_data):
    elements = [e for e in get_ui_elements(ui_data) if e['class'] == 'android.widget.TextView']
    
    for elem in elements:
        x1, y1, x2, y2 = elem['bounds']
        w, h = x2 - x1, y2 - y1
        
        if w > 0 and h > 0:
            text = elem.get('text', 'Sample Text')
            font_scale = max(0.5, min(1.5, h / 30))
            font_thickness = 2
            font_color = random.choice([(0, 0, 0), (128, 128, 128), (255, 255, 255)])
            
            x_offset = random.randint(-max(1, int(0.2 * w)), max(1, int(0.2 * w)))
            y_offset = random.randint(-max(1, int(0.2 * h)), max(1, int(0.2 * h)))
            
            overlap_x1 = max(x1 + x_offset, 0)
            overlap_y1 = max(y1 + y_offset, 0)
            overlap_x2 = min(overlap_x1 + w, image.shape[1])
            overlap_y2 = min(overlap_y1 + h, image.shape[0])
            
            cv2.putText(image, text, (overlap_x1, overlap_y1 + int(h * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
            overlap_bbox = (overlap_x1, overlap_y1, overlap_x2, overlap_y2)
            cv2.rectangle(image, (overlap_bbox[0], overlap_bbox[1]), (overlap_bbox[2], overlap_bbox[3]), (255, 0, 0), 2)
            
            return image, overlap_bbox
    
    print("No valid TextView found for text overlap.")
    return image, None

def generate_component_occlusion(image, ui_data):
    elements = get_ui_elements(ui_data)
    
    for elem in elements:
        x1, y1, x2, y2 = elem['bounds']
        w, h = x2 - x1, y2 - y1
        
        if w > 0 and h > 0:
            rand_factor = random.uniform(-1, 1)
            occlusion_height = int(h * abs(rand_factor))
            
            occlusion_x1, occlusion_x2 = x1, x2
            occlusion_y1 = y1 if rand_factor >= 0 else y2 - occlusion_height
            occlusion_y2 = occlusion_y1 + occlusion_height
            
            upper_left_color = image[y1, x1] if 0 <= y1 < image.shape[0] and 0 <= x1 < image.shape[1] else (0, 0, 0)
            upper_right_color = image[y1, x2 - 1] if 0 <= y1 < image.shape[0] and 0 <= x2 - 1 < image.shape[1] else (0, 0, 0)
            occlusion_color = tuple(map(int, np.mean([upper_left_color, upper_right_color], axis=0)))
            
            cv2.rectangle(image, (occlusion_x1, occlusion_y1), (occlusion_x2, occlusion_y2), occlusion_color, -1)
            occlusion_bbox = (occlusion_x1, occlusion_y1, occlusion_x2, occlusion_y2)
            
            return image, occlusion_bbox
    
    print("No valid UI component found for occlusion.")
    return image, None

def normalize_bbox(bbox, image_shape):
    x_center = (bbox[0] + bbox[2]) / 2 / image_shape[1]
    y_center = (bbox[1] + bbox[3]) / 2 / image_shape[0]
    width = (bbox[2] - bbox[0]) / image_shape[1]
    height = (bbox[3] - bbox[1]) / image_shape[0]
    return max(0, min(x_center, 1)), max(0, min(y_center, 1)), max(0, min(width, 1)), max(0, min(height, 1))

def split_train_val(files, train_ratio=0.8):
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    return files[:split_idx], files[split_idx:]

def process_folder(input_folder, output_folder):
    yolo_cat = {"component_occlusion":0, "text_overlap":1}
    images = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    train_files, val_files = split_train_val(images)
    
    for dataset, files in [("train", train_files), ("val", val_files)]:
        img_folder = os.path.join(output_folder, "images", dataset)
        lbl_folder = os.path.join(output_folder, "labels", dataset)
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(lbl_folder, exist_ok=True)
        
        for image_file in files:
            image_path = os.path.join(input_folder, image_file)
            json_path = os.path.join(input_folder, image_file.replace('.jpg', '.json'))
            
            if not os.path.exists(json_path):
                continue
            
            image, ui_data = load_rico_data(image_path, json_path)
            category = random.choice(['text_overlap', 'component_occlusion'])
            
            if category == 'text_overlap':
                modified_image, bbox = generate_text_overlap(image, ui_data)
            else:
                modified_image, bbox = generate_component_occlusion(image, ui_data)
            
            output_image_path = os.path.join(img_folder, image_file)
            cv2.imwrite(output_image_path, modified_image)
            
            if bbox:
                output_label_path = os.path.join(lbl_folder, image_file.replace('.jpg', '.txt'))
                with open(output_label_path, 'w') as f:
                    f.write(f"{yolo_cat[category]} {' '.join(map(str, normalize_bbox(bbox, image.shape)))}\n")


# Example usage
process_folder("sample_rico/combined", "sample_rico/augmented_images")
