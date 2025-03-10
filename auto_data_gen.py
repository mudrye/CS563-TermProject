import json
import random
import cv2
import numpy as np
import os

'''
Algorithm from the NighHawk paper https://arxiv.org/abs/2205.13945 used to create 
the augmented data for text overlap and opponent occlusion. Chatgpt was also used.
'''

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

def generate_text_overlap(image, ui_data):
    """
    Generates an overlapping text label on the given image for each detected 'TextView' UI element.
    If no valid 'TextView' elements are found, the function returns the original image with None.

    Args:
        image (numpy.ndarray): The input image where the text should be overlaid.
        ui_data (dict): A structured JSON-like dictionary containing UI component information.

    Returns:
        tuple: A modified image with text overlay and the bounding box (x1, y1, x2, y2) of the placed text.
    """
    elements = [e for e in get_ui_elements(ui_data) if e['class'] == 'android.widget.TextView']
    
    for elem in elements:
        x1, y1, x2, y2 = elem['bounds']
        w, h = x2 - x1, y2 - y1
        
        if w > 0 and h > 0:
            text = elem.get('text', 'Sample Text')
            font_scale = max(0.5, min(3, h / 30))
            font_thickness = 5
            
            # Placing text at the same x and y coordinates to fully overlap
            overlap_x1 = max(x1, 0)

            overlap_y1 = max(y1 + h // 2, 0)  # Aligning with the text baseline
            
            # Sample background color from an area near the text
            sample_x = min(max(x1 - 5, 0), image.shape[1] - 1)
            sample_y = min(max(y1 - 5, 0), image.shape[0] - 1)
            bg_color = np.array(image[sample_y, sample_x].tolist())
            
            # Choose a contrasting color for the text
            text_color = (255, 255, 255) if np.mean(bg_color) < 128 else (0, 0, 0)
            
            # Get text size to adjust bounding box
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_w, text_h = text_size
            
            # Draw the text at the correct overlap location
            cv2.putText(image, text, (overlap_x1, overlap_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            overlap_bbox = (overlap_x1, overlap_y1 - text_h, overlap_x1 + text_w, overlap_y1)
            
            return image, overlap_bbox
    
    print("No valid TextView found for text overlap.")
    return image, None

def generate_component_occlusion(image, ui_data):
    """
    Applies a partial occlusion over detected UI components in the given image.
    If no valid UI components are found, the function returns the original image with None.

    Args:
        image (numpy.ndarray): The input image where occlusion should be applied.
        ui_data (dict): A structured JSON-like dictionary containing UI component information.

    Returns:
        tuple: A modified image with the occlusion applied and the bounding box 
               (x1, y1, x2, y2) of the occlusion area.
    """
    elements = get_ui_elements(ui_data)
    
    for elem in elements:
        x1, y1, x2, y2 = elem['bounds']
        w, h = x2 - x1, y2 - y1
        
        if w > 0 and h > 0:
            occlusion_height = int(h * 0.5)
            
            occlusion_x1, occlusion_x2 = x1, x2
            occlusion_y1 = y1
            occlusion_y2 = occlusion_y1 + occlusion_height
            
            # Sample background color from nearby pixels
            sample_x = min(max(x1 + 5, 0), image.shape[1] - 1)
            sample_y = min(max(y1 + h // 2, 0), image.shape[0] - 1)
            occlusion_color = tuple(image[sample_y, sample_x].tolist())
            
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
    """
    Reads images from the input folder and splits them into training and validation sets.
    Loads UI component data from JSON files associated with each image. Randomly applies either 
    a text overlap or component occlusion transformation. Saves the modified image and its 
    bounding box annotation in YOLO format.

    Args:
        input_folder (str): Path to the folder containing original images and JSON UI data.
        output_folder (str): Path to the destination folder for processed images and labels.

    Returns:
        None: The function saves the processed images and label files to the output folder.
    """
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


process_folder("sample_rico/combined", "sample_rico/augmented_images")
