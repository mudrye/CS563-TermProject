'''
Convert the yolo txt files to xml. Used chatgpt to help.
'''
import os
import glob
import cv2
import xml.etree.ElementTree as ET

def yolo_to_voc(yolo_bbox, img_width, img_height):
    class_id, x_center, y_center, width, height = map(float, yolo_bbox)
    x_min = int((x_center - width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    x_max = int((x_center + width / 2) * img_width)
    y_max = int((y_center + height / 2) * img_height)
    return class_id, max(0, x_min), max(0, y_min), min(img_width, x_max), min(img_height, y_max)

def create_voc_xml(img_path, voc_bboxes, class_names, output_dir):
    img_filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = img_filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"
    
    for bbox in voc_bboxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = class_names[int(bbox[0])]
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox[1])
        ET.SubElement(bndbox, "ymin").text = str(bbox[2])
        ET.SubElement(bndbox, "xmax").text = str(bbox[3])
        ET.SubElement(bndbox, "ymax").text = str(bbox[4])
    
    tree = ET.ElementTree(annotation)
    xml_output_path = os.path.join(output_dir, img_filename.replace(".jpg", ".xml"))
    tree.write(xml_output_path)
    print(f"Saved: {xml_output_path}")

def process_dataset(yolo_labels_dir, images_dir, output_dir, class_names):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    for img_path in image_paths:
        label_path = os.path.join(yolo_labels_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            continue
        
        with open(label_path, "r") as f:
            yolo_bboxes = [line.strip().split() for line in f.readlines()]
        
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape
        voc_bboxes = [yolo_to_voc(bbox, img_width, img_height) for bbox in yolo_bboxes]
        create_voc_xml(img_path, voc_bboxes, class_names, output_dir)
        
        # Draw bounding boxes for visualization
        for bbox in voc_bboxes:
            cv2.rectangle(img, (bbox[1], bbox[2]), (bbox[3], bbox[4]), (0, 255, 0), 2)
            cv2.putText(img, class_names[int(bbox[0])], (bbox[1], bbox[2] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # preview_path = os.path.join(output_dir, "preview_" + os.path.basename(img_path))
        # cv2.imwrite(preview_path, img)
        # print(f"Saved preview: {preview_path}")

if __name__ == "__main__":
    class_names = ["component_occlusion","text_overlap"]
    yolo_labels_dir = "sample_rico/augmented_images/labels/train"
    images_dir = "sample_rico/augmented_images/images/train"
    output_dir = "voc_train_annotations"
    process_dataset(yolo_labels_dir, images_dir, output_dir, class_names)

    # Process validation data
    val_yolo_labels_dir = "sample_rico/augmented_images/labels/val"
    val_images_dir = "sample_rico/augmented_images/images/val"
    val_output_dir = "voc_val_annotations"
    process_dataset(val_yolo_labels_dir, val_images_dir, val_output_dir, class_names)
