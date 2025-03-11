# Automatically Detecting and Localizing Display Issues in User Interfaces
## Ella Mudry, Swathi Gangi, Tabitha Rowland
### Steps to Run
1. run `data_generation/create_small_dataset.py` to get the cluster sample
2. run `data_generation/combined_folder` to but all the images in the same folder
3. run `data_generation/auto_data_gen.py` to get the augmentations
4. run `train.py` to bootstrap yolov8
5. run `convert_yolo2xml.py` to get the correct data format to train for nighthawk
6. run `train_NightHawk.py` to train and validate the Faster-RCNN
7. `yolo_image_output.py` creates example images with save model
8. `nighthawk_output.py` creates examples on val image set for trained model 

### Rico dataset
rico dataset gotten from http://www.interactionmining.org/rico.html
`unique_uis.tar.gz` `ui_details.csv` and `app_details.csv` needed to run the code

### Data Generation 
Contains the augmented dataset used