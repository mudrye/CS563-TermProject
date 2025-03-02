import json
import random
import csv
import os
import tarfile
import shutil
from collections import defaultdict

def load_rico_metadata(csv_path):
    """Load RICO dataset metadata from CSV."""
    category_dict = defaultdict(list)
    app_details = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            app_package = row['App Package Name']
            category = row['Category']
            app_details[app_package] = {
                'Play Store Name': row['Play Store Name'],
                'Average Rating': row['Average Rating'],
                'Number of Ratings': row['Number of Ratings'],
                'Number of Downloads': row['Number of Downloads'],
                'Date Updated': row['Date Updated'],
                'Icon URL': row['Icon URL']
            }
            category_dict[category].append(app_package)
    
    return category_dict, app_details

def load_ui_details(ui_csv_path):
    """Load UI details from CSV."""
    ui_details = defaultdict(list)
    
    with open(ui_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            app_package = row['App Package Name']
            ui_number = row['UI Number']
            ui_details[app_package].append({
                'UI Number': ui_number,
                'Interaction Trace Number': row['Interaction Trace Number'],
                'UI Number in Trace': row['UI Number in Trace']
            })
    
    return ui_details

def cluster_sample(category_dict, ui_details, app_details, num_samples_per_cluster):
    """Perform cluster sampling using all 27 categories."""
    sampled_data = defaultdict(list)
    
    for category, apps in category_dict.items():
        sampled_apps = random.sample(apps, min(num_samples_per_cluster, len(apps)))
        for app in sampled_apps:
            rand = random.choice(ui_details.get(app, []))
            rand['UI Number'] = int(rand['UI Number'])
            sampled_data[category].append({
                'App Package Name': app,
                'App Details': app_details.get(app, {}),
                'UI Details': rand
            })
    
    return sampled_data

def extract_and_organize_dataset(tar_gz_path, output_dataset_path, sampled_data):
    """Extract images and JSON files from the tar.gz and organize them into a new dataset folder."""
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        tar.extractall(output_dataset_path)
    
    rico_images_path = os.path.join(output_dataset_path, 'combined')
    new_dataset_path = os.path.join(output_dataset_path, 'rico_sampled_dataset')
    os.makedirs(new_dataset_path, exist_ok=True)
    
    for category, apps in sampled_data.items():
        category_folder = os.path.join(new_dataset_path, category)
        os.makedirs(category_folder, exist_ok=True)
        
        for app in apps:
            # for ui in app['UI Details']:
            ui_number = app['UI Details']['UI Number']
            image_filename = f"{ui_number}.jpg"
            json_filename = f"{ui_number}.json"
            
            src_image = os.path.join(rico_images_path, image_filename)
            src_json = os.path.join(rico_images_path, json_filename)
            
            if os.path.exists(src_image) and os.path.exists(src_json):
                shutil.copy(src_image, os.path.join(category_folder, image_filename))
                shutil.copy(src_json, os.path.join(category_folder, json_filename))

def save_sampled_subset(sampled_data, output_path):
    """Save the sampled subset."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=4)

def main(metadata_csv_path, ui_csv_path, tar_gz_path, output_path, dataset_output_path, num_samples_per_cluster=100):
    category_dict, app_details = load_rico_metadata(metadata_csv_path)
    ui_details = load_ui_details(ui_csv_path)
    sampled_data = cluster_sample(category_dict, ui_details, app_details, num_samples_per_cluster)
    
    # Validate category counts
    for category, apps in sampled_data.items():
        if len(apps) != min(num_samples_per_cluster, len(category_dict[category])):
            print(f"Warning: Category {category} has {len(apps)} apps instead of {num_samples_per_cluster}")
    
    save_sampled_subset(sampled_data, output_path)
    extract_and_organize_dataset(tar_gz_path, dataset_output_path, sampled_data)
    print(f"Saved sampled dataset to {dataset_output_path}")

if __name__ == "__main__":
    metadata_csv_path = "app_details.csv"  # Update with actual path
    ui_csv_path = "ui_details.csv"  # Update with actual path
    tar_gz_path = "unique_uis.tar.gz"  # Update with actual path
    output_path = "sampled_rico_subset.json"
    dataset_output_path = "sample_rico"
    main(metadata_csv_path, ui_csv_path, tar_gz_path, output_path, dataset_output_path)
