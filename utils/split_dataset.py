import os
import shutil
import random

original_dataset_dir = r'D:\python project\corn_leaf_disease_detection v3\data\full_dataset'
base_output_dir = r'D:\python project\corn_leaf_disease_detection v3\data'

# Create train and test folders
train_dir = os.path.join(base_output_dir, 'train')
test_dir = os.path.join(base_output_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Only loop over directories (skip files like .csv)
for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)

    if not os.path.isdir(class_path):
        continue  # Skip files

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(0.7 * len(images))
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create class subfolders in train/test
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copyfile(src, dst)

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        shutil.copyfile(src, dst)

    print(f"Class: {class_name} - Total: {len(images)}, Train: {len(train_images)}, Test: {len(test_images)}")

print("Dataset split complete.")
