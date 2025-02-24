import os
import shutil
from sklearn.model_selection import train_test_split

# 📂 Define Paths
dataset_path = "/home/group2/Phanh/traffic-sign-yolo8/archive"
output_path = "/home/group2/Phanh/traffic-sign-yolo8/archive/split"  # Update this path

image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")

# ✅ Get all image files (assuming label files have the same names)
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Total images found: {len(image_files)}")

# 📌 Ensure labels exist for every image
image_files = [f for f in image_files if os.path.exists(os.path.join(label_folder, f.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')))]
print(f"Total valid image-label pairs: {len(image_files)}")

# 📊 Split Ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 🔄 Split dataset (keeping image-label pairs together)
train_files, temp_files = train_test_split(image_files, test_size=(val_ratio + test_ratio), random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

# 📂 Function to Copy Files
def copy_files(file_list, src_image_folder, src_label_folder, dest_image_folder, dest_label_folder):
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_label_folder, exist_ok=True)

    for file in file_list:
        shutil.copy(os.path.join(src_image_folder, file), os.path.join(dest_image_folder, file))
        label_file = file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        shutil.copy(os.path.join(src_label_folder, label_file), os.path.join(dest_label_folder, label_file))

# 📂 Create train, val, and test folders
copy_files(train_files, image_folder, label_folder, os.path.join(output_path, "train/images"), os.path.join(output_path, "train/labels"))
copy_files(val_files, image_folder, label_folder, os.path.join(output_path, "val/images"), os.path.join(output_path, "val/labels"))
copy_files(test_files, image_folder, label_folder, os.path.join(output_path, "test/images"), os.path.join(output_path, "test/labels"))

print("✅ Data successfully split into train, val, and test sets!")
