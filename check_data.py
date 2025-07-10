import os

data_dir = "dataset-resized"
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path):
        print(f"{category}: {len(os.listdir(category_path))} ảnh")