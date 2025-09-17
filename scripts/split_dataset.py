import os
import random
import shutil
from pathlib import Path

SEED = 42
RATIOS = (0.7, 0.15, 0.15)      # train, val, test
IMAGES_DIR = "./data_for_sirius_2025/data_sirius/"  # Папка с исходными изображениями
LABELS_DIR = "./data_for_sirius_2025/labels_yolo/"  # Папка с YOLO labels
OUTPUT_DIR = "./data_for_sirius_2025/prepared"      # Куда сохранить результат

def main():
    random.seed(SEED)

    for split in ['train', 'val', 'test']:
        (Path(OUTPUT_DIR) / 'images' / split).mkdir(parents=True, exist_ok=True)
        (Path(OUTPUT_DIR) / 'labels' / split).mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        image_files.extend(Path(IMAGES_DIR).glob(f'*{ext}'))
    
    valid_pairs = []

    for img_path in image_files:
        label_path = Path(LABELS_DIR) / f"{img_path.stem}.txt"

        if label_path.exists():
            valid_pairs.append((img_path, label_path))
    
    print(f"Found {len(valid_pairs)} images with corresponding labels")
    
    with_objects = []
    without_objects = []
    
    for img_path, label_path in valid_pairs:
        with open(label_path, 'r') as f:
            content = f.read().strip()
        
        if content:
            with_objects.append((img_path, label_path))
        else: 
            without_objects.append((img_path, label_path))
    
    print(f"With objects: {len(with_objects)}")
    print(f"Without objects: {len(without_objects)}")
    
    random.shuffle(with_objects)
    random.shuffle(without_objects)
    
    def get_split_sizes(total):
        train = int(total * RATIOS[0])
        val = int(total * RATIOS[1])
        test = total - train - val
        return train, val, test
    
    train_obj, val_obj, test_obj = get_split_sizes(len(with_objects))
    train_no_obj, val_no_obj, test_no_obj = get_split_sizes(len(without_objects))
    
    splits = {
        'train': with_objects[:train_obj] + without_objects[:train_no_obj],
        'val': with_objects[train_obj:train_obj+val_obj] + without_objects[train_no_obj:train_no_obj+val_no_obj],
        'test': with_objects[train_obj+val_obj:] + without_objects[train_no_obj+val_no_obj:]
    }
    
    for split_name, pairs in splits.items():
        for img_path, label_path in pairs:
            shutil.copy2(img_path, Path(OUTPUT_DIR) / 'images' / split_name / img_path.name)
            shutil.copy2(label_path, Path(OUTPUT_DIR) / 'labels' / split_name / label_path.name)
    
    print("\nFinal distribution:")
    for split_name in ['train', 'val', 'test']:
        split_images = list((Path(OUTPUT_DIR) / 'images' / split_name).glob('*'))
        with_obj_count = 0
        without_obj_count = 0
        
        for img_path in split_images:
            label_path = Path(OUTPUT_DIR) / 'labels' / split_name / f"{img_path.stem}.txt"
            with open(label_path, 'r') as f:
                content = f.read().strip()
            
            if content:
                with_obj_count += 1
            else:
                without_obj_count += 1
        
        total = with_obj_count + without_obj_count
        print(f"{split_name}: {total} images")
        print(f"  With objects: {with_obj_count} ({with_obj_count/total:.1%})")
        print(f"  Without objects: {without_obj_count} ({without_obj_count/total:.1%})")


if __name__ == "__main__":
    main()