import json
import os
from pathlib import Path

def convert_gmc_to_yolo(json_file_path, output_labels_dir, class_id=0):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    num = 0

    img_width, img_height = data['size']
    
    output_filename = Path(Path(json_file_path).stem).stem + '.txt'
    output_path = os.path.join(output_labels_dir, output_filename)

    with open(output_path, 'w') as out_file:
        if not data['objects']:
            print(f"No objects in {json_file_path}, creating empty file.")
            return 0
        
        for obj in data['objects']:
            if obj['type'] == 'rect':
                x_min, y_min, w, h = obj['data']
                x_max = x_min + w
                y_max = y_min + h
                
                center_x = (x_min + x_max) / 2.0
                center_y = (y_min + y_max) / 2.0
                width = x_max - x_min
                height = y_max - y_min
                
                center_x_norm = center_x / img_width
                center_y_norm = center_y / img_height
                width_norm = width / img_width
                height_norm = height / img_height
                
                yolo_line = f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"
                out_file.write(yolo_line)
                
                print(f"Converted bbox: {[x_min, y_min, x_max, y_max]} -> {yolo_line.strip()}")
        num += 1
    return num


def convert_json_to_yolo(input_json_dir, output_labels_dir, class_id=0):
    os.makedirs(output_labels_dir, exist_ok=True)
    json_files = list(Path(input_json_dir).glob('*.json'))
    num = 0
    
    print(f"Found {len(json_files)} JSON files to convert.")
    
    for json_file in json_files:
        print(f"Converting {json_file.name}...")
        num += convert_gmc_to_yolo(str(json_file), output_labels_dir, class_id)
    
    print("Conversion completed!")
    print("Pict with objects: ", num)


if __name__ == "__main__":
    INPUT_JSON_DIR = "./data_for_sirius_2025/labels_gmc"
    OUTPUT_LABELS_DIR = "./data_for_sirius_2025/labels_yolo"  

    convert_json_to_yolo(INPUT_JSON_DIR, OUTPUT_LABELS_DIR, class_id=0)