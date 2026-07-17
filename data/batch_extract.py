import os
import pandas as pd
from tqdm import tqdm
import sys
import logging
import argparse

# Ensure data package is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.face_extraction import extract_face

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_label_from_path(filepath):
    filepath = filepath.lower()
    if 'fake' in filepath.split(os.sep):
        return 'fake'
    elif 'real' in filepath.split(os.sep):
        return 'real'
    if 'fake' in filepath:
        return 'fake'
    if 'real' in filepath:
        return 'real'
    return 'unknown'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only a limited number of images")
    args = parser.parse_args()

    input_dir = os.path.join("data", "raw_samples")
    output_dir = os.path.join("data", "faces_extracted")
    metadata_path = os.path.join(output_dir, "metadata.csv")

    print(f"Scanning for images in {input_dir} (this may take a moment)...")
    
    image_paths = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.mp4'}
    
    for root, dirs, files in os.walk(input_dir):
        # Skip the output directory if it's nested
        if "faces_extracted" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                image_paths.append(os.path.join(root, file))
                if args.limit and len(image_paths) >= args.limit:
                    break
        if args.limit and len(image_paths) >= args.limit:
            break
            
    if not image_paths:
        print("No images found in", input_dir)
        return

    print(f"Found {len(image_paths)} files. Starting batch extraction...")
    
    metadata = []
    success_count = 0
    failure_count = 0
    label_counts = {'real': {'total': 0, 'success': 0, 'fail': 0}, 
                    'fake': {'total': 0, 'success': 0, 'fail': 0},
                    'unknown': {'total': 0, 'success': 0, 'fail': 0}}
    
    for img_path in tqdm(image_paths, desc="Extracting faces"):
        label = get_label_from_path(img_path)
        label_counts[label]['total'] += 1
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, label, filename)
        
        if os.path.exists(out_path):
            success = True
        else:
            # Create directories as needed
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            success = extract_face(img_path, out_path)
            
        status = "success" if success else "failed"
        
        if success:
            success_count += 1
            label_counts[label]['success'] += 1
        else:
            failure_count += 1
            label_counts[label]['fail'] += 1
            
        metadata.append({
            'filename': filename,
            'label': label,
            'manipulation_type': 'N/A', 
            'source_path': img_path,
            'extraction_status': status
        })
        
    print("\n--- Batch Extraction Summary ---")
    print(f"Total processed: {len(image_paths)}")
    print(f"Total successful: {success_count} ({success_count/len(image_paths)*100:.2f}%)")
    print(f"Total failed: {failure_count} ({failure_count/len(image_paths)*100:.2f}%)")
    print("\nLabel Breakdown:")
    for lbl, counts in label_counts.items():
        if counts['total'] > 0:
            print(f"  {lbl.upper()}: {counts['total']} total | {counts['success']} success | {counts['fail']} fail")
            
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(metadata)
    df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
