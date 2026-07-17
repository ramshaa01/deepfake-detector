import os
import glob
import pandas as pd
import random
import shutil
from tqdm import tqdm
import sys
import logging

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
    input_dir = os.path.join("data", "raw_samples")
    output_dir = os.path.join("data", "faces_extracted")
    metadata_path = os.path.join(output_dir, "metadata.csv")

    print(f"Scanning for all images in {input_dir} to build population lists...")
    
    real_paths = []
    fake_paths = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.mp4'}
    
    for root, dirs, files in os.walk(input_dir):
        # Skip the output directory if it's nested
        if "faces_extracted" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                path = os.path.join(root, file)
                label = get_label_from_path(path)
                if label == 'real':
                    real_paths.append(path)
                elif label == 'fake':
                    fake_paths.append(path)

    print("\n--- True Population Sizes ---")
    print(f"Total REAL images found: {len(real_paths)}")
    print(f"Total FAKE images found: {len(fake_paths)}")
    
    if len(real_paths) < 1000 or len(fake_paths) < 1000:
        print("Error: Not enough images to sample 1000 from each class.")
        return

    # Randomly shuffle and take 1000 from each
    random.seed(42)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    sampled_real = real_paths[:1000]
    sampled_fake = fake_paths[:1000]
    balanced_paths = sampled_real + sampled_fake
    # Shuffle the final batch so real/fake are mixed during extraction
    random.shuffle(balanced_paths)
    
    print("\nClearing old extraction output...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nStarting balanced batch extraction on {len(balanced_paths)} images...")
    
    metadata = []
    success_count = 0
    failure_count = 0
    label_counts = {'real': {'total': 0, 'success': 0, 'fail': 0}, 
                    'fake': {'total': 0, 'success': 0, 'fail': 0}}
    
    for img_path in tqdm(balanced_paths, desc="Extracting balanced batch"):
        label = get_label_from_path(img_path)
        label_counts[label]['total'] += 1
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, label, filename)
        
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
        
    print("\n--- Balanced Batch Extraction Summary ---")
    print(f"Total processed: {len(balanced_paths)}")
    print(f"Total successful: {success_count} ({success_count/len(balanced_paths)*100:.2f}%)")
    print(f"Total failed: {failure_count} ({failure_count/len(balanced_paths)*100:.2f}%)")
    
    print("\nClass Balance Check (Successful Extractions):")
    total_success = max(success_count, 1) # Prevent div by 0
    for lbl in ['real', 'fake']:
        c_succ = label_counts[lbl]['success']
        c_fail = label_counts[lbl]['fail']
        pct = (c_succ / total_success) * 100
        print(f"  {lbl.upper()}: {c_succ} successful ({pct:.2f}%) | {c_fail} failed")
        
    df = pd.DataFrame(metadata)
    df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
