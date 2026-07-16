import argparse
import os
import cv2
import logging

# Set up logging before importing MTCNN to suppress some TF warnings, though TF handles its own logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO and WARNING logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import MTCNN
try:
    from mtcnn import MTCNN
    # Initialize MTCNN once globally to avoid reloading model for every image
    detector = MTCNN()
except ImportError:
    detector = None
    logging.warning("MTCNN not found. Ensure tensorflow and mtcnn are installed.")

def extract_face(image_path, output_path, margin=0.2):
    """
    Extracts the largest face from an image, adds a margin, and resizes to 224x224.
    Uses MTCNN as primary detector, and Haar Cascade as secondary fallback.
    No silent resize fallback.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the extracted face.
        margin (float): Percentage of margin to add around the face.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Load image with OpenCV (BGR format)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logging.error(f"Failure - image unreadable: {image_path}")
            return False
            
        img_h, img_w = img_bgr.shape[:2]
        face_crop = None
        
        # 1. Primary Attempt: MTCNN
        if detector is not None:
            # MTCNN expects RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(img_rgb)
            
            if results:
                # Find the largest face by bounding box area
                largest_face = max(results, key=lambda rect: rect['box'][2] * rect['box'][3])
                x, y, w, h = largest_face['box']
                x, y = abs(x), abs(y)  # Ensure positive coordinates
                
                margin_x = int(w * margin)
                margin_y = int(h * margin)
                
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(img_w, x + w + margin_x)
                y2 = min(img_h, y + h + margin_y)
                
                face_crop = img_bgr[y1:y2, x1:x2]
        
        # 2. Secondary Attempt: Haar Cascades
        if face_crop is None or face_crop.size == 0:
            cascade_class = getattr(cv2, 'CascadeClassifier', None)
            if cascade_class is None and hasattr(cv2, 'objdetect'):
                cascade_class = getattr(cv2.objdetect, 'CascadeClassifier', None)
                
            if cascade_class is not None:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                cascade_path = ""
                if hasattr(cv2, 'data'):
                    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                
                if os.path.exists(cascade_path):
                    face_cascade = cascade_class(cascade_path)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                        x, y, w, h = largest_face
                        
                        margin_x = int(w * margin)
                        margin_y = int(h * margin)
                        
                        x1 = max(0, x - margin_x)
                        y1 = max(0, y - margin_y)
                        x2 = min(img_w, x + w + margin_x)
                        y2 = min(img_h, y + h + margin_y)
                        
                        face_crop = img_bgr[y1:y2, x1:x2]

        # 3. Failure
        if face_crop is None or face_crop.size == 0:
            logging.error(f"Failure - no face detected by either MTCNN or Haar Cascade: {image_path}")
            return False
            
        # Resize to 224x224
        face_resized = cv2.resize(face_crop, (224, 224))
        
        # Save the result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, face_resized)
        
        return True
        
    except Exception as e:
        logging.error(f"Failure - exception on {image_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract faces from images.")
    parser.add_argument("--input_dir", type=str, default="data/raw_samples",
                        help="Input directory containing raw images.")
    parser.add_argument("--output_dir", type=str, default="data/faces_extracted",
                        help="Output directory for extracted faces.")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return
        
    print(f"Starting face extraction from {input_dir} to {output_dir}")
    print("Using MTCNN as primary detector, Haar Cascades as secondary.")
    print("Silent resize-only fallback is disabled.")
    
    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
                
    if not image_paths:
        print("No images found to process.")
        return
        
    # Process a sample of 30 images
    sample_paths = image_paths[:30]
    
    success_count = 0
    failure_count = 0
    
    print(f"Processing a sample of {len(sample_paths)} images...")
    
    for idx, img_path in enumerate(sample_paths):
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        
        print(f"[{idx+1}/{len(sample_paths)}] Processing {rel_path}...", end=" ")
        
        success = extract_face(img_path, out_path)
        if success:
            print("SUCCESS")
            success_count += 1
        else:
            print("FAILED")
            failure_count += 1
            
    print("\n--- Summary ---")
    print(f"Total processed: {len(sample_paths)}")
    print(f"Successful extractions: {success_count}")
    print(f"Failures: {failure_count}")

if __name__ == "__main__":
    main()
