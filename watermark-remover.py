import cv2
import numpy as np
import requests
import base64
from pathlib import Path
import json
from PIL import Image
import io
import time
from tqdm import tqdm

def create_watermark_mask(image_shape, watermark_size=(365, 55)):
    """Create a binary mask for the watermark area"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Calculate center position
    img_height, img_width = image_shape[:2]
    w_width, w_height = watermark_size
    
    # Calculate watermark position
    center_x = img_width // 2
    center_y = img_height // 2
    x1 = center_x - (w_width // 2)
    y1 = center_y - (w_height // 2)
    x2 = x1 + w_width
    y2 = y1 + w_height
    
    # Create mask (white rectangle on black background)
    mask[y1:y2, x1:x2] = 255
    
    return mask

def remove_watermark(image_path, output_dir, api_url="http://localhost:8080"):
    """Remove watermark from a single image"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Create mask
    mask = create_watermark_mask(image.shape)
    
    # Convert image and mask to base64
    _, img_encoded = cv2.imencode('.jpg', image)
    _, mask_encoded = cv2.imencode('.png', mask)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')
    
    # Prepare the request
    payload = {
        "image": img_base64,
        "mask": mask_base64,
        "prompt": "clean natural skin texture, medical image, clear skin detail",
        "negative_prompt": "text, watermark, copyright symbol, artificial, unnatural",
        "sd_steps": 25,
        "sd_sampler": "euler_a",
        "sd_cfg_scale": 7.5
    }
    
    try:
        # Make API request
        response = requests.post(
            f"{api_url}/api/v1/inpaint",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Convert response content to image
        img_array = np.frombuffer(response.content, np.uint8)
        cleaned_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Create output path (maintain original filename without 'cleaned_' prefix)
        output_path = Path(output_dir) / Path(image_path).name
        
        # Save the cleaned image
        cv2.imwrite(str(output_path), cleaned_image)
        
        return True, str(output_path)
        
    except requests.exceptions.RequestException as e:
        return False, f"API request failed: {str(e)}"

def process_dataset(dataset_root, output_root, api_url="http://localhost:8080"):
    """Process entire dataset maintaining class folder structure"""
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    
    # Counter for processed images
    total_successful = 0
    total_failed = 0
    
    # Get all class folders
    class_folders = [f for f in dataset_root.iterdir() if f.is_dir()]
    print(f"Found {len(class_folders)} class folders")
    
    # Process each class folder
    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"\nProcessing class: {class_name}")
        
        # Create output directory for this class
        output_class_dir = output_root / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images in this class folder
        images = list(class_folder.glob("*.[jJ][pP][gG]")) + \
                list(class_folder.glob("*.[jJ][pP][eE][gG]")) + \
                list(class_folder.glob("*.[pP][nN][gG]"))
        
        # Process images with progress bar
        with tqdm(total=len(images), desc=f"Processing {class_name}") as pbar:
            for image_path in images:
                try:
                    success, message = remove_watermark(image_path, output_class_dir, api_url)
                    
                    if success:
                        total_successful += 1
                    else:
                        total_failed += 1
                        print(f"\nFailed - {image_path.name}: {message}")
                    
                    # Add small delay to prevent overwhelming the API
                    time.sleep(0.1)
                    
                except Exception as e:
                    total_failed += 1
                    print(f"\nError processing {image_path.name}: {str(e)}")
                
                pbar.update(1)
    
    return total_successful, total_failed

def main():
    # Update these paths
    dataset_root = r"D:\jameel\Original Dataset\Dermnet"  # Root folder containing class folders
    output_root = r"D:\jameel\Original Dataset\Cleaned Dermnet"  # Where to save processed images
    api_url = "http://localhost:8080"
    
    try:
        print("Starting dataset processing...")
        start_time = time.time()
        
        successful, failed = process_dataset(dataset_root, output_root, api_url)
        
        # Calculate processing time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print("\nProcessing Complete!")
        print(f"Total processing time: {hours}h {minutes}m {seconds}s")
        print(f"Successfully processed: {successful} images")
        print(f"Failed: {failed} images")
        print(f"Cleaned dataset saved in: {output_root}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
