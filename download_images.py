import os
import requests
import json
from pathlib import Path

# YOUR UNSPLASH API KEY HERE
UNSPLASH_API_KEY = "iCVd7IFfRdI3BSx1VUEjpsO-w1XFz-FBJ_NPAe71-fs"  # Replace with your actual key

# Image categories to download
CATEGORIES = ["dog", "cat", "nature", "landscape", "ocean"]
IMAGES_PER_CATEGORY = 100  # Start with 500 total (5 categories x 100)

def download_images():
    """Download images from Unsplash"""
    
    # Create data folder
    data_folder = Path("data/images")
    data_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"📸 Starting image download...")
    print(f"Categories: {CATEGORIES}")
    print(f"Images per category: {IMAGES_PER_CATEGORY}")
    print(f"Total images to download: {len(CATEGORIES) * IMAGES_PER_CATEGORY}")
    print()
    
    image_count = 0
    
    for category in CATEGORIES:
        print(f"🔍 Downloading {category} images...")
        
        for page in range(1, 3):  # Get 2 pages (200 images) per category
            url = "https://api.unsplash.com/search/photos"
            params = {
                "query": category,
                "per_page": IMAGES_PER_CATEGORY // 2,
                "page": page,
                "client_id": UNSPLASH_API_KEY,
                "orientation": "landscape"
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                results = response.json()
                
                if "results" not in results:
                    print(f"   ❌ Error: {results.get('errors', 'Unknown error')}")
                    continue
                
                for idx, photo in enumerate(results["results"]):
                    image_url = photo["urls"]["regular"]
                    image_id = photo["id"]
                    
                    try:
                        # Download image
                        img_response = requests.get(image_url, timeout=10)
                        img_response.raise_for_status()
                        
                        # Save image
                        filename = f"{category}_{image_count:03d}.jpg"
                        filepath = data_folder / filename
                        
                        with open(filepath, "wb") as f:
                            f.write(img_response.content)
                        
                        image_count += 1
                        
                        if image_count % 50 == 0:
                            print(f"   ✓ Downloaded {image_count} images...")
                        
                    except Exception as e:
                        print(f"   ⚠️  Failed to download {image_id}: {str(e)}")
                        continue
                
            except requests.exceptions.RequestException as e:
                print(f"   ❌ API Error: {str(e)}")
                continue
    
    print()
    print(f"✅ Download complete!")
    print(f"📁 Total images saved: {image_count}")
    print(f"📍 Location: {data_folder.absolute()}")
    
    return image_count

if __name__ == "__main__":
    if UNSPLASH_API_KEY ==  "YOUR_ACCESS_KEY_HERE":
        print("❌ ERROR: Please replace 'YOUR_ACCESS_KEY_HERE' with your actual Unsplash API key")
        print("Get it from: https://unsplash.com/developers")
        exit(1)
    
    download_images()