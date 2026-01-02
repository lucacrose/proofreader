import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# This loads the variables from the .env file into the system environment
load_dotenv()

# Now you can access them like normal environment variables
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# Directory to save downloaded images
SAVE_DIR = Path("generator/assets/backgrounds")
SAVE_DIR.mkdir(exist_ok=True)

# Number of random images to download
NUM_IMAGES = 50

# Unsplash API endpoint for random photos
UNSPLASH_RANDOM_URL = "https://api.unsplash.com/photos/random"

headers = {
    "Accept-Version": "v1",
    "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
}

for i in range(NUM_IMAGES):
    try:
        # Request a random image
        params = {
            "orientation": "landscape",
            "count": 1
        }
        response = requests.get(UNSPLASH_RANDOM_URL, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        if not data or not isinstance(data, list):
            print(f"⚠️ Unexpected response at iteration {i}")
            continue

        photo = data[0]
        img_url = photo["urls"]["full"]

        # Download the image
        img_resp = requests.get(img_url, stream=True)
        img_resp.raise_for_status()

        # Create a filename
        filename = SAVE_DIR / f"unsplash_{i + 99:03d}.jpg"

        with open(filename, "wb") as f:
            for chunk in img_resp.iter_content(1024):
                f.write(chunk)

        print(f"✅ Downloaded {filename}")

    except Exception as e:
        print(f"❌ Failed at {i}: {e}")
