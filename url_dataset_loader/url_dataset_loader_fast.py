import os
import csv
import requests
import threading
from PIL import Image
from io import BytesIO
import time
import uuid
import json
from tqdm import tqdm


class URLDatasetLoader:
    """
    class for loading and processing datasets with captions and image URLs
    also handles downloading images from URLs using threading for speed
    """

    def __init__(self, dataset_path=None, images_dir=None, num_threads=10):
        """
        initialize the URL dataset loader.

        Args:
            dataset_path: path to TSV file with captions and image URLs
            images_dir: directory to save downloaded images
            num_threads: nr of threads for parallel downloading
        """
        self.dataset_path = dataset_path
        self.images_dir = images_dir
        self.num_threads = num_threads
        self.captions = {}
        self.image_paths = {}

    def load_dataset(self, dataset_path=None, delimiter='\t'):
        """Load the dataset from a TSV file."""
        if dataset_path:
            self.dataset_path = dataset_path
        if not self.dataset_path:
            raise ValueError("Dataset path not specified")

        caption_dict = {}
        url_dict = {}

        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for idx, row in enumerate(reader):
                if len(row) >= 2:
                    image_id = f"{idx}"
                    caption = row[0].strip()
                    url = row[1].strip()
                    caption_dict[image_id] = [caption]
                    url_dict[image_id] = url

        return caption_dict, url_dict

    def download_image(self, image_id, url, local_paths, lock):
        """
        download a single image
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content))
            ext = ".jpg"
            filename = f"{image_id}{ext}"
            filepath = os.path.join(self.images_dir, filename)

            img.convert("RGB").save(filepath)

            with lock:
                local_paths[image_id] = filepath

        except Exception as e:
            print(f"Failed to download {image_id}: {e}")

    def download_images(self, url_dict):
        """Download images using multiple threads for speed."""
        if not self.images_dir:
            raise ValueError("Images directory not specified")
        os.makedirs(self.images_dir, exist_ok=True)

        local_paths = {}
        lock = threading.Lock()
        threads = []

        for image_id, url in tqdm(url_dict.items(), desc="Downloading Images"):
            thread = threading.Thread(target=self.download_image, args=(image_id, url, local_paths, lock))
            thread.start()
            threads.append(thread)

            if len(threads) >= self.num_threads:
                for t in threads:
                    t.join()
                threads = []

        for t in threads:
            t.join()

        return local_paths

    def process_dataset(self, dataset_path=None, images_dir=None, download=True, save_metadata=True):
        """Load dataset, download images, and save metadata."""
        if dataset_path:
            self.dataset_path = dataset_path
        if images_dir:
            self.images_dir = images_dir

        self.captions, url_dict = self.load_dataset(self.dataset_path)

        if download:
            self.image_paths = self.download_images(url_dict)

        if save_metadata:
            os.makedirs(os.path.dirname(self.images_dir), exist_ok=True)
            with open(os.path.join(self.images_dir, 'captions.json'), 'w', encoding='utf-8') as f:
                json.dump(self.captions, f, indent=4)
            with open(os.path.join(self.images_dir, 'image_paths.json'), 'w', encoding='utf-8') as f:
                json.dump(self.image_paths, f, indent=4)

        return self.image_paths, self.captions


# **Main block to run the script directly**
if __name__ == "__main__":
    dataset_path = "Train_GCC-training.tsv"  
    images_dir = "loadedimages"

    loader = URLDatasetLoader(dataset_path, images_dir, num_threads=10)
    image_paths, captions = loader.process_dataset(download=True, save_metadata=True)

    print(f"\n✅ Downloaded {len(image_paths)} images and processed {len(captions)} captions.")
