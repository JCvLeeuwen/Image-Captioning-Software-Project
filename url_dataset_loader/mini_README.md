# Image dataset downloader

## Features

- **Multi-threaded downloading**: process multiple images simultaneously 
- **Error handling**: handles download failures
- **Image conversion**: automatically converts downloaded images to RGB JPG format
- **Caption alignment**: captions are only kept for successfully downloaded images
- **Metadata storage**: saves caption and file path data in JSON format

## Requirements

- Python 3.6+
- Required packages: 
  - requests
  - Pillow
  - tqdm
  - csv
  - threading
  - json

## Installation

Install required packages:
```bash
pip install requests Pillow tqdm
```

## Usage

### Step 1: Download Images

Run the URL dataset loader:

```bash
python url_dataset_loader_fast.py
```

### Step 2: Filter Captions

After downloading is complete, align captions with the downloaded images:

```bash
python filter_captions_tsv_file.py
```

## Output

After running both scripts, you'll have:

1. A directory of downloaded images named by their index (0.jpg, 1.jpg, etc.)
2. Two JSON files in the images directory:
   - `captions.json`: Maps image IDs to their captions
   - `image_paths.json`: Maps image IDs to their file paths
3. A filtered TSV file (`filtered_captions.tsv`) containing only the captions of successfully downloaded images

## How It Works

### url_dataset_loader_fast.py

This script:
1. Reads the TSV file containing captions and image URLs
2. Creates multiple threads to download images in parallel
3. Saves images to the specified directory
4. Creates JSON metadata files for captions and image paths

### filter_captions_tsv_file.py

This script:
1. Checks which images were successfully downloaded
2. Creates a new TSV file with captions only for downloaded images
3. Uses the image index as the identifier to match captions with images


