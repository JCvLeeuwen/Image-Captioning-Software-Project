import os
import torch
import random
import pandas as pd
import concurrent.futures
import threading
import time
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Set
from pathlib import Path

from src.config import DEVICE


def filter_captions_file(captions_file: str, image_folder: str, 
                        output_file: Optional[str] = None) -> pd.DataFrame:
    """
    make sure the captions file only has entries for images that actually exist

    Args:
        captions_file: path to the original captions TSV file
        image_folder: folder containing the images
        output_file: path to save the filtered captions file (optional)

    Returns:
        dataframe with filtered captions
    """
    import os
    import pandas as pd
    from tqdm import tqdm

    print(f"reading captions from:: {captions_file}")
    # read captions file
    df = pd.read_csv(captions_file, sep='\t', header=None, names=['image_id', 'caption'])
    original_count = len(df)
    print(f"original captions count: {original_count}")

    # list of available images
    print(f"Checking available images in: {image_folder}")
    available_images = set()
    for ext in ['.jpg', '.jpeg', '.png']:
        for file in os.listdir(image_folder):
            if file.lower().endswith(ext):
                # store both with and without extension
                base_name = os.path.splitext(file)[0]
                available_images.add(base_name)
                available_images.add(file)

    print(f"found {len(available_images)} available images")

    # filter captions to only include those for available images
    print("filtering captions...")
    filtered_df = df[df['image_id'].astype(str).isin(available_images)]
    filtered_count = len(filtered_df)
    print(f"nr of filtered captions: {filtered_count}")
    print(f"removed {original_count - filtered_count} captions for missing images")

    # save to output file
    if output_file:
        print(f"saving filtered captions to: {output_file}...")
        filtered_df.to_csv(output_file, sep='\t', header=False, index=False)
        print(f"saved filtered captions file with {filtered_count} entries!")

    return filtered_df


def extract_and_cache_features_threadpool(image_folder: str, 
                                         captions_dict: Dict[str, List[str]], 
                                         cache_path: Optional[str] = None, 
                                         force_reload: bool = False, 
                                         num_workers: int = 4, 
                                         batch_size: int = 64,
                                         model_name: str = 'resnet18') -> Dict[str, torch.Tensor]:
    """
    extract image features using resnet with threadpoolexecutor and cache them to disk

    Args:
        image_folder: path to folder with images
        captions_dict: dictionary mapping image_id to list of captions
        cache_path: path to cache file (optional)
        force_reload: whether to force reloading features
        num_workers: nr of threads to use
        batch_size: batch size for feature extraction
        model_name: name of model to use (default: 'resnet18')

    Returns:
        dictionary mapping image_id to image features
    """
    import os
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    from tqdm import tqdm
    import time
    import concurrent.futures
    import threading

    # default cache path if not provided
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(image_folder), "cached_features.pt")

    # check if cache exists and if it does, load it
    all_features_dict = {}
    if not force_reload and os.path.exists(cache_path):
        print(f"loading cached features from {cache_path}...")
        try:
            all_features_dict = torch.load(cache_path)
            print(f"loaded features for {len(all_features_dict)} images from cache")
        except Exception as e:
            print(f"error loading cache: {e}")
            all_features_dict = {}

    # create a new features_dict that only contains images from captions_dict
    print("aligning feature and caption dictionaries...")
    features_dict = {}
    missing_images = []

    # first, add features from cache for images in captions_dict
    for img_id in captions_dict.keys():
        if img_id in all_features_dict:
            features_dict[img_id] = all_features_dict[img_id]
        else:
            missing_images.append(img_id)

    print(f"using {len(features_dict)} cached features")
    print(f"need to extract features for {len(missing_images)} more images")

    # if we already have all features we need, return them
    if not missing_images:
        print("all required features found in cache! :)")
        return features_dict

    # load the appropriate model
    print(f"loading {model_name} model...")
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"unsupported model: {model_name}")
    
    # remove final classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(DEVICE)

    # image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    def load_image(img_name):
        """
        load and preprocess a single image
        """
        # try different extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(image_folder, f"{img_name}{ext}")
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                    img_tensor = transform(image)
                    return img_name, img_tensor
                except Exception as e:
                    print(f"error loading {img_path}: {e}")
                    break
                
        # try with the name itself (,if it already includes extension)
        img_path = os.path.join(image_folder, img_name)
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
                img_tensor = transform(image)
                return img_name, img_tensor
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
        return None, None

    print(f"extracting features for {len(missing_images)} images using {num_workers} threads and batch size {batch_size}...")
    start_time = time.time()

    # result dictionary and lock for thread safety
    result_lock = threading.Lock()

    # process images in batches
    extracted_count = 0
    with tqdm(total=len(missing_images)) as pbar:
        # process all images in batches
        for i in range(0, len(missing_images), batch_size):
            batch_names = missing_images[i:i + batch_size]

            # load images parallel
            loaded_images = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                for result in executor.map(load_image, batch_names):
                    if result[0] is not None:  # Skip failed images
                        loaded_images.append(result)

            # skip empty batches
            if not loaded_images:
                pbar.update(len(batch_names))
                continue

            # separate names and tensors
            batch_img_names = [img[0] for img in loaded_images]
            batch_tensors = [img[1] for img in loaded_images]

            # process batch on GPU
            try:
                #  move to GPU
                stacked_tensors = torch.stack(batch_tensors).to(DEVICE)

                # extract features in a single forward pass
                with torch.no_grad():
                    features = model(stacked_tensors).squeeze(-1).squeeze(-1).cpu()

                # features to dictionary
                with result_lock:
                    for j, img_name in enumerate(batch_img_names):
                        features_dict[img_name] = features[j]
                        all_features_dict[img_name] = features[j]  
                        extracted_count += 1

                    # save checkpoint every 1000 images
                    if extracted_count % 1000 == 0:
                        print(f"\nsaving checkpoint after {extracted_count} images...")
                        torch.save(all_features_dict, cache_path + ".temp")
            except Exception as e:
                print(f"error processing batch: {e}")

            pbar.update(len(batch_names))

    # save final features dictionary
    print(f"extracted features for {extracted_count} images in {time.time() - start_time:.2f} seconds")
    print(f"total features in cache: {len(all_features_dict)}")
    print(f"features for this run:: {len(features_dict)}")
    print(f"saving features to {cache_path}...")
    torch.save(all_features_dict, cache_path)

    # remove temporary file if it exists
    if os.path.exists(cache_path + ".temp"):
        os.remove(cache_path + ".temp")

    return features_dict


def prepare_data_with_cache(captions_file: str, image_folder: str, 
                           max_images: int = 10000, 
                           cache_path: Optional[str] = None, 
                           force_reload: bool = False, 
                           num_workers: int = 8, 
                           batch_size: int = 64) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]], Dict[str, int], Dict[int, str]]:
    """
    prepare data for training, with feature caching support

    Args:
        captions_file: path to captions TSV file
        image_folder: path to folder containing images
        max_images: max nr of images to use
        cache_path: path to cache file (optional)
        force_reload: whether to force reloading features
        num_workers: nr of threads to use
        batch_size: batch size for feature extraction

    Returns:
        tuple of (features_dict, captions_dict, word2idx, idx2word)
    """
    print("loading captions...")
    # load and clean captions
    df = pd.read_csv(captions_file, sep="\t", names=["image", "caption"])
    df = df.dropna()
    df["image"] = df["image"].astype(str).str.strip()

    # rremove .jpg extension 
    df["image"] = df["image"].apply(lambda x: x[:-4] if x.endswith('.jpg') else x)

    # build dictionary: image -> [captions]
    captions_dict = {}
    for _, row in df.iterrows():
        img_name = row["image"]
        caption = row["caption"]
        if img_name not in captions_dict:

            captions_dict[img_name] = []
        captions_dict[img_name].append(caption)

    # limit to max_images if needed
    if len(captions_dict) > max_images:
        print(f"limiting dataset from {len(captions_dict)} to {max_images} images")
        random.seed(42)
        subset_ids = random.sample(list(captions_dict.keys()), max_images)
        subset_captions_dict = {img_id: captions_dict[img_id] for img_id in subset_ids}
        captions_dict = subset_captions_dict

    print(f"using {len(captions_dict)} images with captions")

    # make vocabulary
    all_captions = sum(captions_dict.values(), [])
    words = [word for caption in all_captions for word in caption.split()]
    most_common = Counter(words).most_common(4900)

    # special tokens first
    vocab = ["<PAD>", "< SOS >", "<EOS>", "<UNK>"] + [w for w, _ in most_common]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(vocab)
    print(f"vocab size: {vocab_size}")

    # image features with caching
    print("extracting image features (with multithreaded caching)...")
    features_dict = extract_and_cache_features_threadpool(
        image_folder,
        captions_dict,
        cache_path=cache_path,
        force_reload=force_reload,
        num_workers=num_workers,
        batch_size=batch_size
    )

    caption_keys = set(captions_dict.keys())
    feature_keys = set(features_dict.keys())

    print(f"caption keys: {len(caption_keys)}, feature keys: {len(feature_keys)}")

    # any discrepancies?
    missing_captions = feature_keys - caption_keys
    missing_features = caption_keys - feature_keys

    if missing_captions:
        print(f"WARNING: {len(missing_captions)} images have features but no captions")
        # remove them from features_dict
        for img_id in missing_captions:
            if img_id in features_dict:
                del features_dict[img_id]

    if missing_features:
        print(f"WARNING: {len(missing_features)} images have captions but no features")
        # remove them from captions_dict
        for img_id in missing_features:
            if img_id in captions_dict:
                del captions_dict[img_id]

    # lastcheck
    print(f"final dataset: {len(features_dict)} images with both features and captions")

    return features_dict, captions_dict, word2idx, idx2word


def filter_to_all_training_images(reference_captions: Dict[str, List[str]], 
                                 generated_captions: Dict[str, str], 
                                 image_paths: Dict[str, str],
                                 train_ids: Set[str], 
                                 val_ids: Set[str]) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str]]:
    """
    filter evaluation images to match exactly those used in both training and validation
    
    Args:
        reference_captions: dictionary of reference captions
        generated_captions: dictionary of generated captions
        image_paths: dictionary of image paths
        train_ids: set of training image IDs
        val_ids: set of validation image IDs
    
    Returns:
        tuple of filtered (reference_captions, generated_captions, image_paths)
    """
    # extract image IDs from both datasets
    all_used_image_ids = train_ids.union(val_ids)

    print(f"found {len(train_ids)} training and {len(val_ids)} validation images")
    print(f"total of {len(all_used_image_ids)} unique images used in training/validation")

    # filter dictionaries to only include these IDs
    filtered_reference = {img_id: captions for img_id, captions in reference_captions.items()
                         if img_id in all_used_image_ids}
    filtered_generated = {img_id: caption for img_id, caption in generated_captions.items()
                         if img_id in all_used_image_ids}
    filtered_paths = {img_id: path for img_id, path in image_paths.items()
                     if img_id in all_used_image_ids}

    print(f"filtered to {len(filtered_reference)} reference images")
    print(f"filtered to {len(filtered_generated)} generated images")
    print(f"filtered to {len(filtered_paths)} image paths")

    return filtered_reference, filtered_generated, filtered_paths