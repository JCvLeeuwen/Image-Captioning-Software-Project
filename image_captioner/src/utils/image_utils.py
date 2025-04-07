import os
import zipfile
import random
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from typing import List, Dict, Set, Optional, Tuple


def extract_images_from_zip(zip_path: str, output_dir: str, 
                           max_images: int = 10000) -> int:
    """
    extract images from a ZIP file to an output directory
    
    Args:
        zip_path: path to the ZIP file
        output_dir: directory to extract images to
        max_images: max nr of images to extract
        
    Returns:
        nr of images extracted
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found at {zip_path}")
    
    # make output directory if it doesnt exist
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # list of image files in the ZIP
        all_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # limit to max_images
        if len(all_files) > max_images:
            print(f"ZIP contains {len(all_files)} images, limiting extraction to {max_images}")
            # shuffle the list to get a random sample
            random.seed(42)  # reproducibility
            random.shuffle(all_files)
            file_list = all_files[:max_images]
        else:
            file_list = all_files
        
        # extract the selected images
        for file in tqdm(file_list, desc="Extracting images"):
            zip_ref.extract(file, output_dir)
        
        print(f"Extracted {len(file_list)} images to {output_dir}")
        return len(file_list)


def check_image_corruption(image_path: str) -> Tuple[bool, str]:
    """
    check if an image file is corrupted
    
    Args:
        image_path: path to the image file
        
    Returns:
        tuple of (is_corrupt, reason)
    """
    # check if file exists
    if not os.path.exists(image_path):
        return True, "file not found"
    
    # check file size
    file_size = os.path.getsize(image_path)
    if file_size < 100:
        return True, f"file too small: {file_size} bytes"
    
    # try to open the image
    try:
        with Image.open(image_path) as img:
            img.verify()  
            
            # check image dimensions
            if img.width < 10 or img.height < 10:
                return True, f"image too small: {img.width}x{img.height}"
            
            return False, "OK"
    except Exception as e:
        return True, f"image verification failed: {str(e)}"


def create_feature_extractor(model_name: str = "resnet18", device: str = "cuda") -> torch.nn.Module:
    """
    make a model for feature extraction
    
    Args:
        model_name: name of the model ('resnet18' or 'resnet50')
        device: device
        
    Returns:
        feature extraction model
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"unsupported model: {model_name}")
    
    # remove classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model.eval().to(device)


def get_image_transform() -> transforms.Compose:
    """
    get transformation pipeline for images
    
    Returns:
        torchvision transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def extract_features_from_image(image_path: str, model: torch.nn.Module, 
                               transform: transforms.Compose,
                               device: str = "cuda") -> Optional[torch.Tensor]:
    """
    extract features from a single image
    
    Args:
        image_path: path to the image
        model: feature extraction model
        transform: image transformation pipeline
        device: device
        
    Returns:
        extracted features as a tensor, or None if extraction failed
    """
    try:
        # open and transform image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # extract features
        with torch.no_grad():
            features = model(image_tensor).squeeze(-1).squeeze(-1).cpu()
        
        return features
    except Exception as e:
        print(f"error extracting features from {image_path}: {e}")
        return None


def preprocess_images_batch(image_dir: str, model: torch.nn.Module, 
                           transform: transforms.Compose, 
                           device: str = "cuda", 
                           batch_size: int = 16) -> Dict[str, torch.Tensor]:
    """
    extract features from all images in a directory in batches
    

    Args:
        image_dir: directory containing images
        model: feature extraction model
        transform: image transformation pipeline
        device: device
        batch_size: batch size for processing
        
    Returns:
        dictionary mapping image names to features
    """
    # list of image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(os.listdir(image_dir)))
        image_files = [f for f in image_files if f.lower().endswith(ext)]
    
    features_dict = {}
    
    # process in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_tensors = []
        batch_names = []
        
        # load images
        for file_name in batch_files:
            try:
                file_path = os.path.join(image_dir, file_name)
                image = Image.open(file_path).convert("RGB")
                tensor = transform(image)
                batch_tensors.append(tensor)
                batch_names.append(os.path.splitext(file_name)[0])  # Remove extension
            except Exception as e:
                print(f"error loading {file_name}: {e}")
        
        if not batch_tensors:
            continue
        
        # process batch
        try:
            stacked_tensors = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                features = model(stacked_tensors).squeeze(-1).squeeze(-1).cpu()
            
            # save features
            for j, name in enumerate(batch_names):
                features_dict[name] = features[j]
                
        except Exception as e:
            print(f"error processing batch: {e}")
    
    return features_dict