import os
import torch
from pathlib import Path

# determine project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data directories
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
CAPTIONS_DIR = DATA_DIR / "captions"
MODELS_DIR = DATA_DIR / "models"

# make sure directories exist
DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
CAPTIONS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# needed file paths
CAPTIONS_FILE = CAPTIONS_DIR / "filtered_captions.tsv"
FILTERED_CAPTIONS_FILE = CAPTIONS_DIR / "filtered_captions_matched.tsv"
MODEL_PATH = MODELS_DIR / "best_model.pt"
CACHE_PATH = MODELS_DIR / "cached_features.pt"

# default device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# default training settings
DEFAULT_CONFIG = {
    'max_images': 10000,
    'batch_size': 64,
    'embed_size': 256,
    'hidden_size': 768,
    'num_layers': 6,
    'learning_rate': 0.0003,
    'num_epochs': 20,
    'early_stopping_patience': 5,
    'checkpoint_frequency': 2,
    'num_workers': 4,
    'feature_extraction_workers': 8,  
    'feature_extraction_batch_size': 32,  
    'clip_loss_weight': 0.3,
    'clip_batch_size': 16,
    'clip_eval_frequency': 50,
    'seed': 42,
    'feature_dim': 512,  # for ResNet18
}

# function to update the config with command line arguments
def update_config(args=None):
    """
    update/change the default configuration with command line arguments
    
    Args:
        args: argumentparser arguments
        
    Returns:
        updated configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if args:
       
        for key, value in vars(args).items():
            if key in config and value is not None:
                config[key] = value
    
    return config