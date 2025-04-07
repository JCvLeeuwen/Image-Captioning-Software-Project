import torch
import random
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any


class CaptionFeatureDataset(Dataset):
    """
    dataset for training/validation with image features and captions
    
    Attributes:
        image_names: list of image IDs in the dataset
        features_dict: dictionary mapping image_id to image features
        captions_dict: dictionary mapping image_id to list of captions
        word2idx: wrd to index mapping
        max_len: max caption length
    """
    
    def __init__(self, features_dict: Dict[str, torch.Tensor], 
                captions_dict: Dict[str, List[str]], 
                word2idx: Dict[str, int], 
                max_len: int = 22, 
                split: str = 'train',
                split_ratio: float = 0.9):
        """
        initialize dataset for training/validation
        
        Args:
            features_dict: dictionary mapping image_id to image features
            captions_dict: dictionary mapping image_id to list of captions
            word2idx: word to index mapping
            max_len: max caption length
            split: 'train' or 'val'
            split_ratio: ratio for train/val split (default: 0.9)
        """
        # ensure data consistency 
        # only use images with both features and captions
        common_ids = set(features_dict.keys()).intersection(set(captions_dict.keys()))
        print(f"dataset '{split}': {len(common_ids)} valid images (from {len(features_dict)} features, {len(captions_dict)} caption sets)")

        if len(common_ids) == 0:
            raise ValueError("no valid images found with both features and captions!")

        # all valid image IDs
        all_ids = list(common_ids)
        random.seed(42)  # for reproducibility
        random.shuffle(all_ids)

        # split train/val
        split_idx = int(split_ratio * len(all_ids))
        if split == 'train':
            self.image_names = all_ids[:split_idx]
        else:
            self.image_names = all_ids[split_idx:]

        self.features_dict = features_dict
        self.captions_dict = captions_dict
        self.word2idx = word2idx
        self.max_len = max_len

        print(f"Created {split} dataset with {len(self.image_names)} images")

    def __len__(self) -> int:
        """
        return the number of items in the dataset
        """
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get a single item from the dataset
        
        Args:
            idx: index of the item
            
        Returns:
            tuple of (feature, tokenized_caption)
        """
        img_name = self.image_names[idx]

        # double check safety check
        if img_name not in self.features_dict:
            raise KeyError(f"image {img_name} not found in features dictionary!")
        if img_name not in self.captions_dict:
            raise KeyError(f"image {img_name} not found in captions dictionary!")

        feature = self.features_dict[img_name]

        # for training, randomly choose one caption
        caption = random.choice(self.captions_dict[img_name]).split()

        # add < SOS > (start of sequence) and <EOS> (end of sequence) tokens
        caption = ["< SOS >"] + caption + ["<EOS>"]
        tokens = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in caption]
        tokens = tokens[:self.max_len]
        tokens += [self.word2idx["<PAD>"]] * (self.max_len - len(tokens))

        return feature, torch.tensor(tokens, dtype=torch.long)


class CaptionEvaluationDataset(Dataset):
    """
    dataset for evaluation with only image features
    
    Attributes:
        image_names: list of image IDs in the dataset
        features_dict: dictionary mapping image_id to image features
    """
    
    def __init__(self, features_dict):
        """
        Initialize the evaluation dataset.
        """
        self.image_names = list(features_dict.keys())
        self.features_dict = features_dict
        print(f"Created evaluation dataset with {len(self.image_names)} images")

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_name = self.image_names[idx]
        feature = self.features_dict[img_name]
        
        # Ensure feature is a tensor and has the correct shape
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature)
        
        return feature, img_name