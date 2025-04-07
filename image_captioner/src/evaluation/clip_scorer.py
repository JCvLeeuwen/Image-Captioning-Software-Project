import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from src.config import DEVICE
from src.models.decoder import Transformer_Decoder
from src.models.beam_search import beam_search_caption


class CLIPCalculator:
    """
    calculator for CLIP-based scores between images and captions
    
    this class uses the CLIP model to compute similarity scores between
    images and captions, which serves as a metric for caption quality
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        initialize the CLIP calculator
        
        Args:
            model_name: name of the CLIP model to use
        """
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def compute(self, reference_captions: Dict[str, List[str]],
                generated_captions: Dict[str, str],
                image_paths: Dict[str, str]) -> float:
        """
        Compute CLIP scores for generated captions against images.
        
        Args:
            reference_captions: Dictionary mapping image_id to list of reference captions
            generated_captions: Dictionary mapping image_id to generated caption
            image_paths: Dictionary mapping image_id to image path
            
        Returns:
            Mean CLIP score (normalized between 0 and 1)
        """
        import torch.nn.functional as F
        import math
        
        clip_scores = []
        common_ids = set(reference_captions.keys()) & set(generated_captions.keys()) & set(image_paths.keys())
        
        if not common_ids:
            print("No common image IDs found for CLIP evaluation!")
            return 0.0
            
        for img_id in common_ids:
            try:
                # Load image
                image = Image.open(image_paths[img_id]).convert("RGB")
                
                # Get generated caption
                caption = generated_captions[img_id]
                
                # Process inputs
                inputs = self.processor(text=[caption], images=image, return_tensors="pt", padding=True).to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Extract embeddings if available (newer CLIP models)
                    if hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds'):
                        image_embeds = outputs.image_embeds
                        text_embeds = outputs.text_embeds
                        
                        # Compute cosine similarity
                        similarity_score = F.cosine_similarity(image_embeds, text_embeds).item()
                    else:
                        # For older models, we need to get the features from the logits
                        # The logits are already the dot product of the normalized embeddings
                        logits = outputs.logits_per_image.cpu().numpy()[0][0]
                        
                        # Convert logits to cosine similarity (bounded between -1 and 1)
                        # This is an approximation - ideally extract embeddings as above
                        similarity_score = min(max(logits / 100.0, -1.0), 1.0)
                    
                    # Ensure value is in valid range for arccos
                    similarity_score = max(min(similarity_score, 1.0), -1.0)
                    
                    # Convert to angular similarity
                    angular_distance = math.acos(similarity_score) / math.pi
                    angular_similarity = 1 - angular_distance
                    
                    clip_scores.append(angular_similarity)
                
            except Exception as e:
                print(f"Error computing CLIP score for image {img_id}: {e}")
                
        if not clip_scores:
            return 0.0
            
        mean_clip_score = np.mean(clip_scores)
        return mean_clip_score
    
    def compute_reference_clip(self, reference_captions: Dict[str, List[str]], 
                            image_paths: Dict[str, str]) -> float:
        """
        compute CLIP scores for reference captions against images using the same method as for generated captions
        
        Args:
            reference_captions: dictionary mapping image_id to list of reference captions
            image_paths: dictionary mapping image_id to image path
            
        Returns:
            mean CLIP score for reference captions
        """
        import torch.nn.functional as F
        import math
        
        clip_scores = []
        common_ids = set(reference_captions.keys()) & set(image_paths.keys())
        
        if not common_ids:
            print("no common image IDs found for reference CLIP evaluation!")
            return 0.0
            
        for img_id in common_ids:
            try:
                # Load image
                image = Image.open(image_paths[img_id]).convert("RGB")
                
                # Use first reference caption
                caption = reference_captions[img_id][0]
                
                # Process inputs
                inputs = self.processor(text=[caption], images=image, return_tensors="pt", padding=True).to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Extract embeddings if available (newer CLIP models)
                    if hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds'):
                        image_embeds = outputs.image_embeds
                        text_embeds = outputs.text_embeds
                        
                        # Compute cosine similarity
                        similarity_score = F.cosine_similarity(image_embeds, text_embeds).item()
                    else:
                        # For older models, we need to get the features from the logits
                        # The logits are already the dot product of the normalized embeddings
                        logits = outputs.logits_per_image.cpu().numpy()[0][0]
                        
                        # Convert logits to cosine similarity (bounded between -1 and 1)
                        similarity_score = min(max(logits / 100.0, -1.0), 1.0)
                    
                    # Ensure value is in valid range for arccos
                    similarity_score = max(min(similarity_score, 1.0), -1.0)
                    
                    # Convert to angular similarity
                    angular_distance = math.acos(similarity_score) / math.pi
                    angular_similarity = 1 - angular_distance
                    
                    clip_scores.append(angular_similarity)
                
            except Exception as e:
                print(f"error computing reference CLIP score for image {img_id}: {e}")
                
        if not clip_scores:
            return 0.0
            
        mean_clip_score = np.mean(clip_scores)
        return mean_clip_score


def compute_clip_reward_loss(features: torch.Tensor, img_ids: List[str], 
                           decoder: Transformer_Decoder, project_features: torch.nn.Module,
                           word2idx: Dict[str, int], idx2word: Dict[int, str],
                           clip_calculator: CLIPCalculator,
                           reference_captions: Dict[str, List[str]], 
                           image_paths: Dict[str, str], device: str) -> Tuple[torch.Tensor, float]:
    """
    compute policy gradient loss using CLIP scores as rewards.

    Args:
        features: batch of image features
        img_ids: image IDs corresponding to the features
        decoder: decoder model
        project_features: projection layer
        word2idx: word to index mapping
        idx2word: index to word mapping
        clip_calculator: CLIP calculator instance
        reference_captions: dictionary of reference captions
        image_paths: dictionary of image paths
        device: device 

    Returns:
        tuple of (loss, mean_clip_score)
    """
    from src.models.decoder import sample_caption
    
    sampled_captions = {}
    log_probs = []
    valid_indices = []

    # generate captions with sampling to allow gradient flow
    for i, (feature, img_id) in enumerate(zip(features, img_ids)):
        img_id = str(img_id)

        # skip images without reference captions or paths
        if img_id not in reference_captions or img_id not in image_paths:
            continue

        # sample a caption and get log probability
        try:
            _, log_prob, caption = sample_caption(
                feature, decoder, project_features,
                word2idx, idx2word, device
            )

            sampled_captions[img_id] = caption
            log_probs.append(log_prob)
            valid_indices.append(i)
        except Exception as e:
            print(f"error sampling caption: {e}")
            continue

    # if no valid captions, return zero loss
    if not sampled_captions:
        return torch.tensor(0.0, device=device), 0.0

    # prep data for CLIP evaluation
    batch_references = {img_id: reference_captions[img_id]
                       for img_id in sampled_captions.keys()}
    batch_paths = {img_id: image_paths[img_id]
                  for img_id in sampled_captions.keys()}

    # CLIP scores
    try:
        clip_scores = clip_calculator.compute(
            batch_references, sampled_captions, batch_paths
        )

        rewards = clip_scores

        # convert to tensor
        rewards = torch.tensor(rewards, device=device)
    except Exception as e:
        print(f"Error calculating CLIP score: {e}")
        return torch.tensor(0.0, device=device), 0.0

    # calculate policy gradient loss
    # higher CLIP score (better CLIP score) = lower loss (hence the negative sign)
    policy_loss = -rewards * torch.stack(log_probs).mean()

    return policy_loss, clip_scores

import torch
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from src.config import DEVICE
from src.models.beam_search import beam_search_caption

def evaluate_model_with_clip_score(
    decoder: Transformer_Decoder, 
    project_features: torch.nn.Module, 
    eval_loader: torch.utils.data.DataLoader, 
    word2idx: Dict[str, int], 
    idx2word: Dict[int, str],
    clip_calculator: CLIPCalculator, 
    captions_file: str, 
    image_dir: str,
    train_image_names: Union[List[str], Set[str]], 
    val_image_names: Union[List[str], Set[str]], 
    max_eval_images: int = 100
) -> Tuple[float, Dict[str, str]]:
    """
    Evaluate image captioning model using CLIP score.
    
    Args:
        decoder: Image captioning decoder model
        project_features: Feature projection layer
        eval_loader: DataLoader for evaluation images
        word2idx: Mapping of words to indices
        idx2word: Mapping of indices to words
        clip_calculator: CLIP score calculator
        captions_file: Path to captions TSV file
        image_dir: Directory containing images
        train_image_names: List or set of training image names
        val_image_names: List or set of validation image names
        max_eval_images: Maximum number of images to evaluate
    
    Returns:
        Tuple of (CLIP score, generated captions dictionary)
    """
    import pandas as pd
    import os
    import random
    from tqdm import tqdm

    # Convert lists to sets if needed
    train_image_set = set(train_image_names)
    val_image_set = set(val_image_names)
    
    # Normalize image names by stripping file extensions for consistency
    all_image_names = set()
    for img_name in list(train_image_set) + list(val_image_set):
        # Strip extension if present
        if isinstance(img_name, str):
            base_name = img_name
            for ext in ['.jpg', '.jpeg', '.png']:
                if base_name.endswith(ext):
                    base_name = base_name[:-len(ext)]
            all_image_names.add(base_name)
        else:
            # If not a string, convert to string and add as is
            all_image_names.add(str(img_name))
    
    # Debug print to check available image names
    print(f"Total unique images in training/validation: {len(all_image_names)}")
    if len(all_image_names) < 5:
        print(f"WARN: Very few image names: {all_image_names}")
    
    # Load reference captions
    print("Loading reference captions...")
    try:
        df = pd.read_csv(captions_file, sep='\t', header=None, names=['image', 'caption'])
    except Exception as e:
        print(f"Error loading captions file: {e}")
        print(f"Attempting to load with different format...")
        try:
            df = pd.read_csv(captions_file, sep='\t')
        except Exception as e2:
            print(f"Still failed: {e2}")
            return 0.0, {}
    

    image_col = 'image' if 'image' in df.columns else 'image_id' if 'image_id' in df.columns else df.columns[0]
    caption_col = 'caption' if 'caption' in df.columns else df.columns[1]
    

    if df.shape[0] > 0:
        print(f"Sample caption entry: {df.iloc[0]}")
    
    # Prepare reference captions dictionary with normalized image names
    reference_captions = {}
    for _, row in df.iterrows():
        img_name = str(row[image_col])
        # Remove extension if present
        base_name = img_name
        for ext in ['.jpg', '.jpeg', '.png']:
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
        
        if base_name not in reference_captions:
            reference_captions[base_name] = []
        reference_captions[base_name].append(row[caption_col])
    
    print(f"Loaded {len(reference_captions)} unique images with reference captions")
    
    # Prepare image paths - try both with and without extensions
    image_paths = {}
    for img_name in all_image_names:
        # Try with different extensions
        found = False
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(image_dir, f"{img_name}{ext}")
            if os.path.exists(path):
                image_paths[img_name] = path
                found = True
                break
        
        # Try without adding extension (in case name already has extension)
        if not found:
            path = os.path.join(image_dir, img_name)
            if os.path.exists(path):
                image_paths[img_name] = path
    
    print(f"Found image paths for {len(image_paths)} images")
    
    # Find common images between reference captions and paths
    common_eval_images = set(reference_captions.keys()) & set(image_paths.keys())
    print(f"Common images with both references and paths: {len(common_eval_images)}")
    
    if len(common_eval_images) == 0:
        print("ERROR: No common images found for evaluation!")

        sample_refs = list(reference_captions.keys())[:5]
        sample_paths = list(image_paths.keys())[:5]

        return 0.0, {}
    
    # Limit number of images if needed
    if len(common_eval_images) > max_eval_images:
        print(f"Limiting evaluation to {max_eval_images} random images")
        random.seed(42)  # For reproducibility
        eval_image_names = set(random.sample(list(common_eval_images), max_eval_images))
    else:
        eval_image_names = common_eval_images
    
    # Generate captions
    generated_captions = {}
    decoder.eval()
    project_features.eval()
    
    print(f"Generating captions for {len(eval_image_names)} images...")
    
    with torch.no_grad():
        for batch_features, batch_img_names in tqdm(eval_loader, desc="Generating Captions"):
            # Move features to device
            batch_features = batch_features.to(DEVICE)
            
            # Generate captions for each image in the batch
            for feature, img_name in zip(batch_features, batch_img_names):
                # Normalize image name
                img_base_name = str(img_name)
                for ext in ['.jpg', '.jpeg', '.png']:
                    if img_base_name.endswith(ext):
                        img_base_name = img_base_name[:-len(ext)]
                
                # Skip if not in our evaluation set
                if img_base_name not in eval_image_names:
                    continue
                
                # Prepare feature for caption generation
                img_feature = feature.unsqueeze(0).to(DEVICE)
                
                try:
                    # Generate caption with beam search
                    caption = beam_search_caption(
                        img_feature, decoder, project_features,
                        word2idx, idx2word, DEVICE, 
                        beam_width=5, max_len=22
                    )
                    
                    # Debug print sample captions
                    if len(generated_captions) < 3:
                        print(f"Sample caption for {img_base_name}: {caption}")
                    
                    # Store non-empty captions
                    if caption and caption.strip():
                        generated_captions[img_base_name] = caption
                    else:
                        print(f"WARNING: Empty caption generated for {img_base_name}")
                
                except Exception as e:
                    print(f"Error generating caption for {img_base_name}: {e}")
    
    # Print generation statistics
    print(f"Generated {len(generated_captions)} captions")
    
    # Prepare for CLIP score calculation
    common_image_names = set(generated_captions.keys()) & set(reference_captions.keys()) & set(image_paths.keys())
    
    print(f"final common images for CLIP evaluation: {len(common_image_names)}")
    
    if len(common_image_names) == 0:
        print("ERROR: no common images found for CLIP score calculation!")
        return 0.0, {}
    
    # Filter dictionaries to common images
    filtered_generated = {img_name: generated_captions[img_name] for img_name in common_image_names}
    filtered_references = {img_name: reference_captions[img_name] for img_name in common_image_names}
    filtered_paths = {img_name: image_paths[img_name] for img_name in common_image_names}
    
    # Compute CLIP scores
    print("Computing CLIP scores...")
    generated_clip_score = clip_calculator.compute(
        filtered_references, 
        filtered_generated, 
        filtered_paths
    )
    
    reference_clip_score = clip_calculator.compute_reference_clip(
        filtered_references, 
        filtered_paths
    )
    
    print(f"CLIP Scores - Generated: {generated_clip_score:.4f}, Reference: {reference_clip_score:.4f}")
    
    return generated_clip_score, filtered_generated

def load_references(tsv_path: str, image_dir: str, 
                   filter_ids: Optional[Set[str]] = None) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    load reference captions and image paths from files
    
    Args:
        tsv_path:path to the captions TSV file
        image_dir: directory containing the images
        filter_ids: optional set of image IDs to filter by
        
    Returns:
        tuple of (reference_captions, image_paths)
    """
    import pandas as pd
    import os
    
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['image_id', 'caption'])
    
    # filter by ID 
    if filter_ids:
        df = df[df['image_id'].astype(str).isin(filter_ids)]
    
    # make reference captions dictionary
    reference_captions = {}
    for _, row in df.iterrows():
        img_id = str(row['image_id'])
        if img_id.endswith('.jpg'):
            img_id = img_id[:-4]
        
        if img_id not in reference_captions:
            reference_captions[img_id] = []
        reference_captions[img_id].append(row['caption'])
    
    # maked image paths dictionary
    image_paths = {}
    for img_id in (filter_ids if filter_ids else set(reference_captions.keys())):
        # try different extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            path = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(path):
                image_paths[img_id] = path
                break
        
        # try with the name itself 
        if img_id not in image_paths:
            path = os.path.join(image_dir, img_id)
            if os.path.exists(path):
                image_paths[img_id] = path
    
    return reference_captions, image_paths