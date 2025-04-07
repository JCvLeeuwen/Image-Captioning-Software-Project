import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPScorer:
    def __init__(self):
        """
        initialize the CLIP model 
        """
        # load CLIP model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def compute_similarity(self, image_path, caption):
        """
        compute similarity score between an image and a caption using CLIP
        
        Args:
            image_path (str): path to the image file
            caption (str): caption text to compare with the image
            
        Returns:
            float: similarity score 
        """
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            
            # Process image and text inputs
            inputs = self.processor(
                text=[caption],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # calculate image-text similarity score
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity_score = torch.nn.functional.softmax(logits_per_image, dim=1).squeeze().item()
                
            return similarity_score
            
        except Exception as e:
            print(f"Error computing CLIP score: {e}")
            return 0.0
    
    def score_batch(self, image_paths, captions):
        """
        Compute similarity scores for a batch of image-caption pairs.
        
        Args:
            image_paths (list): list of paths to image files
            captions (list): list of captions corresponding to the images
            
        Returns:
            list: list of similarity scores 
        """
        scores = []
        for img_path, caption in zip(image_paths, captions):
            scores.append(self.compute_similarity(img_path, caption))
        return scores